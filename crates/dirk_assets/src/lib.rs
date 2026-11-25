#![doc = include_str!("../README.md")]

#[cfg(test)]
mod tests;

mod errors;
use dirk_universe::Universe;
pub use errors::{Error, Result};

mod events;
pub use events::{AssetLoaded, AssetUnloaded};

mod assets;
pub use assets::*;

mod handle;
use handle::AssetRef;
pub use handle::{AssetGeneration, Handle};

use dirk_engine::{EngineBuilder, EngineHandle, EnginePlugin, Subsystem};
use dirk_events::{Consumer, Dispatcher, EventManager};
use dirk_threads::WorkerPool;
use parking_lot::{Mutex, RwLock};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Display,
    future::Future,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
};
use tokio::task::JoinHandle as TaskJoinHandle;
use tracing::warn;

use serde::{Deserialize, Serialize};

use crate::events::InternalAssetUnloaded;

/// Registers the asset registry as an engine subsystem.
pub struct AssetsPlugin;

impl EnginePlugin for AssetsPlugin {
    fn name(&self) -> &'static str {
        "assets"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.add_subsystem(|ctx| {
            let registry = AssetRegistry::init(ctx.events(), ctx.workers().clone())?;
            ctx.add_resource(registry.clone())?;
            Ok(AssetsSubsystem { registry })
        });
        Ok(())
    }
}

/// Runtime asset subsystem.
pub struct AssetsSubsystem {
    registry: AssetRegistry,
}

impl Subsystem for AssetsSubsystem {
    fn name(&self) -> &'static str {
        "assets"
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        _handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        self.registry.tick();
        Ok(())
    }
}

/// Extension used for all asset descriptor files.
pub(crate) const DIRK_ASSET_EXT: &str = "dirkasset";

/// Absolute path to the root of the asset tree, baked in at compile time by
/// `build.rs`.
pub(crate) const ASSETS_PATH: &str = std::env!("ASSETS_PATH");

/// A cheaply-cloneable, serialisable identifier for an asset: a path relative
/// to `ASSETS_PATH` (e.g. `"models/hero.dirkasset"`) and the [`AssetType`] from
/// its `meta.asset_type`, which catches type mismatches before a load.
///
/// Equality and hashing use only that path and type, so handles are unique
/// within one asset root: don't key a map with handles from registries made
/// with different roots. Handles from a registry remember its root for
/// [`AssetHandle::path`] and [`AssetHandle::dir`]; those from
/// [`AssetHandle::from_raw`] or deserialisation fall back to `ASSETS_PATH`
/// until [`AssetRegistry::resolve`] binds them.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct AssetHandle {
    /// Path relative to the asset root, including the `.dirkasset` extension.
    handle: String,
    /// Runtime type tag, validated against the requested `T` in
    /// [`AssetRegistry::load_asset`].
    asset_type: AssetType,
    /// Root of the issuing registry. Excluded from identity and serialisation.
    #[serde(skip)]
    root: Option<PathBuf>,
}

impl PartialEq for AssetHandle {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.asset_type == other.asset_type
    }
}

impl Eq for AssetHandle {}

impl Hash for AssetHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.asset_type.hash(state);
    }
}

impl Display for AssetHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.handle)
    }
}

impl AssetHandle {
    /// Constructs an `AssetHandle` from a raw path string and type tag.
    ///
    /// Intended primarily for tests and editor tooling. In production code,
    /// handles are issued by [`AssetRegistry`] during scanning.
    ///
    /// ```rust
    /// # use dirk_assets::{AssetHandle, AssetType};
    /// let handle = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
    /// assert_eq!(handle.raw(), "models/hero.dirkasset");
    /// assert_eq!(handle.asset_type(), AssetType::Model);
    /// ```
    pub fn from_raw(path: impl Into<String>, asset_type: AssetType) -> Self {
        Self {
            handle: path.into(),
            asset_type,
            root: None,
        }
    }

    /// Returns the **absolute** path to the `.dirkasset` file on disk.
    ///
    /// Unresolved handles resolve against `ASSETS_PATH`; see the
    /// [type-level docs](AssetHandle#identity-and-asset-roots).
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.root
            .as_deref()
            .unwrap_or_else(|| Path::new(ASSETS_PATH))
            .join(&self.handle)
    }

    /// Returns the **absolute** path to the directory containing the
    /// `.dirkasset` file.
    ///
    /// This is the base path used by asset loaders to resolve relative file
    /// references (e.g. a glTF path in a [`ModelConfig`]).
    ///
    /// Returns `ASSETS_PATH` if [`path`] has no parent component (which
    /// should not occur in practice for well-formed handles).
    ///
    /// [`path`]: AssetHandle::path
    #[must_use]
    pub fn dir(&self) -> PathBuf {
        self.path()
            .parent()
            .unwrap_or_else(|| std::path::Path::new(ASSETS_PATH))
            .to_path_buf()
    }

    /// Returns the file name component of the asset path (including the
    /// `.dirkasset` extension).
    ///
    /// ```rust
    /// # use dirk_assets::{AssetHandle, AssetType};
    /// let handle = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
    /// assert_eq!(handle.name(), "hero.dirkasset");
    /// ```
    #[must_use]
    pub fn name(&self) -> String {
        self.path()
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }

    /// Returns the [`AssetType`] discriminant embedded in this handle.
    ///
    /// ```rust
    /// # use dirk_assets::{AssetHandle, AssetType};
    /// let handle = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
    /// assert_eq!(handle.asset_type(), AssetType::Model);
    /// ```
    #[must_use]
    pub fn asset_type(&self) -> AssetType {
        self.asset_type
    }

    /// Returns the raw handle string (the path relative to `ASSETS_PATH`).
    ///
    /// ```rust
    /// # use dirk_assets::{AssetHandle, AssetType};
    /// let handle = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
    /// assert_eq!(handle.raw(), "models/hero.dirkasset");
    /// ```
    #[must_use]
    pub fn raw(&self) -> &str {
        &self.handle
    }
}

/// Metadata common to every `.dirkasset` file.
///
/// Serialised as the `"meta"` object in the JSON:
///
/// ```json
/// { "meta": { "asset_type": "Model" } }
/// ```
#[derive(Serialize, Deserialize)]
pub struct Metadata {
    /// The kind of asset this descriptor represents.
    pub asset_type: AssetType,

    /// Back-reference to the handle that owns this metadata.
    ///
    /// Not serialised. Populated by [`AssetRegistry::load`] immediately after
    /// deserialisation so that validation and loading code can access path
    /// helpers via [`AssetHandle`].
    #[serde(skip)]
    handle: AssetHandle,
}

/// The complete in-memory representation of a `.dirkasset` file.
///
/// Deserialised from JSON by [`AssetRegistry::load`]. Fields for asset types
/// that are not present in a given file are deserialised as `None`.
///
/// # Adding a new asset type
///
/// Add an `Option<YourConfig>` field here, add a corresponding JSON key to
/// asset files of that type, and wire the new arm into both
/// [`AssetRegistry::asset_config`] and [`DirkAsset::validate`].
///
/// [`DirkAsset::validate`]: validation
#[derive(Serialize, Deserialize)]
struct DirkAsset {
    /// Common fields (type tag, back-reference handle).
    pub meta: Metadata,
    /// Configuration for [`AssetType::Model`] assets. `None` for all other
    /// asset types.
    pub model: Option<ModelConfig>,
}

impl DirkAsset {
    /// Validates the deserialised descriptor, warning for each failure.
    ///
    /// The registry prunes assets that fail it. It checks structure only
    /// (right config section present, referenced files exist); content errors
    /// surface at [`Asset::load`]. Returns `true` to keep the asset.
    ///
    /// It fails when `asset_type` is [`AssetType::Unknown`], when it is
    /// [`AssetType::Model`] but the `"model"` section is absent, or when
    /// [`AssetConfig::validate`] fails.
    fn validate(&self) -> bool {
        match self.meta.asset_type {
            AssetType::Unknown => {
                warn!(
                    "asset {} cannot be of type Unknown, please specify an asset_type",
                    self.meta.handle.raw()
                );
                false
            }
            AssetType::Model => {
                if let Some(model_config) = &self.model {
                    model_config.validate(&self.meta)
                } else {
                    warn!(
                        "asset {} must specify a [model] section when asset_type = Model",
                        self.meta.handle.raw()
                    );
                    false
                }
            }
        }
    }
}

/// Central asset management system. At initialisation it scans `ASSETS_PATH`
/// for `.dirkasset` files and keeps the valid ones in memory.
/// [`load_asset`] schedules IO on the worker pool and returns a [`Handle<T>`],
/// firing an [`AssetLoaded<T>`]; the registry unloads an asset when its last
/// handle drops. It is cheaply cloneable: clones share one descriptor cache
/// and load path.
///
/// ```rust
/// use dirk_assets::AssetRegistry;
///
/// # fn test() -> anyhow::Result<()> {
/// # let workers = dirk_threads::WorkerPool::new("test");
/// # let events = dirk_events::EventManager::new(workers.clone());
/// let registry = AssetRegistry::init(&events, workers)?;
/// # Ok(()) }
/// ```
///
/// # Per-frame update
///
/// ```rust
/// # fn test() -> anyhow::Result<()> {
/// # let workers = dirk_threads::WorkerPool::new("test");
/// # let events = dirk_events::EventManager::new(workers.clone());
/// # let registry = dirk_assets::AssetRegistry::init(&events, workers).unwrap();
/// // Must be called once per frame.
/// registry.tick();
/// # Ok(()) }
/// ```
///
/// [`load_asset`]: AssetRegistry::load_asset
/// [`tick`]: AssetRegistry::tick
/// [`AssetLoaded<T>`]: events::AssetLoaded
#[derive(Clone)]
pub struct AssetRegistry {
    inner: Arc<AssetRegistryInner>,
}

/// A pending asset load running on the engine worker pool.
///
/// This is returned immediately by [`AssetRegistry::load_asset`]. In async code
/// you can await it directly. In synchronous, frame-driven code, call
/// [`try_poll`] periodically to check whether the worker task has completed
/// without blocking the current thread.
///
/// [`try_poll`]: AssetLoad::try_poll
pub struct AssetLoad<T: Asset> {
    task: Option<TaskJoinHandle<Result<Handle<T>>>>,
}

impl<T: Asset> AssetLoad<T> {
    fn new(task: TaskJoinHandle<Result<Handle<T>>>) -> Self {
        Self { task: Some(task) }
    }

    /// Polls this load once with a no-op waker.
    ///
    /// Returns `None` while the worker task is still running. Returns `Some`
    /// exactly once when the asset load completes; the result is consumed by
    /// this call.
    pub fn try_poll(&mut self) -> Option<Result<Handle<T>>> {
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        match Pin::new(self).poll(&mut context) {
            Poll::Ready(result) => Some(result),
            Poll::Pending => None,
        }
    }

    /// Returns `true` if this load has already yielded its result.
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.task.is_none()
    }
}

impl<T: Asset> Unpin for AssetLoad<T> {}

impl<T: Asset> Future for AssetLoad<T> {
    type Output = Result<Handle<T>>;

    #[expect(
        clippy::panic,
        reason = "Polling a future after it completed is a caller bug, as with std's futures."
    )]
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(task) = self.task.as_mut() else {
            panic!("AssetLoad polled after completion");
        };

        match Pin::new(task).poll(context) {
            Poll::Ready(result) => {
                self.task = None;
                Poll::Ready(match result {
                    Ok(result) => result,
                    Err(err) => Err(Error::AssetLoadError(anyhow::Error::new(err))),
                })
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: Asset> std::fmt::Debug for AssetLoad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetLoad")
            .field("pending", &self.task.is_some())
            .finish()
    }
}

struct AssetRegistryInner {
    /// All validated asset descriptors, keyed by their handle.
    assets: HashMap<AssetHandle, DirkAsset>,

    /// Shared event bus used to clone dispatchers for new asset types.
    event_manager: EventManager,

    /// Worker pool used for blocking asset IO and decoding work.
    workers: WorkerPool,

    /// Receives [`InternalAssetUnloaded`] from every live `AssetRef` when its
    /// ref-count reaches zero.
    internal_unload_consumer: Mutex<Consumer<InternalAssetUnloaded>>,

    /// Emits the public [`AssetUnloaded`] event consumed by e.g. the renderer.
    unload_dispatcher: Dispatcher<AssetUnloaded>,

    /// One [`Dispatcher<AssetLoaded<T>>`] per concrete asset type `T`,
    /// keyed by `TypeId::of::<T>()`.
    ///
    /// Lazily populated on the first [`load_asset::<T>`] call for each `T`.
    ///
    /// [`load_asset::<T>`]: AssetRegistry::load_asset
    load_dispatchers: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,

    /// One lock per descriptor, so same-key loads share work without holding
    /// a registry-wide lock across blocking disk IO.
    load_locks: Mutex<HashMap<AssetHandle, Arc<Mutex<()>>>>,

    /// Notified when a load waits for another load of the same asset, so a
    /// test can tell the loads overlap.
    #[cfg(test)]
    load_waits: tokio::sync::Notify,

    /// Weak references to currently live assets.
    loaded_assets: RwLock<HashMap<AssetHandle, CachedAsset>>,
}

struct CachedAsset {
    generation: AssetGeneration,
    weak: Box<dyn Any + Send + Sync>,
}

impl AssetRegistry {
    /// Creates the registry: registers its event channels with `event_manager`,
    /// walks and deserialises every `.dirkasset` under `ASSETS_PATH`, and
    /// prunes the invalid ones.
    ///
    /// # Errors
    ///
    /// [`Error::IoError`] if `ASSETS_PATH` or a `.dirkasset` file cannot be
    /// read, [`Error::SerialisationError`] if one holds malformed JSON.
    ///
    /// [`Error::IoError`]: crate::Error::IoError
    /// [`Error::SerialisationError`]: crate::Error::SerialisationError
    pub fn init(event_manager: &EventManager, workers: WorkerPool) -> Result<Self> {
        Self::init_at(event_manager, workers, ASSETS_PATH)
    }

    /// Creates a registry with an explicit asset root, useful for isolated
    /// projects and tests that do not use the build-time default path.
    ///
    /// # Errors
    /// Returns an error if the root or any descriptor cannot be read or parsed.
    pub fn init_at(
        event_manager: &EventManager,
        workers: WorkerPool,
        root: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut inner = AssetRegistryInner {
            assets: HashMap::new(),

            unload_dispatcher: event_manager.register(),
            internal_unload_consumer: Mutex::new(event_manager.subscribe()),
            event_manager: event_manager.clone(),
            workers,

            load_dispatchers: RwLock::new(HashMap::new()),
            load_locks: Mutex::new(HashMap::new()),
            #[cfg(test)]
            load_waits: tokio::sync::Notify::new(),
            loaded_assets: RwLock::new(HashMap::new()),
        };

        let assets_path = root.as_ref().canonicalize()?;
        inner.load(&assets_path, &assets_path)?;
        inner.validate();

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Processes deferred asset-unload notifications and emits public events.
    ///
    /// Must be called **exactly once per frame**. Skipping this call means
    /// [`AssetUnloaded`] events are never delivered, causing potential memory
    /// leaks (e.g. the renderer cannot free GPU resources).
    pub fn tick(&self) {
        let unloaded: Vec<_> = self
            .inner
            .internal_unload_consumer
            .lock()
            .consume_all()
            .collect();

        for InternalAssetUnloaded { handle, generation } in unloaded {
            self.clear_loaded_asset(&handle, generation);
            self.inner
                .unload_dispatcher
                .dispatch(AssetUnloaded { handle, generation });
        }
    }

    /// Loads an asset of type `T` on the engine worker pool. Await it, or call
    /// [`AssetLoad::try_poll`] periodically from synchronous code.
    ///
    /// # Errors
    ///
    /// | Condition | Error |
    /// |-----------|-------|
    /// | `handle.asset_type() != T::asset_type()` | [`Error::TypeMismatch`] |
    /// | Handle not found in registry | [`Error::NotFound`] |
    /// | Asset-type-specific load failure | [`Error::AssetLoadError`] |
    /// | Worker task cancellation or panic | [`Error::AssetLoadError`] |
    ///
    #[must_use]
    pub fn load_asset<T: Asset>(&self, handle: &AssetHandle) -> AssetLoad<T> {
        let registry = self.clone();
        let handle = handle.clone();
        let task = self
            .inner
            .workers
            .spawn_blocking(move || registry.load_asset_immediate::<T>(&handle));

        AssetLoad::new(task)
    }

    fn load_asset_immediate<T: Asset>(&self, handle: &AssetHandle) -> Result<Handle<T>> {
        if handle.asset_type() != T::asset_type() {
            return Err(Error::TypeMismatch(handle.raw().to_owned()));
        }

        let canonical_handle = self
            .resolve(handle)
            .ok_or_else(|| Error::NotFound(handle.raw().to_owned()))?;

        let load_lock = self
            .inner
            .load_locks
            .lock()
            .entry(canonical_handle.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _load_guard = load_lock.try_lock().unwrap_or_else(|| {
            #[cfg(test)]
            self.inner.load_waits.notify_one();
            load_lock.lock()
        });

        if let Some(typed_handle) = self.cached_handle::<T>(&canonical_handle) {
            return Ok(typed_handle);
        }

        let config = self
            .asset_config::<T>(&canonical_handle)
            .ok_or_else(|| Error::NotFound(handle.raw().to_owned()))?;

        let typed_handle = Handle::new(AssetRef::new(
            canonical_handle.clone(),
            T::load(&config, &canonical_handle)?,
            self.inner.event_manager.register(),
        ));

        self.cache_handle(&typed_handle);
        self.dispatch_loaded(typed_handle.clone());
        Ok(typed_handle)
    }

    /// Returns this registry's handle for `handle`, bound to the registry root.
    ///
    /// Use this for handles built with [`AssetHandle::from_raw`] or
    /// deserialised from scene files before calling [`AssetHandle::path`] or
    /// [`AssetHandle::dir`]. Returns `None` if the registry has no valid
    /// descriptor with the same path and type.
    #[must_use]
    pub fn resolve(&self, handle: &AssetHandle) -> Option<AssetHandle> {
        self.inner
            .assets
            .get_key_value(handle)
            .map(|(key, _)| key.clone())
    }

    fn cached_handle<T: Asset>(&self, handle: &AssetHandle) -> Option<Handle<T>> {
        let cached = self.inner.loaded_assets.read();
        let weak = cached
            .get(handle)?
            .weak
            .downcast_ref::<Weak<Mutex<AssetRef<T>>>>()?;

        weak.upgrade().map(Handle::from_inner)
    }

    fn cache_handle<T: Asset>(&self, handle: &Handle<T>) {
        self.inner.loaded_assets.write().insert(
            handle.handle(),
            CachedAsset {
                generation: handle.generation(),
                weak: Box::new(handle.downgrade()),
            },
        );
    }

    fn clear_loaded_asset(&self, handle: &AssetHandle, generation: AssetGeneration) {
        let mut cached = self.inner.loaded_assets.write();
        if cached
            .get(handle)
            .is_some_and(|entry| entry.generation == generation)
        {
            cached.remove(handle);
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "Dispatchers are keyed by the TypeId of the event they dispatch."
    )]
    fn dispatch_loaded<T: Asset>(&self, handle: Handle<T>) {
        let type_id = TypeId::of::<T>();
        let mut dispatchers = self.inner.load_dispatchers.write();
        let dispatcher = dispatchers
            .entry(type_id)
            .or_insert_with(|| Box::new(self.inner.event_manager.register::<AssetLoaded<T>>()))
            .downcast_ref::<Dispatcher<AssetLoaded<T>>>()
            .expect("dispatcher type invariant violated: TypeId key must match Dispatcher<T>");

        dispatcher.dispatch(AssetLoaded { handle });
    }

    /// Resolves and deserialises the typed config for a handle, or `None` if
    /// the handle is not in the registry or its config section is absent.
    ///
    /// Each config is a concrete `Option<T>` on [`DirkAsset`], so it round-trips
    /// through `serde_json::Value` to become `T::Config`. That always succeeds
    /// for well-formed configs, since [`AssetConfig`] bounds
    /// `Serialize + DeserializeOwned`.
    ///
    /// [`load_asset`]: AssetRegistry::load_asset
    /// [`AssetConfig`]: assets::AssetConfig
    fn asset_config<T: Asset>(&self, handle: &AssetHandle) -> Option<T::Config> {
        let asset = self.inner.assets.get(handle)?;

        let raw = match T::asset_type() {
            AssetType::Unknown => return None,
            AssetType::Model => serde_json::to_value(asset.model.as_ref()?).ok()?,
        };

        serde_json::from_value(raw).ok()
    }
}

impl AssetRegistryInner {
    /// Recursively scans `dir` for `.dirkasset` files and deserialises them.
    ///
    /// `base` is the `ASSETS_PATH` root, used to compute relative paths for
    /// handle strings. All handles are stored as paths relative to `base` so
    /// that the registry is portable across machines.
    fn load(&mut self, base: &Path, dir: &Path) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path().canonicalize()?;
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                self.load(base, &path)?;
            } else if metadata.is_file()
                && path.extension().and_then(|e| e.to_str()) == Some(DIRK_ASSET_EXT)
            {
                let relative_path = path
                    .strip_prefix(base)
                    .map(std::path::Path::to_path_buf)
                    .map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "path '{}' is not relative to base '{}'",
                                path.display(),
                                base.display()
                            ),
                        )
                    })?;

                let data = std::fs::read(path)?;
                let mut config: DirkAsset = serde_json::from_slice(&data)?;

                let handle = AssetHandle {
                    handle: relative_path.display().to_string(),
                    asset_type: config.meta.asset_type,
                    root: Some(base.to_path_buf()),
                };
                config.meta.handle = handle.clone();

                self.assets.insert(handle, config);
            }
        }
        Ok(())
    }

    /// Prunes all invalid assets from the registry in-place.
    ///
    /// Delegates per-asset validation to [`DirkAsset::validate`]; anything
    /// that returns `false` is removed. Warnings are emitted inside
    /// `validate` so this method produces no output of its own.
    fn validate(&mut self) {
        self.assets.retain(|_, conf| conf.validate());
    }
}
