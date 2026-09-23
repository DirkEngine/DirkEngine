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

/// A lightweight, cheaply-cloneable identifier for a single asset.
///
/// An `AssetHandle` consists of two parts:
/// - A **path string** relative to `ASSETS_PATH` (e.g.
///   `"models/hero.dirkasset"`).
/// - An [`AssetType`] discriminant embedded at load time from the asset's
///   `meta.asset_type` field, used to catch type mismatches before attempting
///   a load.
///
/// Handles are created internally by [`AssetRegistry`] during directory
/// scanning and are not normally constructed by hand. They are serialisable
/// so they can be stored in scene files or editor state.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct AssetHandle {
    /// Path relative to `ASSETS_PATH`, including the `.dirkasset` extension.
    handle: String,
    /// Runtime type tag, validated against the requested `T` in
    /// [`AssetRegistry::load_asset`].
    asset_type: AssetType,
    /// Runtime asset root; omitted from serialized identity.
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
    /// Validates the deserialised asset descriptor.
    ///
    /// After the registry scans and deserialises all `.dirkasset` files, it calls
    /// [`DirkAsset::validate`] on each entry. Assets that fail validation are
    /// silently pruned from the registry and a [`tracing::warn`] message is
    /// emitted. They will not appear in any subsequent [`AssetRegistry::load_asset`]
    /// call.
    ///
    /// Validation is intentionally kept cheap — it checks structural consistency
    /// (right config section present, referenced files exist on disk) rather than
    /// parsing full asset content. Full content errors surface later at
    /// [`Asset::load`] time.
    ///
    /// Returns `true` if the asset is structurally sound and should be kept
    /// in the registry; `false` if it should be pruned. A warning is logged
    /// for every failure condition.
    ///
    /// # Failure conditions
    ///
    /// - `asset_type` is [`AssetType::Unknown`] — the `.dirkasset` file is
    ///   missing or has an unrecognised `asset_type` field.
    /// - `asset_type` is [`AssetType::Model`] but the `"model"` JSON section
    ///   is absent.
    /// - The [`AssetConfig::validate`] failes on all configurations.
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

/// Central asset management system.
///
/// [`AssetRegistry`] is responsible for the full lifecycle of assets.
/// At initialisation, it scans `ASSETS_PATH` for all `.dirkasset` files.
/// It validates all the `.dirkasset` configurations & stores them in
/// memory (pruning invalid ones with a [`tracing::warn`].
///
/// Systems can then call [`load_asset`] to schedule asset IO on the engine's
/// worker pool. Awaiting it returns a [`Handle<T>`] and fires an
/// [`AssetLoaded<T>`] event.
///
/// Finally, [`AssetRegistry`] handles unloading assets when [`Handle<T>`] is
/// no longer referenced.
///
/// # Thread safety
///
/// [`AssetRegistry`] is cheaply cloneable. Every clone shares the same
/// descriptor cache, event consumers, and worker-pool-backed load path.
///
/// # Initialisation
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

    /// Weak references to currently live assets.
    loaded_assets: RwLock<HashMap<AssetHandle, CachedAsset>>,
}

struct CachedAsset {
    generation: AssetGeneration,
    weak: Box<dyn Any + Send + Sync>,
}

impl AssetRegistry {
    /// Creates and fully initialises the registry.
    ///
    /// This is the only constructor. It:
    ///
    /// 1. Registers internal event channels with `event_manager`.
    /// 2. Canonicalises `ASSETS_PATH` and walks it recursively.
    /// 3. Deserialises every `.dirkasset` file found.
    /// 4. Validates all descriptors; invalid ones are pruned and warned about.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IoError`] if `ASSETS_PATH` does not exist or cannot
    /// be read, or if any `.dirkasset` file cannot be opened.
    ///
    /// Returns [`Error::SerialisationError`] if a `.dirkasset` file contains
    /// malformed JSON.
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

    /// Loads an asset of type `T` on the engine worker pool.
    ///
    /// Await this method from a Tokio runtime, or keep the returned
    /// [`AssetLoad`] and call [`AssetLoad::try_poll`] periodically from
    /// synchronous code.
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
    pub fn load_asset<T: Asset>(&self, handle: &AssetHandle) -> AssetLoad<T> {
        let registry = self.clone();
        let handle = handle.clone();
        let task = self
            .inner
            .workers
            .spawn_blocking(move || registry.load_asset_immediate::<T>(handle));

        AssetLoad::new(task)
    }

    fn load_asset_immediate<T: Asset>(&self, handle: AssetHandle) -> Result<Handle<T>> {
        if handle.asset_type() != T::asset_type() {
            return Err(Error::TypeMismatch(handle.raw().to_owned()));
        }

        let canonical_handle = self
            .inner
            .assets
            .get_key_value(&handle)
            .map(|(key, _)| key.clone())
            .ok_or_else(|| Error::NotFound(handle.raw().to_owned()))?;

        let load_lock = self
            .inner
            .load_locks
            .lock()
            .entry(canonical_handle.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _load_guard = load_lock.lock();

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

    /// Resolves and deserialises the typed config for a given handle.
    ///
    /// Returns `None` if the handle is not in the registry or if the required
    /// config section is absent on the [`DirkAsset`].
    ///
    /// # Implementation note — the serde round-trip
    ///
    /// Because `DirkAsset` stores each config type as a concrete `Option<T>`
    /// (e.g. `Option<ModelConfig>`), but this method must return a generic
    /// `T::Config`, it uses a two-step serde conversion:
    ///
    /// 1. Serialise the concrete `Option<ModelConfig>` → `serde_json::Value`.
    /// 2. Deserialise the `Value` → `T::Config`.
    ///
    /// Both steps are guaranteed to succeed for any well-formed config because
    /// [`AssetConfig`] bounds `Serialize + DeserializeOwned`. The conversion
    /// happens once per [`load_asset`] call (not per frame), so the overhead
    /// is negligible.
    ///
    /// **TODO**: A future refactor could replace this with a type-erased config
    /// map to avoid the round-trip entirely.
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
                let relative_path =
                    path.strip_prefix(base)
                        .map(|p| p.to_path_buf())
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
