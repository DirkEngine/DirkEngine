//! Reference-counted asset handles.
//!
//! A [`Handle<T>`] is the public token granted to callers of
//! [`AssetRegistry::load_asset`]. It is cheaply cloneable, `Send`-able, and
//! ref-counted — the underlying asset data is freed only when **all** clones
//! of the handle have been dropped.
//!
//! [`AssetRegistry::load_asset`]: crate::AssetRegistry::load_asset

use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

use dirk_events::Dispatcher;
use parking_lot::Mutex;

use crate::{Asset, AssetHandle, Error, Result, events::InternalAssetUnloaded};

/// Identifies one loaded instance of an asset, including across unload/reload cycles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssetGeneration(u64);

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Private inner state shared by all clones of a [`Handle<T>`].
///
/// Wrapped in `Arc<Mutex<>>` so that:
/// - All [`Handle<T>`] clones share the same data without copying.
/// - The [`Drop`] impl fires [`InternalAssetUnloaded`] exactly once — when
///   the last [`Arc`] reference is released (i.e. when the last `Handle` clone
///   is dropped).
pub(crate) struct AssetRef<T: Asset> {
    /// The identifier of this asset; carried along so the drop event can
    /// include the handle for registry bookkeeping.
    pub(crate) asset_handle: AssetHandle,
    pub(crate) generation: AssetGeneration,

    /// The loaded asset data.
    ///
    /// - Starts as `Some(data)` after loading.
    /// - Becomes `None` in after the first [`Handle::take`] call (data is moved out).
    /// - Remains `Some` in after [`Handle::get`] call (data is cloned on consume).
    data: Option<T>,

    /// Dedicated dispatcher used in [`Drop::drop`] to fire the unload event.
    ///
    /// Each `AssetRef` owns its own dispatcher so the event is sent even
    /// when the [`AssetRegistry`] itself does not hold a reference to this
    /// particular asset.
    ///
    /// [`AssetRegistry`]: crate::AssetRegistry
    unload_dispatcher: Dispatcher<InternalAssetUnloaded>,
}

impl<T: Asset> AssetRef<T> {
    pub(crate) fn new(
        asset_handle: AssetHandle,
        data: T,
        unload_dispatcher: Dispatcher<InternalAssetUnloaded>,
    ) -> Self {
        Self {
            asset_handle,
            generation: AssetGeneration(
                NEXT_GENERATION
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                        value.checked_add(1)
                    })
                    .expect("asset generation counter exhausted"),
            ),
            data: Some(data),
            unload_dispatcher,
        }
    }
}

impl<T: Asset> Drop for AssetRef<T> {
    /// Fires [`InternalAssetUnloaded`] when the last [`Handle<T>`] is dropped.
    ///
    /// The [`AssetRegistry`] listens for this internal event in
    /// [`AssetRegistry::tick`] and re-emits the public [`AssetUnloaded`]
    /// event so that downstream systems (renderer, audio, …) can free their
    /// own resources.
    ///
    /// [`AssetRegistry`]: crate::AssetRegistry
    /// [`AssetRegistry::tick`]: crate::AssetRegistry::tick
    /// [`AssetUnloaded`]: crate::AssetUnloaded
    fn drop(&mut self) {
        self.unload_dispatcher.dispatch(InternalAssetUnloaded {
            handle: self.asset_handle.clone(),
            generation: self.generation,
        });
    }
}

/// A reference-counted, cheaply-cloneable public handle to a loaded asset.
///
/// Obtain one via [`AssetRegistry::load_asset`]. Cloning a handle does not
/// copy the underlying data — all clones share the same data inside an [`Arc`].
/// The asset's CPU memory is freed only when **all** clones are dropped.
///
/// # Using the data
///
/// Call [`take`] to take ownership of the handle's data. This
/// should be used when an asset's data is only needed by one system (for
/// example, the renderer would take the data & upload it to the GPU, no
/// other system needs it).
/// Subsequent calls to [`take`] or [`get`] will return a [`Error::AlreadyTaken`].
///
/// Call [`get`] to just clone the data.
///
/// Prefer [`take`] as it saves memory & has less performance cost.
///
/// # Sending across threads
///
/// [`Handle<T>`] is [`Send`]. You can send a handle to a worker thread
/// for background GPU upload and drop it there safely.
///
/// # Debug formatting
///
/// The `Debug` impl prints the raw asset path without locking the inner
/// `Mutex` for data access:
///
/// [`AssetRegistry::load_asset`]: crate::AssetRegistry::load_asset
/// [`take`]: Handle::take
/// [`get`]: Handle::get
#[derive(Clone)]
pub struct Handle<T: Asset>(Arc<Mutex<AssetRef<T>>>);

impl<T: Asset> Handle<T> {
    pub(crate) fn new(asset_ref: AssetRef<T>) -> Self {
        Self(Arc::new(Mutex::new(asset_ref)))
    }

    pub(crate) fn from_inner(inner: Arc<Mutex<AssetRef<T>>>) -> Self {
        Self(inner)
    }

    pub(crate) fn downgrade(&self) -> Weak<Mutex<AssetRef<T>>> {
        Arc::downgrade(&self.0)
    }

    /// Returns the loaded asset data.
    /// This function performs a full copy of the asset data.
    /// Beware of the performance cost.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyTaken`] in release builds if the data has
    /// already been taken by a prior call.
    pub fn get(&self) -> Result<T> {
        let inner = self.0.lock();
        inner.data.clone().ok_or(Error::AlreadyTaken)
    }

    /// Returns the loaded asset data.
    /// This function moves the data, performance cost is low.
    ///
    /// Prefer this over [`get`] as it helps save memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AlreadyTaken`] in release builds if the data has
    /// already been taken by a prior call.
    ///
    /// [`get`]: Handle::get
    pub fn take(&self) -> Result<T> {
        let mut inner = self.0.lock();
        inner.data.take().ok_or(Error::AlreadyTaken)
    }

    /// Returns the [`AssetHandle`] of the asset.
    pub fn handle(&self) -> AssetHandle {
        let inner = self.0.lock();
        inner.asset_handle.clone()
    }

    /// Returns the identity of this load, which changes after unload and reload.
    #[must_use]
    pub fn generation(&self) -> AssetGeneration {
        self.0.lock().generation
    }
}

impl<T: Asset> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("asset", &self.0.lock().asset_handle.raw())
            .finish_non_exhaustive()
    }
}
