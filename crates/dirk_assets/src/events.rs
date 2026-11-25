//! Public and internal events emitted by the asset subsystem.

use crate::{Asset, AssetGeneration, AssetHandle, Handle};
use dirk_events::Event;

/// **Internal** event fired by an [`AssetRef`] when its owning [`Handle`]
/// ref-count drops to zero.
///
/// This event is private to the crate. [`AssetRegistry::tick`] consumes it
/// and re-emits the public [`AssetUnloaded`] event, giving the registry a
/// chance to do any internal bookkeeping before notifying external systems.
///
/// Carries both the path and the particular load being released.
///
/// [`AssetRef`]: crate::handle::AssetRef
/// [`AssetRegistry::tick`]: crate::AssetRegistry::tick
#[derive(Event, Clone, Debug)]
#[event("unload asset {handle}")]
pub(crate) struct InternalAssetUnloaded {
    pub handle: AssetHandle,
    pub generation: AssetGeneration,
}

/// Fired by [`AssetRegistry`] when an asset of type `T` has been fully loaded;
/// the enclosed [`Handle`] gives access to the data via [`Handle::take`].
///
/// Subscriptions are type-specific: a consumer of `AssetLoaded<Model>` does
/// not receive other asset types.
///
/// # Example
///
/// ```rust
/// use dirk_assets::{AssetLoaded, Handle, Model};
/// use dirk_events::Consumer;
///
/// # let workers = dirk_threads::WorkerPool::new("test");
/// # let events = dirk_events::EventManager::new(workers);
/// let mut consumer: Consumer<AssetLoaded<Model>> = events.subscribe();
///
/// // Once per frame:
/// for event in consumer.consume_all() {
///     let model_data = event.handle.take().expect("should not be consumed yet");
///     // ... gpu.upload_model(model_data);
/// }
/// ```
///
/// [`AssetRegistry`]: crate::AssetRegistry
/// [`Handle::take`]: crate::Handle::take
/// [`Model`]: crate::Model
#[derive(Event, Clone, Debug)]
#[event("asset {handle:?} unloaded")]
pub struct AssetLoaded<T: Asset> {
    /// A cloneable handle to the loaded asset.
    ///
    /// Call [`Handle::take`] to take ownership of the underlying data.
    /// In release builds the data can only be taken once; subsequent
    /// calls on any clone of this handle return [`Error::AlreadyTaken`].
    ///
    /// [`Handle::take`]: crate::Handle::take
    /// [`Error::AlreadyTaken`]: crate::Error::AlreadyTaken
    pub handle: Handle<T>,
}

/// Fired by [`AssetRegistry`] when an asset's CPU-side data has been freed
/// (the last [`Handle`] was dropped), so consumers can release what they
/// created on its [`AssetLoaded`]. It carries only the [`AssetHandle`].
///
/// It is dispatched from [`AssetRegistry::tick`], so it arrives at most one
/// frame after the last [`Handle`] drops.
///
/// # Example
///
/// ```rust
/// use dirk_assets::AssetUnloaded;
/// use dirk_events::Consumer;
///
/// # let workers = dirk_threads::WorkerPool::new("test");
/// # let events = dirk_events::EventManager::new(workers);
/// let mut consumer: Consumer<AssetUnloaded> = events.subscribe();
///
/// // Once per frame:
/// for AssetUnloaded { handle, generation } in consumer.consume_all() {
///     // Check that generation is the one currently uploaded for handle.
///     // ... destroy the GPU model only when its generation matches
/// }
/// ```
///
/// [`AssetRegistry`]: crate::AssetRegistry
/// [`AssetRegistry::tick`]: crate::AssetRegistry::tick
/// [`Handle`]: crate::Handle
#[derive(Event, Clone, Debug)]
#[event("asset {handle} unloaded")]
pub struct AssetUnloaded {
    /// The identifier of the asset that was unloaded.
    ///
    /// Use this to correlate with the handle stored when the corresponding
    /// [`AssetLoaded`] event was received.
    pub handle: AssetHandle,
    /// Identifies the particular loaded instance that was released. A newer
    /// load of the same path must not be removed by this event.
    pub generation: AssetGeneration,
}
