//! Error types for the asset loading pipeline.
//!
//! All fallible operations in this crate return [`Result<T>`], which is an
//! alias for `std::result::Result<T, `[`Error`]`>`.

use thiserror::Error;

/// Convenience alias — every fallible function in this crate returns this.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors that can occur within the asset subsystem.
#[derive(Debug, Error)]
pub enum Error {
    /// A filesystem operation failed while scanning or reading asset files.
    ///
    /// This is the `From<std::io::Error>` conversion target, so the `?`
    /// operator on any [`std::io`] call will automatically wrap the error.
    ///
    /// # Common causes
    /// - The `ASSETS_PATH` environment variable points to a non-existent
    ///   directory.
    /// - A `.dirkasset` file was deleted between directory scan and read.
    /// - Insufficient filesystem permissions.
    #[error("IO error while loading assets: {0}")]
    IoError(#[from] std::io::Error),

    /// A `.dirkasset` file contained malformed JSON, or a config struct could
    /// not be serialised.
    ///
    /// This is the `From<serde_json::Error>` conversion target.
    ///
    /// # Common causes
    /// - Typo or missing field in a hand-written `.dirkasset` file.
    /// - A new required field was added to a config struct without updating
    ///   existing asset files.
    #[error("error during .dirkasset JSON serialisation: {0}")]
    SerialisationError(#[from] serde_json::Error),

    /// [`Handle::take`] was called on an asset that has already been taken.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dirk_assets::{Model, AssetType, AssetHandle};
    /// # async fn test() -> anyhow::Result<()> {
    /// # let workers = dirk_threads::WorkerPool::new("test");
    /// # let events = dirk_events::EventManager::new(workers.clone());
    /// # let registry = dirk_assets::AssetRegistry::init(&events, workers)?;
    /// # let asset_handle = AssetHandle::from_raw("", AssetType::Model);
    /// // release build only
    /// let handle = registry.load_asset::<Model>(&asset_handle).await?;
    /// let _data = handle.take()?;              // OK — data moved out
    /// let err = handle.get().unwrap_err(); // AlreadyConsumed
    /// # Ok(()) }
    /// ```
    ///
    /// [`Handle::take`]: crate::Handle::take
    #[error("asset data has already been consumed")]
    AlreadyTaken,

    /// The requested [`AssetHandle`] was not found in the registry.
    ///
    /// The inner `String` is the raw handle path (e.g.
    /// `"models/hero.dirkasset"`). Common causes are a mistyped handle, a
    /// `.dirkasset` file that failed validation and was pruned at startup, or
    /// a file added after the registry was initialised.
    ///
    /// [`AssetHandle`]: crate::AssetHandle
    #[error("asset not found in registry: {0}")]
    NotFound(String),

    /// A [`Handle`] of the wrong asset type was passed to
    /// [`AssetRegistry::load_asset`].
    ///
    /// For example, passing a handle whose `asset_type` is
    /// [`AssetType::Model`] to a `load_asset::<AudioClip>()` call would
    /// produce this error.
    ///
    /// The inner `String` is the raw handle path.
    ///
    /// [`Handle`]: crate::Handle
    /// [`AssetRegistry::load_asset`]: crate::AssetRegistry::load_asset
    /// [`AssetType::Model`]: crate::AssetType::Model
    #[error("asset {0} has wrong type for the requested load")]
    TypeMismatch(String),

    /// The asset-type-specific loader returned an error.
    ///
    /// Wraps an [`anyhow::Error`] so that loaders can attach rich context
    /// without having to define their own error enums. The source chain is
    /// preserved and printed by `{0}`.
    ///
    /// # Common causes
    /// - A glTF file referenced by a `ModelConfig` is missing or corrupt.
    /// - A required buffer or image embedded in a glTF is inaccessible.
    #[error("Asset load error: {0}")]
    AssetLoadError(#[source] anyhow::Error),
}
