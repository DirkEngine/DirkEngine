//! Core asset abstractions: the [`Asset`] trait, [`AssetType`] discriminant,
//! and the [`AssetConfig`] marker trait.
//!
//! Every concrete asset type (models, textures, audio clips, …) must implement
//! [`Asset`]. The trait is intentionally minimal — it describes only how to
//! load data from disk and how to identify the asset type at runtime.
//!
//! # Implementing a new asset type
//!
//! 1. Define a **config struct** that derives [`AssetConfig`] and holds
//!    whatever fields the `.dirkasset` file needs for this type.
//! 2. Define a **data struct** that holds the fully-loaded, in-memory
//!    representation.
//! 3. Implement [`Asset`] for the data struct.
//! 4. Add a variant to [`AssetType`].
//! 5. Follow compile errors untile everything is ready.
//!
//! ```rust
//! use dirk_assets::{Asset, AssetConfig, AssetHandle, AssetType, Result};
//! use serde::{Deserialize, Serialize};
//!
//! /// Fields from the `.dirkasset` JSON for an audio clip.
//! #[derive(Serialize, Deserialize, Clone)]
//! pub struct AudioConfig {
//!     pub wav: String,
//! }
//!
//! # use dirk_assets::Metadata;
//! impl AssetConfig for AudioConfig {
//!     fn validate(&self, _: &Metadata) -> bool {
//!         // ...
//!         # false
//!     }
//! }
//!
//! /// In-memory audio data handed to the audio subsystem.
//! #[derive(Clone)]
//! pub struct Audio {
//!     pub samples: Vec<f32>,
//!     pub sample_rate: u32,
//! }
//!
//! impl Asset for Audio {
//!     type Config = AudioConfig;
//!
//!     fn load(config: &AudioConfig, handle: &AssetHandle) -> Result<Self> {
//!         let path = handle.dir().join(&config.wav);
//!         // … decode WAV from `path` …
//!         todo!()
//!     }
//!
//!     fn asset_type() -> AssetType {
//!     # /*
//!         AssetType::Audio
//!     # */
//!     # AssetType::Unknown
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize, de::DeserializeOwned};

mod model;
pub use model::*;

use crate::{AssetHandle, Metadata, Result};

/// Implemented by every concrete asset data type.
///
/// An [`Asset`] is a value that:
/// - can be **loaded from disk** given its configuration and an
///   [`AssetHandle`] (which carries the directory context),
/// - carries a static [`AssetType`] discriminant so the registry can validate
///   handle/type compatibility at runtime,
/// - is [`Clone`] + [`Send`] + `'static` so it can be easily duplicated and
///   freely moved across threads.
///
/// Implementations should be pure data containers. GPU uploads, sound
/// submissions, and other subsystem-side work happen *after* the asset is
/// consumed via [`Handle::take`], typically in response to an
/// [`AssetLoaded`] event.
///
/// [`AssetLoaded`]: crate::events::AssetLoaded
/// [`Handle::take`]: crate::Handle::take
pub trait Asset: Clone + Sized + Send + 'static {
    /// The configuration type that describes this asset inside a `.dirkasset`
    /// file.
    ///
    /// The config is deserialised from JSON at registry startup. Its fields
    /// drive the [`Asset::load`] implementation (e.g. a file path, encoding
    /// hints, LOD levels, …).
    type Config: AssetConfig;

    /// Reads the asset from disk and returns an owned, fully-initialised value.
    ///
    /// # Parameters
    ///
    /// - `config` — the deserialised config section from the `.dirkasset` file.
    /// - `handle` — carries the asset's path and directory so loaders can
    ///   resolve relative file references without needing global state.
    ///
    /// # Errors
    ///
    /// Implementations should wrap any foreign error in
    /// [`Error::AssetLoadError`] via [`anyhow::Context`]:
    ///
    /// ```rust
    /// use anyhow::Context as _;
    /// use dirk_assets::{Error, Result};
    ///
    /// # fn test() -> anyhow::Result<()> {
    /// # let path = "";
    /// let bytes = std::fs::read(&path)
    ///     .context("reading audio file")
    ///     .map_err(Error::AssetLoadError)?;
    /// # Ok(()) }
    /// ```
    ///
    /// [`Error::AssetLoadError`]: crate::Error::AssetLoadError
    fn load(config: &Self::Config, handle: &AssetHandle) -> Result<Self>;

    /// Returns the [`AssetType`] discriminant for this asset type.
    ///
    /// Used by the registry to ensure a handle's embedded type tag matches
    /// the concrete `T` being requested before attempting to load.
    fn asset_type() -> AssetType;
}

/// A compact, copy-able discriminant that identifies the *kind* of an asset.
///
/// Stored inside every [`AssetHandle`] so the registry can detect
/// handle/type mismatches at load time without needing type parameters.
///
/// # Serialisation
///
/// [`AssetType`] serialises as a JSON string:
///
/// ```rust
/// # use dirk_assets::AssetType;
/// let json = serde_json::to_string(&AssetType::Model).unwrap();
/// assert_eq!(json, r#""Model""#);
///
/// let round_tripped: AssetType = serde_json::from_str(&json).unwrap();
/// assert_eq!(round_tripped, AssetType::Model);
/// ```
///
/// # Default
///
/// The default variant is [`AssetType::Unknown`], which is intentionally
/// invalid — any asset whose `.dirkasset` file omits the `asset_type` field
/// will deserialise to `Unknown` and be rejected by validation:
///
/// ```rust
/// # use dirk_assets::AssetType;
/// assert_eq!(AssetType::default(), AssetType::Unknown);
/// ```
#[derive(Default, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum AssetType {
    /// Sentinel value. An asset with this type fails validation and is
    /// excluded from the registry. It is also the `serde` default so that
    /// `.dirkasset` files that omit `asset_type` are caught early rather than
    /// silently mapped to a concrete type.
    #[default]
    Unknown,

    /// A 3-D mesh asset backed by a glTF file.
    ///
    /// Loaded into [`model::Model`] by [`model::Model::load`].
    Model,
}

/// Marker trait for all asset configuration structs.
///
/// A type that implements [`AssetConfig`] can be embedded inside a
/// `.dirkasset` JSON file and deserialised by the registry at startup.
///
/// ```rust
/// use dirk_assets::{AssetConfig, Metadata};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize, Clone)]
/// pub struct MyConfig {
///     pub source_file: String,
/// }
///
/// impl AssetConfig for MyConfig {
///     fn validate(&self, meta: &Metadata) -> bool {
///         // ...
///         # false
///     }
/// }
/// ```
pub trait AssetConfig: Serialize + DeserializeOwned + Send + 'static {
    /// Validates the configuration for the asset.
    fn validate(&self, meta: &Metadata) -> bool;
}
