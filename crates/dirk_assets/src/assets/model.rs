//! Model asset: a `.dirkasset` file bound to a glTF file, read into memory as
//! a [`Model`] with its buffers and images. The renderer takes it with
//! [`Handle::take`] on [`AssetLoaded<Model>`] and frees its GPU side on
//! [`AssetUnloaded`].
//!
//! ```json
//! {
//!   "meta": { "asset_type": "Model" },
//!   "model": {
//!     "gltf": "hero.gltf"
//!   }
//! }
//! ```
//!
//! The `gltf` path is relative to the directory of the `.dirkasset` file.
//!
//! [`AssetLoaded<Model>`]: crate::events::AssetLoaded
//! [`AssetUnloaded`]: crate::AssetUnloaded
//! [`Handle::take`]: crate::Handle::take

use crate::{Asset, AssetHandle, AssetType, Error, Metadata, Result, assets::AssetConfig};

use serde::{Deserialize, Serialize};

use anyhow::Context;
use tracing::warn;

/// Configuration for a model asset, deserialised from the `"model"` section
/// of a `.dirkasset` file.
///
/// # Serialisation
///
/// ```rust
/// # use dirk_assets::ModelConfig;
/// let json = r#"{"gltf":"meshes/hero.gltf"}"#;
/// let config: ModelConfig = serde_json::from_str(json).unwrap();
/// assert_eq!(config.gltf, "meshes/hero.gltf");
///
/// let round_tripped = serde_json::to_string(&config).unwrap();
/// assert_eq!(round_tripped, json);
/// ```
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    /// Path to the `.gltf` (or `.glb`) file, **relative to the directory**
    /// that contains the `.dirkasset` file.
    ///
    /// Example values: `"hero.gltf"`, `"../shared/cube.glb"`.
    pub gltf: String,
}

impl AssetConfig for ModelConfig {
    /// Validates that the glTF file actually exists on disk.
    ///
    /// Resolves the path as `handle.dir().join(&self.gltf)` — i.e. relative
    /// to the `.dirkasset` file's own directory — and checks [`Path::exists`].
    ///
    /// Logs a warning and returns `false` if the file is absent.
    ///
    /// [`Path::exists`]: std::path::Path::exists
    fn validate(&self, meta: &Metadata) -> bool {
        let path = meta.handle.dir().join(&self.gltf);
        if !path.exists() {
            warn!(
                "asset {}: glTF file not found at '{}'",
                meta.handle.raw(),
                path.display()
            );
            return false;
        }
        true
    }
}

/// Raw glTF data for a model asset, ready for GPU upload.
///
/// [`Model`] is a thin wrapper around the three components that the
/// [`gltf`] crate returns from [`gltf::import`]:
///
/// # Cloning
///
/// [`Clone`] is derived to satisfy the [`Asset`] bound, but cloning a
/// [`Model`] duplicates potentially large buffer and image vecs. Prefer
/// consuming and uploading over repeated clones.
#[derive(Debug, Clone)]
pub struct Model {
    /// The parsed glTF document describing the scene hierarchy, meshes,
    /// materials, animations, and skins.
    pub gltf: gltf::Document,

    /// Decoded texture image data (RGBA pixels, dimensions, format) for all
    /// images referenced by the glTF document.
    ///
    /// Indexed by the `image.index()` of any [`gltf::Image`] in the
    /// document.
    pub images: Vec<gltf::image::Data>,

    /// Binary buffer blobs (vertex data, index data, animation keyframes,
    /// …) referenced by accessors in [`Model::gltf`].
    ///
    /// Indexed by the `buffer.index()` of any [`gltf::Buffer`] view.
    pub buffers: Vec<gltf::buffer::Data>,
}

impl Asset for Model {
    type Config = ModelConfig;

    /// Loads a glTF model from disk.
    ///
    /// Resolves the glTF path relative to the asset directory
    /// (`handle.dir().join(&config.gltf)`), then delegates to
    /// [`gltf::import`] which reads the document, decodes all buffers, and
    /// decodes all referenced images in one pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AssetLoadError`] if:
    /// - The resolved path does not exist.
    /// - The file is not valid glTF/GLB.
    /// - A referenced external buffer or image cannot be read.
    fn load(config: &Self::Config, handle: &AssetHandle) -> Result<Self> {
        let path = handle.dir().join(&config.gltf);
        let (gltf, buffers, images) = gltf::import(path)
            .context("loading glTF model")
            .map_err(Error::AssetLoadError)?;

        Ok(Self {
            gltf,
            images,
            buffers,
        })
    }

    fn asset_type() -> AssetType {
        AssetType::Model
    }
}
