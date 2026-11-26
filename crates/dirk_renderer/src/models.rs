//! The GPU side of model assets: their meshes and base color textures,
//! uploaded when the asset loads and dropped when it unloads.

use std::collections::{HashMap, VecDeque};

use dirk_assets::{AssetGeneration, AssetHandle, AssetLoaded};
use dirk_render_utils::Uploads;
use dirk_rhi::{
    AddressMode, Buffer, BufferAccess, BufferUsage, Extent3d, Format, Image, ImageDesc,
    ImageDimension, ImageUsage, Rhi, SampleCount, Sampler, SamplerDesc, SamplerIndex, ShaderStages,
    TextureIndex,
};
use dirk_shaders::types::MeshVertex;
use tracing::warn;

use crate::{Error, Result, Sampled};

/// Every model loaded on the GPU, keyed by asset.
pub struct ModelRegistry {
    models: HashMap<AssetHandle, Model>,
    /// White texture of primitives without a base color texture.
    fallback: Image,
    /// Samples every model texture.
    sampler: Sampler,
    /// Models loaded but not uploaded yet, oldest first.
    pending: VecDeque<AssetLoaded<dirk_assets::Model>>,
    /// Bytes a frame's uploads stage before [`update`](Self::update) leaves
    /// the models left to later frames.
    budget: u64,

    load_consumer: dirk_events::Consumer<AssetLoaded<dirk_assets::Model>>,
    unload_consumer: dirk_events::Consumer<dirk_assets::AssetUnloaded>,
}

/// A model on the GPU.
pub struct Model {
    /// The loaded instance of the asset this model was uploaded from.
    generation: AssetGeneration,
    /// Base color textures, which the primitives sample.
    _textures: Vec<Image>,
    primitives: Vec<Primitive>,
}

/// A mesh primitive: indexed triangles of one material.
pub struct Primitive {
    /// The vertices, pulled by the mesh shaders.
    pub vertices: Buffer,
    /// 32-bit indices.
    pub indices: Buffer,
    pub index_count: u32,
    /// The base color texture.
    pub texture: TextureIndex,
}

impl ModelRegistry {
    /// Creates an empty registry listening to asset loads on `events`, which
    /// uploads models until a frame's uploads staged `budget` bytes. Its
    /// fallback texture is uploaded with `uploads`.
    pub fn new(
        rhi: &Rhi,
        uploads: &mut Uploads<'_>,
        events: &dirk_events::EventManager,
        budget: u64,
    ) -> Result<Self> {
        let fallback = Model::upload_texture(uploads, "fallback texture", 1, 1, &[255; 4])?;
        let max_anisotropy = rhi.capabilities().limits.max_sampler_anisotropy;
        let sampler = rhi.create_sampler(&SamplerDesc {
            label: "model textures",
            address_modes: [AddressMode::Repeat; 3],
            max_anisotropy,
            ..SamplerDesc::default()
        })?;
        Ok(Self {
            models: HashMap::new(),
            fallback,
            sampler,
            pending: VecDeque::new(),
            budget,
            load_consumer: events.subscribe(),
            unload_consumer: events.subscribe(),
        })
    }

    /// Drops the models unloaded since the last update and uploads those
    /// loaded with `uploads`, oldest first, until they staged the budget:
    /// the rest wait for later frames, so a large scene loads over several
    /// frames instead of stalling one. Models that cannot be read or
    /// uploaded are skipped with a warning. Models unloaded before their
    /// upload are never uploaded.
    pub fn update(&mut self, uploads: &mut Uploads<'_>) {
        self.pending.extend(self.load_consumer.consume_all());
        for event in self.unload_consumer.consume_all() {
            self.pending.retain(|loaded| {
                loaded.handle.handle() != event.handle
                    || loaded.handle.generation() != event.generation
            });
            if self
                .models
                .get(&event.handle)
                .is_some_and(|model| model.generation == event.generation)
            {
                self.models.remove(&event.handle);
            }
        }
        while uploads.staged() < self.budget
            && let Some(event) = self.pending.pop_front()
        {
            let handle = event.handle.handle();
            let asset = match event.handle.get() {
                Ok(asset) => asset,
                Err(error) => {
                    warn!(model = %handle, %error, "skipping a model that cannot be read");
                    continue;
                }
            };
            match Model::upload(&asset, event.handle.generation(), uploads, &self.fallback) {
                Ok(model) => {
                    self.models.insert(handle, model);
                }
                Err(error) => {
                    warn!(model = %handle, %error, "skipping a model that cannot be uploaded");
                }
            }
        }
    }

    /// The primitives of the model of `handle`, if it is loaded.
    pub fn primitives(&self, handle: &AssetHandle) -> Option<&[Primitive]> {
        self.models
            .get(handle)
            .map(|model| model.primitives.as_slice())
    }

    /// The sampler of every model texture.
    pub fn sampler(&self) -> SamplerIndex {
        self.sampler.index()
    }
}

impl Model {
    /// Uploads `model`, loaded as `generation`: its meshes and the
    /// textures their materials use as base color, using `fallback` for
    /// primitives without one.
    fn upload(
        model: &dirk_assets::Model,
        generation: AssetGeneration,
        uploads: &mut Uploads<'_>,
        fallback: &Image,
    ) -> Result<Self> {
        let textures = model
            .images
            .iter()
            .map(|image| {
                let pixels = match image.format {
                    gltf::image::Format::R8G8B8 => &image
                        .pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|&[r, g, b]| [r, g, b, 255])
                        .collect(),
                    gltf::image::Format::R8G8B8A8 => &image.pixels,
                    format => return Err(Error::UnsupportedImageFormat(format)),
                };
                Self::upload_texture(uploads, "model texture", image.width, image.height, pixels)
            })
            .collect::<Result<Vec<_>>>()?;
        let fallback = fallback.texture()?;
        let base_colors = model
            .gltf
            .materials()
            .map(|material| {
                let Some(info) = material.pbr_metallic_roughness().base_color_texture() else {
                    return Ok(fallback);
                };
                let image = info.texture().source().index();
                Ok(textures
                    .get(image)
                    .ok_or(Error::TextureIndexOutOfRange(image))?
                    .texture()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut primitives = Vec::new();
        for primitive in model.gltf.meshes().flat_map(|mesh| mesh.primitives()) {
            let texture = primitive
                .material()
                .index()
                .and_then(|material| base_colors.get(material).copied())
                .unwrap_or(fallback);
            primitives.extend(Primitive::upload(
                &primitive,
                &model.buffers,
                texture,
                uploads,
            )?);
        }
        Ok(Self {
            generation,
            _textures: textures,
            primitives,
        })
    }

    /// Uploads the `width` by `height` sRGB RGBA `texels` as a texture
    /// fragment shaders sample, with its mips generated on the GPU.
    fn upload_texture(
        uploads: &mut Uploads<'_>,
        label: &str,
        width: u32,
        height: u32,
        texels: &[u8],
    ) -> Result<Image> {
        Ok(uploads.mipmapped_image(
            &ImageDesc {
                label,
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(width, height),
                format: Format::Rgba8Srgb,
                usage: ImageUsage::SAMPLED,
                // Down to one texel: one level per bit of the larger side.
                mip_levels: width.max(height).bit_width(),
                array_layers: 1,
                samples: SampleCount::X1,
                rest: None,
            },
            texels,
        )?)
    }
}

impl Primitive {
    /// Uploads `primitive`'s vertices and indices, or returns `None` for
    /// primitives without vertices or indices, which are not drawn.
    fn upload(
        primitive: &gltf::Primitive<'_>,
        buffers: &[gltf::buffer::Data],
        texture: TextureIndex,
        uploads: &mut Uploads<'_>,
    ) -> Result<Option<Self>> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        let positions: Vec<_> = reader.read_positions().map_or_default(Iterator::collect);
        let normals: Vec<_> = reader.read_normals().map_or_default(Iterator::collect);
        let uvs: Vec<_> = reader
            .read_tex_coords(0)
            .map_or_default(|uvs| uvs.into_f32().collect());
        let indices: Vec<_> = reader
            .read_indices()
            .map_or_default(|indices| indices.into_u32().collect());
        if positions.is_empty() || indices.is_empty() {
            warn!("skipping a glTF primitive without vertices or indices");
            return Ok(None);
        }
        let vertices: Vec<_> = positions
            .iter()
            .enumerate()
            .map(|(i, &position)| MeshVertex {
                position,
                normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                uv: uvs.get(i).copied().unwrap_or_default(),
            })
            .collect();
        Ok(Some(Self {
            vertices: uploads.buffer(
                "mesh vertices",
                BufferUsage::STORAGE,
                BufferAccess::ShaderRead(ShaderStages::VERTEX),
                bytemuck::cast_slice(&vertices),
            )?,
            indices: uploads.buffer(
                "mesh indices",
                BufferUsage::INDEX,
                BufferAccess::Index,
                bytemuck::cast_slice(&indices),
            )?,
            index_count: u32::try_from(indices.len()).map_err(|_| dirk_rhi::Error::Invalid {
                object: "mesh",
                label: "glTF primitive".to_owned(),
                reason: format!("{} indices exceed a 32-bit count", indices.len()),
            })?,
            texture,
        }))
    }
}
