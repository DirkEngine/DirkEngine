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
        let fallback = Mips::generate(1, 1, vec![255; 4]).upload(uploads, "fallback texture")?;
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
    /// frames instead of stalling one. Models unloaded before their upload
    /// are never uploaded.
    pub fn update(&mut self, uploads: &mut Uploads<'_>) -> Result<()> {
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
            let model = Model::upload(
                &event.handle.get()?,
                event.handle.generation(),
                uploads,
                &self.fallback,
            )?;
            self.models.insert(event.handle.handle(), model);
        }
        Ok(())
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
                    gltf::image::Format::R8G8B8 => image
                        .pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|&[r, g, b]| [r, g, b, 255])
                        .collect(),
                    gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
                    format => return Err(Error::UnsupportedImageFormat(format)),
                };
                Mips::generate(image.width, image.height, pixels).upload(uploads, "model texture")
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

/// The mip chain of an sRGB RGBA image, down to one texel.
struct Mips {
    width: u32,
    height: u32,
    /// Texels of each level, from the full image down.
    levels: Vec<Vec<u8>>,
}

impl Mips {
    /// Generates the chain of the `width` by `height` image `pixels`, each
    /// level averaging 2x2 texels of the previous one in linear space.
    fn generate(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        let decode: Vec<f32> = (0..=u8::MAX)
            .map(|value| Self::linear_from_srgb(f32::from(value) / 255.0))
            .collect();
        let mut levels = vec![pixels];
        let (full_width, full_height) = (width, height);
        let (mut width, mut height) = (width as usize, height as usize);
        while width > 1 || height > 1 {
            let previous = levels.last().map_or(&[][..], Vec::as_slice);
            let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
            let mut next = Vec::with_capacity(next_width * next_height * 4);
            for y in 0..next_height {
                for x in 0..next_width {
                    let texels = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| {
                        let texel =
                            (2 * y + dy).min(height - 1) * width + (2 * x + dx).min(width - 1);
                        &previous[texel * 4..texel * 4 + 4]
                    });
                    for channel in 0..4 {
                        let sum: f32 = texels
                            .iter()
                            .map(|texel| match channel {
                                3 => f32::from(texel[3]) / 255.0,
                                _ => decode[usize::from(texel[channel])],
                            })
                            .sum();
                        let average = sum / 4.0;
                        let encoded = if channel == 3 {
                            average
                        } else {
                            Self::srgb_from_linear(average)
                        };
                        // Rounded and clamped to `0.0..=255.0` first.
                        #[expect(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            reason = "the value is rounded and clamped to a byte"
                        )]
                        next.push((encoded * 255.0).round().clamp(0.0, 255.0) as u8);
                    }
                }
            }
            levels.push(next);
            (width, height) = (next_width, next_height);
        }
        Self {
            width: full_width,
            height: full_height,
            levels,
        }
    }

    /// Uploads the chain as a texture fragment shaders sample.
    fn upload(&self, uploads: &mut Uploads<'_>, label: &str) -> Result<Image> {
        let levels: Vec<&[u8]> = self.levels.iter().map(Vec::as_slice).collect();
        Ok(uploads.image(
            &ImageDesc {
                label,
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(self.width, self.height),
                format: Format::Rgba8Srgb,
                usage: ImageUsage::SAMPLED,
                // One level per bit of the larger side, as generated.
                mip_levels: self.width.max(self.height).bit_width(),
                array_layers: 1,
                samples: SampleCount::X1,
                rest: None,
            },
            &levels,
        )?)
    }

    /// Decodes an sRGB-encoded channel in `0.0..=1.0`.
    fn linear_from_srgb(value: f32) -> f32 {
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Encodes a linear channel in `0.0..=1.0` with the sRGB curve.
    fn srgb_from_linear(value: f32) -> f32 {
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mips;

    #[test]
    fn mips_average_colors_in_linear_space() {
        let black_and_white = vec![0, 0, 0, 255, 255, 255, 255, 255];
        let mips = Mips::generate(2, 1, black_and_white);
        // Half the light of white is sRGB 188, not 128.
        assert_eq!(mips.levels[1..], [vec![188, 188, 188, 255]]);
    }
}
