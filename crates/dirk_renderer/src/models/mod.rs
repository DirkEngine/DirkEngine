//! The GPU side of model assets: their meshes, base color textures and
//! samplers, uploaded when the asset loads and dropped when it unloads.
//!
//! glTF support: triangle lists, strips and fans, with or without 8-, 16- or
//! 32-bit indices, including sparse accessors; positions, normals (flat per
//! face when missing), vertex colors and the UV set of the material's base
//! color texture; base color factors; RGB and RGBA 8-bit base color textures
//! with their sampler's filters and wrapping. Unsupported content is logged and skipped: such primitives are
//! not drawn, and such images are replaced by white.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet, VecDeque, hash_map::Entry},
};

use dirk_assets::{AssetGeneration, AssetHandle, AssetLoaded};
use dirk_render_utils::Uploads;
use dirk_rhi::{
    AddressMode, Buffer, BufferAccess, BufferUsage, Extent3d, Filter, Format, Image, ImageDesc,
    ImageDimension, ImageUsage, Rhi, SampleCount, Sampler, SamplerDesc, SamplerIndex, ShaderStages,
    TextureIndex,
};
use dirk_shaders::types::MeshVertex;
use glam::Vec3;
use tracing::warn;

use crate::{Result, Sampled};

/// Every model loaded on the GPU, keyed by asset.
pub struct ModelRegistry {
    models: HashMap<AssetHandle, Model>,
    /// White texture of primitives without a base color texture.
    fallback: Image,
    /// Samplers of model textures, shared by every model: glTF samplers
    /// have few combinations.
    samplers: HashMap<SamplerKey, Sampler>,
    /// Models loaded but not uploaded yet, oldest first.
    pending: VecDeque<AssetLoaded<dirk_assets::Model>>,
    /// Bytes a frame's uploads stage before [`update`](Self::update) leaves
    /// the models left to later frames.
    budget: u64,

    load_consumer: dirk_events::Consumer<AssetLoaded<dirk_assets::Model>>,
    unload_consumer: dirk_events::Consumer<dirk_assets::AssetUnloaded>,
}

/// A model on the GPU. Dropping it frees its meshes and textures.
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
    /// 32-bit indices of a triangle list.
    pub indices: Buffer,
    pub index_count: u32,
    /// The base color texture.
    pub texture: TextureIndex,
    /// The sampler reading it.
    pub sampler: SamplerIndex,
}

/// How a glTF material's primitives sample their base color.
#[derive(Clone, Copy)]
struct Material {
    texture: TextureIndex,
    sampler: SamplerIndex,
    /// The UV set the texture is mapped with.
    uv_set: u32,
}

/// The filtering and wrapping of a glTF sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SamplerKey {
    mag_filter: Filter,
    min_filter: Filter,
    /// Filter between mips, or `None` to sample mip 0 only.
    mipmap_filter: Option<Filter>,
    /// Wrapping along U and V.
    address_modes: [AddressMode; 2],
}

/// Triangle-list geometry read from a glTF primitive.
struct Geometry {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    /// The number of indices.
    index_count: u32,
}

/// glTF content the renderer skips.
#[derive(Debug, thiserror::Error)]
enum Unsupported {
    #[error("{0:?} primitives are not drawn")]
    Topology(gltf::mesh::Mode),
    #[error("the primitive has no positions")]
    NoPositions,
    #[error("the primitive has no triangles")]
    NoTriangles,
    #[error("a triangle list of {0} indices, not a multiple of three")]
    PartialTriangle(usize),
    #[error("index {index} is outside the primitive's {vertices} vertices")]
    IndexOutOfRange { index: u32, vertices: u32 },
    #[error("{0} vertices or indices exceed a 32-bit count")]
    TooLarge(usize),
    #[error("image format {0:?} is not supported")]
    ImageFormat(gltf::image::Format),
}

impl ModelRegistry {
    /// Creates an empty registry listening to asset loads on `events`, which
    /// uploads models until a frame's uploads staged `budget` bytes. Its
    /// fallback texture is uploaded with `uploads`.
    pub fn new(
        uploads: &mut Uploads<'_>,
        events: &dirk_events::EventManager,
        budget: u64,
    ) -> Result<Self> {
        Ok(Self {
            models: HashMap::new(),
            fallback: Self::upload_texture(uploads, "fallback texture", 1, 1, &[255; 4])?,
            samplers: HashMap::new(),
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
    /// uploaded are skipped with a warning. An unload only drops the
    /// generation it names, not a newer one loaded since; models unloaded
    /// before their upload are never uploaded.
    pub fn update(&mut self, rhi: &Rhi, uploads: &mut Uploads<'_>) {
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
            match self.upload(rhi, &asset, event.handle.generation(), uploads) {
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

    /// Uploads `model`, loaded as `generation`: its primitives, and the
    /// images their materials use as base color.
    fn upload(
        &mut self,
        rhi: &Rhi,
        model: &dirk_assets::Model,
        generation: AssetGeneration,
        uploads: &mut Uploads<'_>,
    ) -> Result<Model> {
        let used: HashSet<_> = model
            .gltf
            .materials()
            .filter_map(|material| material.pbr_metallic_roughness().base_color_texture())
            .map(|info| info.texture().source().index())
            .collect();
        let mut textures = HashMap::new();
        for index in used {
            let image = &model.images[index];
            match Self::rgba(image) {
                Ok(texels) => {
                    let (width, height) = (image.width, image.height);
                    let texture =
                        Self::upload_texture(uploads, "model texture", width, height, &texels)?;
                    textures.insert(index, texture);
                }
                Err(error) => warn!(image = index, %error, "using white for a glTF image"),
            }
        }
        let fallback = self.fallback.texture()?;
        let default = Material {
            texture: fallback,
            sampler: self.sampler(rhi, SamplerKey::DEFAULT)?,
            uv_set: 0,
        };
        let materials = model
            .gltf
            .materials()
            .map(|material| {
                let Some(info) = material.pbr_metallic_roughness().base_color_texture() else {
                    return Ok(default);
                };
                let texture = info.texture();
                Ok(Material {
                    texture: textures
                        .get(&texture.source().index())
                        .map_or(Ok(fallback), Image::texture)?,
                    sampler: self.sampler(rhi, SamplerKey::from(&texture.sampler()))?,
                    uv_set: info.tex_coord(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut primitives = Vec::new();
        for mesh in model.gltf.meshes() {
            for primitive in mesh.primitives() {
                let material = primitive
                    .material()
                    .index()
                    .map_or(default, |index| materials[index]);
                match Geometry::read(&primitive, &model.buffers, material.uv_set) {
                    Ok(geometry) => {
                        primitives.push(Primitive::upload(&geometry, material, uploads)?);
                    }
                    Err(error) => warn!(
                        mesh = mesh.index(),
                        primitive = primitive.index(),
                        %error,
                        "skipping a glTF primitive"
                    ),
                }
            }
        }
        Ok(Model {
            generation,
            _textures: textures.into_values().collect(),
            primitives,
        })
    }

    /// The texels of a glTF image as RGBA, or why its format is not
    /// supported.
    fn rgba(image: &gltf::image::Data) -> std::result::Result<Cow<'_, [u8]>, Unsupported> {
        match image.format {
            gltf::image::Format::R8G8B8 => Ok(image
                .pixels
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|&[r, g, b]| [r, g, b, 255])
                .collect()),
            gltf::image::Format::R8G8B8A8 => Ok(Cow::Borrowed(&image.pixels)),
            format => Err(Unsupported::ImageFormat(format)),
        }
    }

    /// The sampler of `key`, created on first use.
    fn sampler(&mut self, rhi: &Rhi, key: SamplerKey) -> Result<SamplerIndex> {
        Ok(match self.samplers.entry(key) {
            Entry::Occupied(sampler) => sampler.get().index(),
            Entry::Vacant(entry) => entry.insert(rhi.create_sampler(&key.desc(rhi))?).index(),
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

impl SamplerKey {
    /// glTF's default: trilinear filtering, repeating.
    const DEFAULT: Self = Self {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_filter: Some(Filter::Linear),
        address_modes: [AddressMode::Repeat; 2],
    };

    /// The sampler's description. Only the default sampler, trilinear and
    /// repeating, is anisotropic.
    fn desc(self, rhi: &Rhi) -> SamplerDesc<'static> {
        let max_anisotropy = if self == Self::DEFAULT {
            rhi.capabilities().limits.max_sampler_anisotropy
        } else {
            1
        };
        let [u, v] = self.address_modes;
        SamplerDesc {
            label: "model texture",
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
            mipmap_filter: self.mipmap_filter.unwrap_or(Filter::Nearest),
            address_modes: [u, v, AddressMode::Repeat],
            max_anisotropy,
            max_lod: if self.mipmap_filter.is_some() {
                f32::MAX
            } else {
                0.0
            },
            ..SamplerDesc::default()
        }
    }
}

impl From<&gltf::texture::Sampler<'_>> for SamplerKey {
    /// Unspecified filters are linear, mipmapped.
    fn from(sampler: &gltf::texture::Sampler<'_>) -> Self {
        use gltf::texture::{MagFilter, MinFilter, WrappingMode};

        let wrap = |mode| match mode {
            WrappingMode::ClampToEdge => AddressMode::ClampToEdge,
            WrappingMode::MirroredRepeat => AddressMode::MirrorRepeat,
            WrappingMode::Repeat => AddressMode::Repeat,
        };
        let (min_filter, mipmap_filter) = match sampler.min_filter() {
            Some(MinFilter::Nearest) => (Filter::Nearest, None),
            Some(MinFilter::Linear) => (Filter::Linear, None),
            Some(MinFilter::NearestMipmapNearest) => (Filter::Nearest, Some(Filter::Nearest)),
            Some(MinFilter::LinearMipmapNearest) => (Filter::Linear, Some(Filter::Nearest)),
            Some(MinFilter::NearestMipmapLinear) => (Filter::Nearest, Some(Filter::Linear)),
            Some(MinFilter::LinearMipmapLinear) | None => (Filter::Linear, Some(Filter::Linear)),
        };
        Self {
            mag_filter: match sampler.mag_filter() {
                Some(MagFilter::Nearest) => Filter::Nearest,
                Some(MagFilter::Linear) | None => Filter::Linear,
            },
            min_filter,
            mipmap_filter,
            address_modes: [wrap(sampler.wrap_s()), wrap(sampler.wrap_t())],
        }
    }
}

impl Geometry {
    /// Reads `primitive`'s triangles, with the texture coordinates of
    /// `uv_set`, and colors multiplied by its material's base color factor.
    /// Without normals, each triangle gets its own vertices facing its way,
    /// as glTF asks.
    fn read(
        primitive: &gltf::Primitive<'_>,
        buffers: &[gltf::buffer::Data],
        uv_set: u32,
    ) -> std::result::Result<Self, Unsupported> {
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or(Unsupported::NoPositions)?
            .collect();
        let count =
            u32::try_from(positions.len()).map_err(|_| Unsupported::TooLarge(positions.len()))?;
        let indices: Vec<u32> = reader.read_indices().map_or_else(
            || (0..count).collect(),
            |indices| indices.into_u32().collect(),
        );
        if let Some(&index) = indices.iter().find(|&&index| index >= count) {
            return Err(Unsupported::IndexOutOfRange {
                index,
                vertices: count,
            });
        }
        let indices = Self::triangle_list(primitive.mode(), &indices)?;
        let index_count =
            u32::try_from(indices.len()).map_err(|_| Unsupported::TooLarge(indices.len()))?;
        let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(Iterator::collect);
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(uv_set)
            .map_or_default(|uvs| uvs.into_f32().collect());
        let colors: Vec<[f32; 4]> = reader
            .read_colors(0)
            .map_or_default(|colors| colors.into_rgba_f32().collect());
        let factor = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_factor();
        let vertex = |index: usize| MeshVertex {
            position: positions[index],
            normal: normals
                .as_ref()
                .and_then(|normals| normals.get(index).copied())
                .unwrap_or_default(),
            uv: uvs.get(index).copied().unwrap_or_default(),
            color: colors.get(index).map_or(factor, |color| {
                std::array::from_fn(|i| color[i] * factor[i])
            }),
        };
        if normals.is_some() {
            return Ok(Self {
                vertices: (0..positions.len()).map(vertex).collect(),
                indices,
                index_count,
            });
        }
        let mut vertices: Vec<_> = indices
            .iter()
            .map(|&index| vertex(index as usize))
            .collect();
        for triangle in vertices.as_chunks_mut::<3>().0 {
            let [a, b, c] = triangle
                .each_ref()
                .map(|vertex| Vec3::from(vertex.position));
            let normal = (b - a).cross(c - a).normalize_or_zero().into();
            for vertex in triangle {
                vertex.normal = normal;
            }
        }
        Ok(Self {
            vertices,
            indices: (0..index_count).collect(),
            index_count,
        })
    }

    /// The triangle list drawing `indices` with `mode`, keeping the winding
    /// of strips' odd triangles and of fans.
    fn triangle_list(
        mode: gltf::mesh::Mode,
        indices: &[u32],
    ) -> std::result::Result<Vec<u32>, Unsupported> {
        use gltf::mesh::Mode;

        let triangles: Vec<u32> = match mode {
            Mode::Triangles if !indices.len().is_multiple_of(3) => {
                return Err(Unsupported::PartialTriangle(indices.len()));
            }
            Mode::Triangles => indices.to_vec(),
            Mode::TriangleStrip => indices
                .windows(3)
                .enumerate()
                .flat_map(|(i, v)| {
                    if i % 2 == 0 {
                        [v[0], v[1], v[2]]
                    } else {
                        [v[0], v[2], v[1]]
                    }
                })
                .collect(),
            Mode::TriangleFan => match indices.split_first() {
                Some((&center, rest)) => rest
                    .windows(2)
                    .flat_map(|edge| [edge[0], edge[1], center])
                    .collect(),
                None => Vec::new(),
            },
            Mode::Points | Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
                return Err(Unsupported::Topology(mode));
            }
        };
        if triangles.is_empty() {
            return Err(Unsupported::NoTriangles);
        }
        Ok(triangles)
    }
}

impl Primitive {
    /// Uploads `geometry`, drawn with `material`.
    fn upload(geometry: &Geometry, material: Material, uploads: &mut Uploads<'_>) -> Result<Self> {
        Ok(Self {
            vertices: uploads.buffer(
                "mesh vertices",
                BufferUsage::STORAGE,
                BufferAccess::ShaderRead(ShaderStages::VERTEX),
                bytemuck::cast_slice(&geometry.vertices),
            )?,
            indices: uploads.buffer(
                "mesh indices",
                BufferUsage::INDEX,
                BufferAccess::Index,
                bytemuck::cast_slice(&geometry.indices),
            )?,
            index_count: geometry.index_count,
            texture: material.texture,
            sampler: material.sampler,
        })
    }
}

#[cfg(test)]
mod tests;
