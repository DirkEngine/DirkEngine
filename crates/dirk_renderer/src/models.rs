//! This module contains all the logic necessary to rendering models.
//! As models are complex & have funny inter-dependencies, this is a
//! centralised system that has all textures, meshes, materials, ...
//!
//! When someone needs to render a model to the screen, all they have to do
//! is call [`ModelRegistry::render_model`] with their asset handle & a command buffer.
//! We handle the rest.

use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
};

use anyhow::Context;
use dirk_rhi::IndexFormat;
use glam::Vec3;

use crate::{
    Error, Result,
    pipeline::{MainPipelineSpec, graphics::GraphicsPipelineRenderingContext},
    resources::{
        Rhi, Sampler,
        buffer::VertexBuffer,
        descriptors::{
            BindingLayout, DescriptorSet,
            sets::{MaterialSet, ObjectSet, SceneSet},
        },
        image::{Image, TextureSampling},
        upload::MipGeneration,
    },
    utils::Vertex,
};

struct Handle<T> {
    key: slotmap::DefaultKey,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.key.fmt(f)
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl<T> Handle<T> {
    fn new(key: slotmap::DefaultKey) -> Self {
        Self {
            key,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for Handle<T> {}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Deref for Handle<T> {
    type Target = slotmap::DefaultKey;
    fn deref(&self) -> &Self::Target {
        &self.key
    }
}

struct Primitive {
    pub vertex_buffer: VertexBuffer<Vertex>,
    pub index_buffer: dirk_rhi::Buffer,
    pub index_count: u32,
    pub material_handle: Option<Handle<Material>>,
}

/// Triangle-list geometry read from a glTF primitive.
struct PrimitiveData {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

struct Mesh {
    pub primitives: Vec<Primitive>,
}

struct Material {
    pub set: DescriptorSet<MaterialSet>,
    /// Referenced by `set`; kept alive as long as the material.
    _sampler: Sampler,
}

struct Model {
    pub meshes: Vec<Handle<Mesh>>,
    pub materials: Vec<Handle<Material>>,
    pub textures: Vec<Handle<Image>>,
    generation: dirk_assets::AssetGeneration,
}

/// GPU resources created for one glTF model before it is registered.
#[derive(Default)]
struct ModelParts {
    meshes: Vec<Handle<Mesh>>,
    materials: Vec<Handle<Material>>,
    textures: Vec<Handle<Image>>,
}

pub struct ModelRegistry {
    textures: slotmap::SlotMap<slotmap::DefaultKey, Image>,
    meshes: slotmap::SlotMap<slotmap::DefaultKey, Mesh>,
    materials: slotmap::SlotMap<slotmap::DefaultKey, Material>,
    models: HashMap<dirk_assets::AssetHandle, Model>,

    fallback_material: Material,
    fallback_texture: Image,
    material_alloc: BindingLayout<MaterialSet>,

    asset_load_consumer: dirk_events::Consumer<::dirk_assets::AssetLoaded<::dirk_assets::Model>>,
    asset_unload_consumer: dirk_events::Consumer<::dirk_assets::AssetUnloaded>,
}

impl ModelRegistry {
    pub fn new(device: &mut Rhi, events: &dirk_events::EventManager) -> Result<Self> {
        let mut material_alloc = BindingLayout::<MaterialSet>::new(device)?;
        let mut uploads = dirk_render_utils::upload::UploadBatch::new();
        let mut mips = MipGeneration::new();
        let (fallback_material, fallback_texture) =
            Self::create_fallback_material(device, &mut uploads, &mut mips, &mut material_alloc)?;
        uploads.finish(device)?;
        if let Some(commands) = mips.finish()? {
            // SAFETY: the recording only touches the fallback texture created above.
            unsafe {
                device
                    .queue::<dirk_rhi::Graphics>()
                    .submit(vec![commands], &dirk_rhi::SubmitInfo::default())?
            }
            .wait(u64::MAX)?;
        }

        Ok(Self {
            textures: slotmap::SlotMap::new(),
            meshes: slotmap::SlotMap::new(),
            materials: slotmap::SlotMap::new(),
            models: HashMap::new(),
            fallback_material,
            fallback_texture,
            material_alloc,

            asset_load_consumer: events.subscribe(),
            asset_unload_consumer: events.subscribe(),
        })
    }

    /// Loads and unloads models announced by the asset registry.
    ///
    /// Models that cannot be loaded are logged and skipped so that one bad
    /// asset does not stop the renderer; a previously loaded generation of the
    /// same asset stays in use.
    pub fn tick(
        &mut self,
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        mips: &mut MipGeneration,
    ) {
        let events = self.asset_load_consumer.consume_all().collect::<Vec<_>>();
        for event in events {
            if let Err(error) = self.load_model(device, uploads, mips, &event.handle) {
                tracing::error!(
                    model = %event.handle.handle(),
                    %error,
                    "skipping glTF model that could not be loaded"
                );
            }
        }

        let events = self.asset_unload_consumer.consume_all().collect::<Vec<_>>();
        for event in events {
            if self
                .models
                .get(&event.handle)
                .is_some_and(|model| model.generation == event.generation)
            {
                self.unload_model(&event.handle);
            }
        }
    }
    pub unsafe fn render_model(
        &self,
        handle: &dirk_assets::AssetHandle,
        scene_set: &DescriptorSet<SceneSet>,
        proxy_set: &DescriptorSet<ObjectSet>,
        ctx: &mut GraphicsPipelineRenderingContext<'_, '_, MainPipelineSpec>,
    ) -> Result<()> {
        unsafe {
            if handle.asset_type() != dirk_assets::AssetType::Model {
                return Err(dirk_assets::Error::TypeMismatch(handle.to_string()).into());
            }

            let model = self
                .models
                .get(handle)
                .ok_or_else(|| dirk_assets::Error::NotFound(handle.to_string()))?;

            let primitives = model
                .meshes
                .iter()
                .flat_map(|&mesh| self.meshes[*mesh].primitives.iter());

            for prim in primitives {
                let material_set = prim
                    .material_handle
                    .map_or(&self.fallback_material.set, |mat| &self.materials[*mat].set);

                ctx.bind_descriptor_sets(&(scene_set, proxy_set, material_set))?;
                ctx.bind_vertex_buffer(&prim.vertex_buffer)?;
                ctx.command()
                    .bind_index_buffer(&prim.index_buffer, 0, IndexFormat::Uint32)?;
                ctx.command().draw_indexed(prim.index_count, 1, 0, 0, 0)?;
            }
            Ok(())
        }
    }

    fn create_fallback_material(
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        mips: &mut MipGeneration,
        material_alloc: &mut BindingLayout<MaterialSet>,
    ) -> Result<(Material, Image)> {
        let white = gltf::image::Data {
            pixels: vec![255, 255, 255, 255],
            format: gltf::image::Format::R8G8B8A8,
            width: 1,
            height: 1,
        };
        let texture = Image::upload_texture(device, uploads, mips, &white)?;
        let sampler = texture.create_sampler(device, TextureSampling::default())?;
        let set = material_alloc.sampled_image(device, 0, texture.rhi_view(), &sampler)?;

        Ok((
            Material {
                set,
                _sampler: sampler,
            },
            texture,
        ))
    }

    fn load_model(
        &mut self,
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        mips: &mut MipGeneration,
        handle: &dirk_assets::Handle<dirk_assets::Model>,
    ) -> Result<()> {
        let model = handle.get()?;
        let asset_handle = handle.handle();

        let mut parts = ModelParts::default();
        if let Err(error) = self.create_model_parts(device, uploads, mips, &model, &mut parts) {
            self.remove_model_parts(&parts);
            return Err(error);
        }

        self.unload_model(&asset_handle);
        self.models.insert(
            asset_handle,
            Model {
                meshes: parts.meshes,
                materials: parts.materials,
                textures: parts.textures,
                generation: handle.generation(),
            },
        );
        Ok(())
    }

    /// Uploads the renderable parts of a glTF model into `parts`.
    ///
    /// Unsupported images fall back to a white texture and unsupported
    /// primitives are skipped, each with a warning. Errors leave the parts
    /// created so far in `parts` for the caller to remove.
    fn create_model_parts(
        &mut self,
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        mips: &mut MipGeneration,
        model: &dirk_assets::Model,
        parts: &mut ModelParts,
    ) -> Result<()> {
        let dirk_assets::Model {
            gltf,
            buffers,
            images,
        } = model;
        // Normal, metallic/roughness and emissive images are not rendered by this
        // pipeline. Upload only images referenced as material base colors.
        let base_color_images = gltf
            .materials()
            .filter_map(|mat| mat.pbr_metallic_roughness().base_color_texture())
            .map(|info| info.texture().source().index())
            .collect::<HashSet<_>>();
        let mut texture_handles = vec![None; images.len()];
        for index in base_color_images {
            let image = images
                .get(index)
                .ok_or(Error::TextureIndexOutOfRange(index))?;
            match Image::upload_texture(device, uploads, mips, image) {
                Ok(texture) => {
                    let handle = Handle::new(self.textures.insert(texture));
                    parts.textures.push(handle);
                    texture_handles[index] = Some(handle);
                }
                Err(error) => tracing::warn!(
                    image = index,
                    %error,
                    "using a white texture for an unsupported glTF base-color image"
                ),
            }
        }

        self.create_materials(device, gltf, &texture_handles, parts)?;

        // World placement is supplied by entity transforms; mesh coordinates
        // stay in the units used by existing worlds and movement speeds.
        for mesh in gltf.meshes() {
            let mut primitives = Vec::new();
            for primitive in mesh.primitives() {
                let data = match Self::read_primitive(&primitive, buffers) {
                    Ok(data) => data,
                    Err(error) => {
                        tracing::warn!(
                            mesh = mesh.index(),
                            primitive = primitive.index(),
                            %error,
                            "skipping unsupported glTF primitive"
                        );
                        continue;
                    }
                };
                let material_handle = primitive
                    .material()
                    .index()
                    .and_then(|index| parts.materials.get(index).copied());
                primitives.push(Self::upload_primitive(
                    device,
                    uploads,
                    &data,
                    material_handle,
                )?);
            }
            parts
                .meshes
                .push(Handle::new(self.meshes.insert(Mesh { primitives })));
        }
        Ok(())
    }

    /// Creates one material per glTF material, in document order.
    fn create_materials(
        &mut self,
        device: &Rhi,
        gltf: &gltf::Document,
        texture_refs: &[Option<Handle<Image>>],
        parts: &mut ModelParts,
    ) -> Result<()> {
        for mat in gltf.materials() {
            let base_color = mat.pbr_metallic_roughness().base_color_texture();
            let texture = base_color
                .as_ref()
                .and_then(|info| texture_refs.get(info.texture().source().index()))
                .copied()
                .flatten()
                .map_or(&self.fallback_texture, |handle| &self.textures[*handle]);
            let sampling = base_color.map_or_else(TextureSampling::default, |info| {
                Self::texture_sampling(&info.texture().sampler())
            });
            let sampler = texture.create_sampler(device, sampling)?;
            let set = self
                .material_alloc
                .sampled_image(device, 0, texture.rhi_view(), &sampler)?;
            parts
                .materials
                .push(Handle::new(self.materials.insert(Material {
                    set,
                    _sampler: sampler,
                })));
        }
        Ok(())
    }

    /// Maps a glTF sampler to the renderer's texture sampling state.
    fn texture_sampling(sampler: &gltf::texture::Sampler) -> TextureSampling {
        use dirk_rhi::{AddressMode, FilterMode};
        use gltf::texture::{MagFilter, MinFilter, WrappingMode};

        let address = |mode| match mode {
            WrappingMode::ClampToEdge => AddressMode::ClampToEdge,
            WrappingMode::MirroredRepeat => AddressMode::MirrorRepeat,
            WrappingMode::Repeat => AddressMode::Repeat,
        };
        let (minification, mipmapping) = match sampler.min_filter() {
            Some(MinFilter::Nearest) => (FilterMode::Nearest, None),
            Some(MinFilter::Linear) => (FilterMode::Linear, None),
            Some(MinFilter::NearestMipmapNearest) => {
                (FilterMode::Nearest, Some(FilterMode::Nearest))
            }
            Some(MinFilter::LinearMipmapNearest) => (FilterMode::Linear, Some(FilterMode::Nearest)),
            Some(MinFilter::NearestMipmapLinear) => (FilterMode::Nearest, Some(FilterMode::Linear)),
            Some(MinFilter::LinearMipmapLinear) | None => {
                (FilterMode::Linear, Some(FilterMode::Linear))
            }
        };
        TextureSampling {
            mag_filter: match sampler.mag_filter() {
                Some(MagFilter::Nearest) => FilterMode::Nearest,
                Some(MagFilter::Linear) | None => FilterMode::Linear,
            },
            min_filter: minification,
            mip_filter: mipmapping,
            address_u: address(sampler.wrap_s()),
            address_v: address(sampler.wrap_t()),
        }
    }

    /// Converts glTF triangle topologies to a triangle list.
    fn triangle_list(mode: gltf::mesh::Mode, indices: &[u32]) -> anyhow::Result<Vec<u32>> {
        use gltf::mesh::Mode;
        let triangles = match mode {
            Mode::Triangles => {
                anyhow::ensure!(
                    indices.len().is_multiple_of(3),
                    "triangle list has {} indices, which is not a multiple of three",
                    indices.len()
                );
                indices.to_vec()
            }
            // glTF orders odd strip triangles as (i, i + 2, i + 1) to keep their winding.
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
                anyhow::bail!("unsupported topology {mode:?}")
            }
        };
        anyhow::ensure!(!triangles.is_empty(), "primitive has no triangles");
        Ok(triangles)
    }

    /// glTF requires flat normals when NORMAL is absent. Split shared vertices
    /// before assigning face normals, preserving their UVs and vertex colors.
    fn flat_vertices(vertices: &[Vertex], indices: &[u32]) -> Result<Vec<Vertex>> {
        let mut expanded = indices
            .iter()
            .map(|&index| {
                let index = usize::try_from(index)
                    .context("glTF vertex index exceeds host address space")?;
                vertices
                    .get(index)
                    .copied()
                    .context("glTF vertex index is out of range")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        for triangle in expanded.as_chunks_mut::<3>().0 {
            let [a, b, c] = triangle.map(|vertex| Vec3::from_array(vertex.position));
            let normal = (b - a).cross(c - a).normalize_or_zero().to_array();
            for vertex in triangle {
                vertex.normal = normal;
            }
        }
        Ok(expanded)
    }

    /// Reads a primitive's vertices and triangle-list indices.
    ///
    /// All materials render as opaque and back-face culled, matching the
    /// model rendering behavior before the RHI migration.
    fn read_primitive(
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
    ) -> anyhow::Result<PrimitiveData> {
        let material = primitive.material();
        let reader =
            primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));
        let tex_coord_set = material
            .pbr_metallic_roughness()
            .base_color_texture()
            .map_or(0, |info| info.tex_coord());

        let positions: Vec<_> = reader
            .read_positions()
            .map(Iterator::collect)
            .unwrap_or_default();
        let normals: Vec<_> = reader
            .read_normals()
            .map(Iterator::collect)
            .unwrap_or_default();
        let texcoords: Vec<_> = reader
            .read_tex_coords(tex_coord_set)
            .map(|iter| iter.into_f32().collect())
            .unwrap_or_default();
        let colors: Vec<[f32; 4]> = reader
            .read_colors(0)
            .map(|iter| iter.into_rgba_f32().collect())
            .unwrap_or_default();
        anyhow::ensure!(!positions.is_empty(), "primitive has no positions");
        anyhow::ensure!(
            normals.is_empty() || normals.len() == positions.len(),
            "primitive normal count does not match its positions"
        );
        let vertex_count = u32::try_from(positions.len())
            .context("primitive has too many vertices for indexed drawing")?;
        let vertex_indices: Vec<_> = reader.read_indices().map_or_else(
            || (0..vertex_count).collect(),
            |iter| iter.into_u32().collect(),
        );
        let mut indices = Self::triangle_list(primitive.mode(), &vertex_indices)?;
        anyhow::ensure!(
            indices.iter().all(|&index| index < vertex_count),
            "primitive has out-of-range indices"
        );
        let factor = material.pbr_metallic_roughness().base_color_factor();

        let mut vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &position)| Vertex {
                position,
                normal: normals.get(i).copied().unwrap_or([0.0; 3]),
                texcoord: texcoords.get(i).copied().unwrap_or([0.0, 0.0]),
                color: {
                    let color = colors.get(i).copied().unwrap_or([1.0; 4]);
                    std::array::from_fn(|channel| color[channel] * factor[channel])
                },
            })
            .collect();

        if normals.is_empty() {
            vertices = Self::flat_vertices(&vertices, &indices)?;
            let index_count = u32::try_from(vertices.len())
                .context("primitive has too many indices for indexed drawing")?;
            indices = (0..index_count).collect();
        }
        Ok(PrimitiveData { vertices, indices })
    }

    fn upload_primitive(
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        data: &PrimitiveData,
        material_handle: Option<Handle<Material>>,
    ) -> Result<Primitive> {
        let index_count = u32::try_from(data.indices.len())
            .context("glTF primitive has too many indices for indexed drawing")?;
        let vertex_buffer = VertexBuffer::upload(device, uploads, &data.vertices)?;
        let index_buffer = uploads.buffer(
            device,
            bytemuck::cast_slice(&data.indices),
            dirk_rhi::BufferUsages::INDEX,
            dirk_rhi::ResourceAccess::Index,
        )?;

        Ok(Primitive {
            vertex_buffer,
            index_buffer,
            index_count,
            material_handle,
        })
    }

    fn unload_model(&mut self, handle: &dirk_assets::AssetHandle) {
        if let Some(model) = self.models.remove(handle) {
            self.remove_model_parts(&ModelParts {
                meshes: model.meshes,
                materials: model.materials,
                textures: model.textures,
            });
        }
    }

    fn remove_model_parts(&mut self, parts: &ModelParts) {
        for mesh_handle in &parts.meshes {
            self.meshes.remove(**mesh_handle);
        }
        for material_handle in &parts.materials {
            self.materials.remove(**material_handle);
        }
        for texture_handle in &parts.textures {
            self.textures.remove(**texture_handle);
        }
    }
}

impl Drop for ModelRegistry {
    fn drop(&mut self) {
        // Materials hold DescriptorSet values; clear them before the allocator
        // enqueues descriptor pool destruction.
        self.materials.clear();
        self.textures.clear();
        self.meshes.clear();
        self.models.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelParts, ModelRegistry, PrimitiveData, Vertex};
    use crate::resources::{image::TextureSampling, upload::MipGeneration};
    use dirk_rhi::{AddressMode, FilterMode};

    fn fixture(name: &str) -> anyhow::Result<dirk_assets::Model> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let (gltf, buffers, images) = gltf::import(path)?;
        Ok(dirk_assets::Model {
            gltf,
            images,
            buffers,
        })
    }

    /// Reads every primitive of the fixture's first mesh.
    fn read_primitives<const N: usize>(
        name: &str,
    ) -> anyhow::Result<[anyhow::Result<PrimitiveData>; N]> {
        let model = fixture(name)?;
        let mesh = model
            .gltf
            .meshes()
            .next()
            .ok_or_else(|| anyhow::anyhow!("{name} has no mesh"))?;
        mesh.primitives()
            .map(|primitive| ModelRegistry::read_primitive(&primitive, &model.buffers))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|primitives: Vec<_>| {
                anyhow::anyhow!("{name} has {} primitives, not {N}", primitives.len())
            })
    }

    fn positions(data: &PrimitiveData) -> Vec<[f32; 3]> {
        data.indices
            .iter()
            .map(|&index| data.vertices[usize::try_from(index).expect("small index")].position)
            .collect()
    }

    fn faces_up(data: &PrimitiveData) -> bool {
        data.vertices.iter().all(|vertex| {
            (glam::Vec3::from_array(vertex.normal) - glam::Vec3::Z).length() < f32::EPSILON
        })
    }

    #[test]
    fn non_indexed_primitives_generate_indices() -> anyhow::Result<()> {
        let [Ok(data)] = read_primitives("non_indexed.gltf")? else {
            panic!("non-indexed primitive should load");
        };
        assert_eq!(data.indices, [0, 1, 2]);
        assert_eq!(
            positions(&data),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert!(faces_up(&data));
        Ok(())
    }

    #[test]
    fn unsigned_byte_indices_are_widened() -> anyhow::Result<()> {
        let [Ok(data)] = read_primitives("u8_indices.gltf")? else {
            panic!("u8-indexed primitive should load");
        };
        assert_eq!(
            positions(&data),
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn sparse_accessors_substitute_values() -> anyhow::Result<()> {
        let [Ok(data)] = read_primitives("sparse.gltf")? else {
            panic!("sparse primitive should load");
        };
        assert_eq!(
            positions(&data),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        Ok(())
    }

    #[test]
    fn triangle_topologies_load_and_others_are_skipped() -> anyhow::Result<()> {
        let [Ok(list), Ok(strip), Ok(fan), Err(_lines), Err(_partial)] =
            read_primitives("multi_primitive.gltf")?
        else {
            panic!("only the triangle primitives should load");
        };
        assert_eq!(list.indices.len(), 3);
        for converted in [&strip, &fan] {
            assert_eq!(converted.indices.len(), 6);
            // Converted triangles keep the counter-clockwise winding of the quad.
            assert!(faces_up(converted));
        }
        Ok(())
    }

    #[test]
    fn custom_samplers_and_uv_sets_are_supported() -> anyhow::Result<()> {
        let model = fixture("unsupported_sampler.gltf")?;
        let texture = model
            .gltf
            .textures()
            .next()
            .ok_or_else(|| anyhow::anyhow!("fixture has no texture"))?;
        assert_eq!(
            ModelRegistry::texture_sampling(&texture.sampler()),
            TextureSampling {
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mip_filter: Some(FilterMode::Nearest),
                address_u: AddressMode::ClampToEdge,
                address_v: AddressMode::MirrorRepeat,
            }
        );

        let [Ok(data)] = read_primitives("unsupported_sampler.gltf")? else {
            panic!("textured primitive should load");
        };
        let texcoords = data
            .indices
            .iter()
            .map(|&index| data.vertices[usize::try_from(index).expect("small index")].texcoord)
            .collect::<Vec<_>>();
        assert_eq!(texcoords, [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a native Vulkan or Metal device"]
    fn fixture_models_upload_without_errors() -> anyhow::Result<()> {
        let mut rhi = dirk_rhi::Rhi::new(&dirk_rhi::RhiCreateInfo {
            engine_name: "DirkEngine",
            engine_version: (0, 1, 0),
            application_name: "model loading validation",
            application_version: (0, 1, 0),
            validation: true,
            compatible_surface: None,
        })?;
        let events = dirk_events::EventManager::new(dirk_threads::WorkerPool::new("model test"));
        let mut registry = ModelRegistry::new(&mut rhi, &events)?;
        for (name, primitive_count) in [
            ("non_indexed.gltf", 1),
            ("u8_indices.gltf", 1),
            ("sparse.gltf", 1),
            ("multi_primitive.gltf", 3),
            ("unsupported_sampler.gltf", 1),
        ] {
            let model = fixture(name)?;
            let mut uploads = dirk_render_utils::upload::UploadBatch::new();
            let mut mips = MipGeneration::new();
            let mut parts = ModelParts::default();
            registry.create_model_parts(&rhi, &mut uploads, &mut mips, &model, &mut parts)?;
            let [mesh] = parts.meshes[..] else {
                panic!("{name} should create one mesh");
            };
            assert_eq!(
                registry.meshes[*mesh].primitives.len(),
                primitive_count,
                "{name}"
            );
            uploads.finish(&mut rhi)?;
            if let Some(commands) = mips.finish()? {
                // SAFETY: the recording only touches textures created above.
                unsafe {
                    rhi.queue::<dirk_rhi::Graphics>()
                        .submit(vec![commands], &dirk_rhi::SubmitInfo::default())?
                }
                .wait(u64::MAX)?;
            }
            registry.remove_model_parts(&parts);
            rhi.finish_cycle()?;
        }
        drop(registry);
        rhi.wait_idle()?;
        assert_eq!(
            rhi.validation_error_count(),
            0,
            "native model validation errors"
        );
        Ok(())
    }

    #[test]
    fn missing_normals_split_shared_vertices_and_preserve_attributes() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let vertices: Vec<_> = positions
            .into_iter()
            .map(|position| Vertex {
                position,
                normal: [0.0; 3],
                texcoord: [position[0], position[1]],
                color: [position[0], position[1], position[2], 0.5],
            })
            .collect();
        let indices = [0, 1, 2, 0, 3, 1];
        let flat = ModelRegistry::flat_vertices(&vertices, &indices).expect("flat normals");
        assert_eq!(flat.len(), 6);
        for (i, vertex) in flat.iter().enumerate() {
            let expected = if i < 3 { glam::Vec3::Z } else { glam::Vec3::Y };
            assert!((glam::Vec3::from_array(vertex.normal) - expected).length() < f32::EPSILON);
            let original = &vertices[usize::try_from(indices[i]).expect("small index")];
            assert_eq!(
                vertex.position.map(f32::to_bits),
                original.position.map(f32::to_bits)
            );
            assert_eq!(
                vertex.texcoord.map(f32::to_bits),
                original.texcoord.map(f32::to_bits)
            );
            assert_eq!(
                vertex.color.map(f32::to_bits),
                original.color.map(f32::to_bits)
            );
        }
    }

    #[test]
    fn non_indexed_triangle_can_generate_normals() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].map(|position| Vertex {
            position,
            normal: [0.0; 3],
            texcoord: [0.0; 2],
            color: [1.0; 4],
        });
        let flat = ModelRegistry::flat_vertices(&vertices, &[0, 1, 2]).expect("triangle normals");
        assert!(flat.iter().all(|vertex| {
            (glam::Vec3::from_array(vertex.normal) - glam::Vec3::Z).length() < f32::EPSILON
        }));
    }
}
