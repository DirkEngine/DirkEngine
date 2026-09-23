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
use glam::{Mat4, Vec3};

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
        image::Image,
    },
    utils::Vertex,
};

macro_rules! model_ensure {
    ($condition:expr, $($message:tt)*) => {
        if !$condition {
            return Err(anyhow::anyhow!($($message)*).into());
        }
    };
}

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

pub struct Texture {
    pub image: Image,
    pub sampler: Sampler,
}

struct Primitive {
    pub vertex_buffer: VertexBuffer<Vertex>,
    pub index_buffer: dirk_rhi::Buffer,
    pub index_count: u32,
    pub material_handle: Option<Handle<Material>>,
}

struct Mesh {
    pub primitives: Vec<Primitive>,
}

struct Material {
    pub set: DescriptorSet<MaterialSet>,
}

struct Model {
    pub meshes: Vec<Handle<Mesh>>,
    pub materials: Vec<Handle<Material>>,
    pub textures: Vec<Handle<Texture>>,
    generation: dirk_assets::AssetGeneration,
}

pub struct ModelRegistry {
    textures: slotmap::SlotMap<slotmap::DefaultKey, Texture>,
    meshes: slotmap::SlotMap<slotmap::DefaultKey, Mesh>,
    materials: slotmap::SlotMap<slotmap::DefaultKey, Material>,
    models: HashMap<dirk_assets::AssetHandle, Model>,

    fallback_material: Material,
    #[allow(unused)]
    fallback_texture: Texture,
    material_alloc: BindingLayout<MaterialSet>,

    asset_load_consumer: dirk_events::Consumer<::dirk_assets::AssetLoaded<::dirk_assets::Model>>,
    asset_unload_consumer: dirk_events::Consumer<::dirk_assets::AssetUnloaded>,
}

impl ModelRegistry {
    pub fn new(device: &mut Rhi, events: &dirk_events::EventManager) -> Result<Self> {
        let mut material_alloc = BindingLayout::<MaterialSet>::new(device)?;
        let mut uploads = dirk_render_utils::upload::UploadBatch::new();
        let (fallback_material, fallback_texture) =
            Self::create_fallback_material(device, &mut uploads, &mut material_alloc)?;
        uploads.finish(device)?;

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
    pub fn tick(
        &mut self,
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
    ) -> Result<()> {
        let events = self.asset_load_consumer.consume_all().collect::<Vec<_>>();
        for event in events {
            self.load_model(device, uploads, &event.handle)?;
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
        Ok(())
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
        material_alloc: &mut BindingLayout<MaterialSet>,
    ) -> Result<(Material, Texture)> {
        let white = gltf::image::Data {
            pixels: vec![255, 255, 255, 255],
            format: gltf::image::Format::R8G8B8A8,
            width: 1,
            height: 1,
        };
        let texture = Image::upload_texture(device, uploads, &white)?;
        let set =
            material_alloc.sampled_image(device, 0, texture.image.rhi_view(), &texture.sampler)?;

        Ok((Material { set }, texture))
    }

    fn load_model(
        &mut self,
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        handle: &dirk_assets::Handle<dirk_assets::Model>,
    ) -> Result<()> {
        let dirk_assets::Model {
            gltf,
            buffers,
            images,
        } = handle.get()?;
        let asset_handle = handle.handle();

        // Normal, metallic/roughness and emissive images are not rendered by this
        // pipeline. Upload only images referenced as material base colors.
        let base_color_images = gltf
            .materials()
            .filter_map(|mat| mat.pbr_metallic_roughness().base_color_texture())
            .map(|info| info.texture().source().index())
            .collect::<HashSet<_>>();
        if let Some(&index) = base_color_images
            .iter()
            .find(|&&index| index >= images.len())
        {
            return Err(Error::TextureIndexOutOfRange(index));
        }
        let mut texture_handles = vec![None; images.len()];
        for index in base_color_images {
            let image = images
                .get(index)
                .ok_or(Error::TextureIndexOutOfRange(index))?;
            match Image::upload_texture(device, uploads, image) {
                Ok(tex) => texture_handles[index] = Some(Handle::new(self.textures.insert(tex))),
                Err(error) => {
                    self.remove_model_parts(
                        &[],
                        &[],
                        &texture_handles
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                    );
                    return Err(error);
                }
            }
        }

        let mut material_handles = Vec::new();
        let mut mesh_handles = Vec::new();

        let result = (|| -> Result<()> {
            material_handles =
                self.create_materials(device, gltf.materials().collect(), &texture_handles)?;

            for (mesh, transform) in Self::scene_meshes(&gltf)? {
                let normal = transform.inverse().transpose();
                let primitives = mesh
                    .primitives()
                    .map(|prim| {
                        Self::upload_primitive(
                            device,
                            uploads,
                            &prim,
                            &buffers,
                            &material_handles,
                            transform,
                            normal,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                mesh_handles.push(Handle::new(self.meshes.insert(Mesh { primitives })));
            }

            self.unload_model(&asset_handle);
            self.models.insert(
                asset_handle,
                Model {
                    meshes: mesh_handles.clone(),
                    materials: material_handles.clone(),
                    textures: texture_handles.iter().flatten().copied().collect(),
                    generation: handle.generation(),
                },
            );
            Ok(())
        })();

        if result.is_err() {
            self.remove_model_parts(
                &mesh_handles,
                &material_handles,
                &texture_handles
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            );
        }

        result
    }

    fn create_materials(
        &mut self,
        device: &Rhi,
        materials: Vec<gltf::Material>,
        texture_refs: &[Option<Handle<Texture>>],
    ) -> Result<Vec<Handle<Material>>> {
        let mut pending = Vec::with_capacity(materials.len());

        for mat in materials {
            let base_color = mat
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|texture| {
                    let tex_index = texture.texture().source().index();
                    texture_refs
                        .get(tex_index)
                        .copied()
                        .flatten()
                        .ok_or(Error::TextureIndexOutOfRange(tex_index))
                })
                .transpose()?;

            let texture =
                base_color.map_or(&self.fallback_texture, |handle| &self.textures[*handle]);
            let set = self.material_alloc.sampled_image(
                device,
                0,
                texture.image.rhi_view(),
                &texture.sampler,
            )?;
            pending.push(set);
        }

        Ok(pending
            .into_iter()
            .map(|set| Handle::new(self.materials.insert(Material { set })))
            .collect())
    }

    /// Resolve all instances in the selected scene. Nodes outside the scene
    /// must not render, even when their mesh definitions exist in the asset.
    fn scene_meshes(gltf: &gltf::Document) -> Result<Vec<(gltf::Mesh<'_>, Mat4)>> {
        let scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .context("glTF model has no scene")?;
        let mut meshes = Vec::new();
        let mut nodes = scene
            .nodes()
            .map(|node| (node, Mat4::IDENTITY))
            .collect::<Vec<_>>();
        nodes.reverse();
        let mut seen = HashSet::new();
        while let Some((node, parent)) = nodes.pop() {
            model_ensure!(
                seen.insert(node.index()),
                "glTF scene contains a cyclic or multiply parented node {}",
                node.index()
            );
            let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
            model_ensure!(
                transform.is_finite(),
                "glTF node {} has a non-finite transform",
                node.index()
            );
            if let Some(mesh) = node.mesh() {
                let determinant = transform.determinant();
                model_ensure!(
                    determinant.is_finite() && determinant != 0.0,
                    "glTF node {} has a singular mesh transform",
                    node.index()
                );
                meshes.push((mesh, transform));
            }
            nodes.extend(
                node.children()
                    .map(|child| (child, transform))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev(),
            );
        }
        Ok(meshes)
    }

    fn validate_primitive(primitive: &gltf::Primitive) -> Result<()> {
        model_ensure!(
            primitive.mode() == gltf::mesh::Mode::Triangles,
            "glTF primitive {} uses unsupported topology {:?}",
            primitive.index(),
            primitive.mode()
        );
        let material = primitive.material();
        model_ensure!(
            material.alpha_mode() == gltf::material::AlphaMode::Opaque && !material.double_sided(),
            "glTF primitive {} uses unsupported alpha mode or double-sided material",
            primitive.index()
        );
        if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
            let sampler = texture.texture().sampler();
            model_ensure!(
                texture.tex_coord() == 0
                    && sampler.wrap_s() == gltf::texture::WrappingMode::Repeat
                    && sampler.wrap_t() == gltf::texture::WrappingMode::Repeat
                    && matches!(
                        sampler.mag_filter(),
                        None | Some(gltf::texture::MagFilter::Linear)
                    )
                    && matches!(
                        sampler.min_filter(),
                        None | Some(gltf::texture::MinFilter::LinearMipmapLinear)
                    ),
                "glTF primitive {} uses unsupported base-color texture coordinates or sampler",
                primitive.index()
            );
        }
        Ok(())
    }

    /// glTF normals are optional; generate smooth vertex normals when absent.
    fn primitive_normals(
        positions: &[[f32; 3]],
        indices: &[u32],
        normals: Vec<[f32; 3]>,
    ) -> Result<Vec<[f32; 3]>> {
        if !normals.is_empty() {
            model_ensure!(
                normals.len() == positions.len(),
                "glTF primitive normal count does not match its positions"
            );
            return Ok(normals);
        }
        let mut accumulated = vec![Vec3::ZERO; positions.len()];
        for &[a, b, c] in indices.as_chunks::<3>().0 {
            let a = usize::try_from(a).context("glTF vertex index exceeds host address space")?;
            let b = usize::try_from(b).context("glTF vertex index exceeds host address space")?;
            let c = usize::try_from(c).context("glTF vertex index exceeds host address space")?;
            let face = (Vec3::from_array(positions[b]) - Vec3::from_array(positions[a]))
                .cross(Vec3::from_array(positions[c]) - Vec3::from_array(positions[a]));
            accumulated[a] += face;
            accumulated[b] += face;
            accumulated[c] += face;
        }
        Ok(accumulated
            .into_iter()
            .map(|normal| normal.normalize_or_zero().to_array())
            .collect())
    }

    fn upload_primitive(
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
        mat_refs: &[Handle<Material>],
        transform: Mat4,
        normal_transform: Mat4,
    ) -> Result<Primitive> {
        Self::validate_primitive(primitive)?;
        let material = primitive.material();
        let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

        let positions: Vec<_> = reader
            .read_positions()
            .map(Iterator::collect)
            .unwrap_or_default();
        let normals: Vec<_> = reader
            .read_normals()
            .map(Iterator::collect)
            .unwrap_or_default();
        let texcoords: Vec<_> = reader
            .read_tex_coords(0)
            .map(|iter| iter.into_f32().collect())
            .unwrap_or_default();
        let colors: Vec<[f32; 4]> = reader
            .read_colors(0)
            .map(|iter| iter.into_rgba_f32().collect())
            .unwrap_or_default();
        model_ensure!(!positions.is_empty(), "glTF primitive has no positions");
        let vertex_count = u32::try_from(positions.len())
            .context("glTF primitive has too many vertices for indexed drawing")?;
        let mut indices: Vec<_> = reader.read_indices().map_or_else(
            || (0..vertex_count).collect(),
            |iter| iter.into_u32().collect(),
        );
        model_ensure!(
            !indices.is_empty()
                && indices.len().is_multiple_of(3)
                && indices.iter().all(|&i| i < vertex_count),
            "glTF triangle primitive has empty, incomplete, or out-of-range indices"
        );
        let normals = Self::primitive_normals(&positions, &indices, normals)?;
        let index_count = u32::try_from(indices.len())
            .context("glTF primitive has too many indices for indexed drawing")?;
        if transform.determinant() < 0.0 {
            for triangle in indices.as_chunks_mut::<3>().0 {
                triangle.swap(1, 2);
            }
        }
        let factor = material.pbr_metallic_roughness().base_color_factor();

        let vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &position)| Vertex {
                position: transform
                    .transform_point3(Vec3::from_array(position))
                    .to_array(),
                normal: normal_transform
                    .transform_vector3(Vec3::from_array(normals[i]))
                    .normalize_or_zero()
                    .to_array(),
                texcoord: texcoords.get(i).copied().unwrap_or([0.0, 0.0]),
                color: {
                    let color = colors.get(i).copied().unwrap_or([1.0; 4]);
                    std::array::from_fn(|channel| color[channel] * factor[channel])
                },
            })
            .collect();

        let vertex_buffer = VertexBuffer::upload(device, uploads, &vertices)?;
        let index_buffer = uploads.buffer(
            device,
            bytemuck::cast_slice(&indices),
            dirk_rhi::BufferUsages::INDEX,
            dirk_rhi::ResourceAccess::Index,
        )?;

        Ok(Primitive {
            vertex_buffer,
            index_buffer,
            index_count,
            material_handle: primitive.material().index().map(|idx| mat_refs[idx]),
        })
    }

    fn unload_model(&mut self, handle: &dirk_assets::AssetHandle) {
        let Some(model) = self.models.remove(handle) else {
            return;
        };

        for mesh_handle in model.meshes {
            self.meshes.remove(*mesh_handle);
        }
        for material_handle in model.materials {
            self.materials.remove(*material_handle);
        }
        for texture_handle in model.textures {
            self.textures.remove(*texture_handle);
        }
    }

    fn remove_model_parts(
        &mut self,
        mesh_handles: &[Handle<Mesh>],
        material_handles: &[Handle<Material>],
        texture_handles: &[Handle<Texture>],
    ) {
        for mesh_handle in mesh_handles {
            self.meshes.remove(**mesh_handle);
        }
        for material_handle in material_handles {
            self.materials.remove(**material_handle);
        }
        for texture_handle in texture_handles {
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
    use super::ModelRegistry;

    #[test]
    fn non_indexed_triangle_can_generate_normals() {
        let normals = ModelRegistry::primitive_normals(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[0, 1, 2],
            Vec::new(),
        )
        .expect("triangle normals");
        assert_eq!(normals, vec![[0.0, 0.0, 1.0]; 3]);
    }

    #[test]
    fn scene_meshes_compose_transforms_and_keep_instances() {
        let source = br#"{
          "asset":{"version":"2.0"},
          "scene":0,
          "scenes":[{"nodes":[0,2]}],
          "nodes":[
            {"translation":[1,0,0],"children":[1]},
            {"mesh":0,"translation":[0,2,0]},
            {"mesh":0,"translation":[0,0,3]},
            {"mesh":1}
          ],
          "meshes":[
            {"primitives":[{"attributes":{"POSITION":0}}]},
            {"primitives":[{"attributes":{"POSITION":0}}]}
          ],
          "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
          "bufferViews":[{"buffer":0,"byteLength":36}],
          "buffers":[{"byteLength":36}]
        }"#;
        let gltf = gltf::Gltf::from_slice(source).expect("valid glTF scene");
        let meshes = ModelRegistry::scene_meshes(&gltf.document).expect("scene traversal");
        assert_eq!(meshes.len(), 2);
        assert!(meshes.iter().all(|(mesh, _)| mesh.index() == 0));
        assert_eq!(
            meshes[0].1.transform_point3(glam::Vec3::ZERO),
            glam::Vec3::new(1.0, 2.0, 0.0)
        );
        assert_eq!(
            meshes[1].1.transform_point3(glam::Vec3::ZERO),
            glam::Vec3::new(0.0, 0.0, 3.0)
        );
    }
}
