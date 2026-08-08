//! This module contains all the logic necessary to rendering models.
//! As models are complex & have funny inter-dependencies, this is a
//! centralised system that has all textures, meshes, materials, ...
//!
//! When someone needs to render a model to the screen, all they have to do
//! is call [`ModelRegistry::render`] with their asset handle & a command buffer.
//! We handle the rest.

use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
};

use ash::vk;

use crate::{
    Error, Result,
    pipeline::{MainPipelineSpec, graphics::GraphicsPipelineRenderingContext},
    resources::{
        buffer::{IndexBuffer, VertexBuffer},
        command_pool::CommandBuffer,
        descriptors::{
            DescriptorAllocator, DescriptorSet, DescriptorWriter,
            sets::{MaterialSet, ObjectSet, SceneSet},
        },
        device::RenderDevice,
        image::Image,
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

pub struct Texture {
    pub image: Image,
    pub sampler: dirk_rhi_vulkan::VulkanSampler,
}

struct Primitive {
    pub vertex_buffer: VertexBuffer<Vertex>,
    pub index_buffer: IndexBuffer,
    pub index_count: u32,
    pub material_handle: Option<Handle<Material>>,
}

struct Mesh {
    pub primitives: Vec<Primitive>,
}

struct Material {
    pub base_color: Option<Handle<Texture>>,
    pub set: DescriptorSet<MaterialSet>,
}

struct Model {
    // TODO: store transform with each mesh handle
    pub meshes: Vec<Handle<Mesh>>,
}

pub struct ModelRegistry {
    device: RenderDevice,

    textures: slotmap::SlotMap<slotmap::DefaultKey, Texture>,
    meshes: slotmap::SlotMap<slotmap::DefaultKey, Mesh>,
    materials: slotmap::SlotMap<slotmap::DefaultKey, Material>,
    models: HashMap<dirk_assets::AssetHandle, Model>,

    fallback_material: Material,
    #[allow(unused)]
    fallback_texture: Texture,
    material_alloc: DescriptorAllocator<MaterialSet>,

    asset_load_consumer: dirk_events::Consumer<::dirk_assets::AssetLoaded<::dirk_assets::Model>>,
    asset_unload_consumer: dirk_events::Consumer<::dirk_assets::AssetUnloaded>,
}

impl ModelRegistry {
    pub fn new(device: &RenderDevice, events: &dirk_events::EventManager) -> Result<Self> {
        let mut material_alloc = DescriptorAllocator::<MaterialSet>::new(device, 64)?;
        let (fallback_material, fallback_texture) =
            Self::create_fallback_material(device, &mut material_alloc)?;

        Ok(Self {
            device: device.clone(),
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
    pub fn tick(&mut self) -> Result<()> {
        let events = self.asset_load_consumer.consume_all().collect::<Vec<_>>();
        for event in events {
            self.load_model(&event.handle)?;
        }

        let events = self.asset_unload_consumer.consume_all().collect::<Vec<_>>();
        for event in events {
            self.unload_model(&event.handle);
        }
        Ok(())
    }
    pub fn render_model(
        &self,
        handle: &dirk_assets::AssetHandle,
        cmd: &CommandBuffer,
        scene_set: &DescriptorSet<SceneSet>,
        proxy_set: &DescriptorSet<ObjectSet>,
        ctx: &GraphicsPipelineRenderingContext<'_, MainPipelineSpec>,
    ) -> dirk_assets::Result<()> {
        if handle.asset_type() != dirk_assets::AssetType::Model {
            return Err(dirk_assets::Error::TypeMismatch(handle.to_string()));
        }

        let model = self
            .models
            .get(handle)
            .ok_or(dirk_assets::Error::NotFound(handle.to_string()))?;

        let primitives = model
            .meshes
            .iter()
            .flat_map(|&mesh| self.meshes[*mesh].primitives.iter());

        for prim in primitives {
            let material_set = prim
                .material_handle
                .map_or(&self.fallback_material.set, |mat| &self.materials[*mat].set);

            ctx.bind_descriptor_sets(&(scene_set, proxy_set, material_set));
            ctx.bind_vertex_buffer(&prim.vertex_buffer);
            cmd.bind_index_buffer(prim.index_buffer.buffer(), 0, vk::IndexType::UINT32);
            cmd.draw_indexed(prim.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    fn create_fallback_material(
        device: &RenderDevice,
        material_alloc: &mut DescriptorAllocator<MaterialSet>,
    ) -> Result<(Material, Texture)> {
        let white = gltf::image::Data {
            pixels: vec![255, 255, 255, 255],
            format: gltf::image::Format::R8G8B8A8,
            width: 1,
            height: 1,
        };
        let texture = Image::upload_texture(device, &white)?;
        let set = material_alloc.allocate()?;

        DescriptorWriter::new(&device.device)
            .combined_image_sampler(&set, 0, texture.image.view(), texture.sampler.raw())
            .flush();

        Ok((
            Material {
                base_color: None,
                set,
            },
            texture,
        ))
    }

    fn load_model(&mut self, handle: &dirk_assets::Handle<dirk_assets::Model>) -> Result<()> {
        let dirk_assets::Model {
            gltf,
            buffers,
            images,
        } = handle.get()?;
        let asset_handle = handle.handle();

        let mut texture_handles = Vec::with_capacity(images.len());
        for image in &images {
            match Image::upload_texture(&self.device, image) {
                Ok(tex) => texture_handles.push(Handle::new(self.textures.insert(tex))),
                Err(error) => {
                    self.remove_model_parts(&[], &[], &texture_handles);
                    return Err(error);
                }
            }
        }

        let mut material_handles = Vec::new();
        let mut mesh_handles = Vec::new();

        let result = (|| -> Result<()> {
            material_handles =
                self.create_materials(gltf.materials().collect(), &texture_handles)?;

            for mesh in gltf.meshes() {
                let primitives = mesh
                    .primitives()
                    .map(|prim| self.upload_primitive(&prim, &buffers, &material_handles))
                    .collect::<Result<Vec<_>>>()?;
                mesh_handles.push(Handle::new(self.meshes.insert(Mesh { primitives })));
            }

            self.unload_model(&asset_handle);
            self.models.insert(
                asset_handle,
                Model {
                    meshes: mesh_handles.clone(),
                },
            );
            Ok(())
        })();

        if result.is_err() {
            self.remove_model_parts(&mesh_handles, &material_handles, &texture_handles);
        }

        result
    }

    fn create_materials(
        &mut self,
        materials: Vec<gltf::Material>,
        texture_refs: &[Handle<Texture>],
    ) -> Result<Vec<Handle<Material>>> {
        let mut pending = Vec::with_capacity(materials.len());

        for mat in materials {
            let set = self.material_alloc.allocate()?;
            let base_color = mat
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|texture| {
                    let tex_index = texture.texture().source().index();
                    texture_refs
                        .get(tex_index)
                        .copied()
                        .ok_or(Error::TextureIndexOutOfRange(tex_index))
                })
                .transpose()?;

            let (view, sampler) = base_color.map_or_else(
                || {
                    (
                        self.fallback_texture.image.view(),
                        self.fallback_texture.sampler.raw(),
                    )
                },
                |tex_handle| {
                    let tex = &self.textures[*tex_handle];
                    (tex.image.view(), tex.sampler.raw())
                },
            );
            pending.push((base_color, set, view, sampler));
        }

        let mut writer = DescriptorWriter::new(&self.device.device);
        for (_, set, view, sampler) in &pending {
            writer = writer.combined_image_sampler(set, 0, *view, *sampler);
        }
        writer.flush();

        Ok(pending
            .into_iter()
            .map(|(base_color, set, _, _)| {
                Handle::new(self.materials.insert(Material { base_color, set }))
            })
            .collect())
    }

    fn upload_primitive(
        &self,
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
        mat_refs: &[Handle<Material>],
    ) -> Result<Primitive> {
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
        let indices: Vec<_> = reader
            .read_indices()
            .map(|iter| iter.into_u32().collect())
            .unwrap_or_default();

        let vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &position)| Vertex {
                position,
                normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                texcoord: texcoords.get(i).copied().unwrap_or([0.0, 0.0]),
            })
            .collect();

        let vertex_buffer = VertexBuffer::upload_slice(&self.device, &vertices)?;
        let index_buffer = IndexBuffer::upload_slice(&self.device, &indices)?;

        // indices.len() will not surpass u32::MAX
        #[allow(clippy::cast_possible_truncation)]
        Ok(Primitive {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            material_handle: primitive.material().index().map(|idx| mat_refs[idx]),
        })
    }

    fn unload_model(&mut self, handle: &dirk_assets::AssetHandle) {
        let Some(model) = self.models.remove(handle) else {
            return;
        };

        let mut material_handles = HashSet::new();
        for mesh_handle in model.meshes {
            if let Some(mesh) = self.meshes.remove(*mesh_handle) {
                material_handles.extend(
                    mesh.primitives
                        .into_iter()
                        .filter_map(|primitive| primitive.material_handle),
                );
            }
        }

        let mut texture_handles = HashSet::new();
        for material_handle in material_handles {
            if let Some(material) = self.materials.remove(*material_handle) {
                texture_handles.extend(material.base_color);
            }
        }

        for texture_handle in texture_handles {
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
