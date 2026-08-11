use std::collections::{HashMap, HashSet};

use dirk_rhi::{
    CommandBuffer as _, Extent3d, Format, ImageUsages, MemoryDomain, Rect, SampleCount, Viewport,
};
use dirk_shaders::types::{ProxyUbo, SceneUbo};
use dirk_universe::{Entity, WorldId};

use crate::{
    Error, MAX_FRAMES_IN_FLIGHT, Result,
    frame_graph::{AttachmentInfo, RenderGraph, TextureDesc, TextureHandle},
    models::ModelRegistry,
    pipeline::{MainPipelineSpec, graphics::GraphicsPipeline},
    resources::{
        buffer::UniformBuffer,
        command_pool::CommandBuffer,
        descriptors::{
            DescriptorAllocator, DescriptorSet,
            sets::{ObjectSet, SceneSet},
        },
        device::RenderDevice,
    },
};

pub(crate) struct SceneRenderSettings {
    pub extent: Extent3d,
    pub format: Format,
    pub clear_color: [f32; 4],
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

/// This is the renderer proxy for the [`Universe`]. It also has
/// most of the rendering state needed to render each scene.
pub struct SceneManager {
    device: RenderDevice,

    scenes: HashMap<WorldId, Scene>,
    entities: HashMap<Entity, WorldId>,
    proxies: HashMap<Entity, SceneProxy>,

    // TODO: see about centralising the different pipelines
    graphics_pipeline: GraphicsPipeline<MainPipelineSpec>,

    scene_alloc: DescriptorAllocator<SceneSet>,
    proxy_alloc: DescriptorAllocator<ObjectSet>,
}

impl SceneManager {
    pub fn init(device: &RenderDevice) -> Result<Self> {
        let scene_alloc = DescriptorAllocator::<SceneSet>::new(device, 16)?;
        let proxy_alloc = DescriptorAllocator::<ObjectSet>::new(device, 256)?;
        let graphics_pipeline = GraphicsPipeline::build(device)?;

        Ok(Self {
            device: device.clone(),
            entities: HashMap::new(),
            scenes: HashMap::new(),
            proxies: HashMap::new(),
            graphics_pipeline,
            scene_alloc,
            proxy_alloc,
        })
    }
    pub fn render<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        models: &'a ModelRegistry,
        world: WorldId,
        camera: Entity,
        settings: SceneRenderSettings,
        target: TextureHandle,
    ) {
        let depth = graph.create_texture(TextureDesc {
            width: settings.extent.width,
            height: settings.extent.height,
            format: self.device.properties.depth_format,
            usage: ImageUsages::DEPTH_STENCIL_ATTACHMENT,
            samples: self.device.properties.msaa_samples,
            imported: None,
        });

        let msaa_color = if self.device.properties.msaa_samples == SampleCount::One {
            None
        } else {
            Some(graph.create_texture(TextureDesc {
                width: settings.extent.width,
                height: settings.extent.height,
                format: settings.format,
                usage: ImageUsages::TRANSIENT_ATTACHMENT | ImageUsages::COLOR_ATTACHMENT,
                samples: self.device.properties.msaa_samples,
                imported: None,
            }))
        };

        let mut pass = graph.add_pass("scene");
        let [r, g, b, a] = settings.clear_color;
        if let Some(msaa_color) = msaa_color {
            pass.write_color_attachment_with_resolve(
                msaa_color,
                target,
                AttachmentInfo::clear_color(r, g, b, a),
            );
        } else {
            pass.write_color_attachment(target, AttachmentInfo::clear_color(r, g, b, a));
        }
        pass.write_depth_attachment(depth, AttachmentInfo::clear_discard_depth(1., 0));
        pass.execute(Box::new(move |_, cmd, _| {
            self.record_scene_draws(models, cmd, world, &settings, camera)
        }));
    }
    fn record_scene_draws(
        &self,
        models: &ModelRegistry,
        cmd: &mut CommandBuffer,
        world: WorldId,
        settings: &SceneRenderSettings,
        camera: Entity,
    ) -> Result<()> {
        let frame = self.device.current_frame();
        let scene = self
            .scenes
            .get(&world)
            .ok_or(Error::WorldDoesNotExist(world))?;

        let proxies = scene
            .entities
            .iter()
            .filter_map(|e| self.proxies.get(e))
            .collect::<Vec<_>>();

        // CAMERA
        {
            let proxy = &self
                .proxies
                .get(&camera)
                .ok_or(Error::CameraDoesNotExist(camera))?;

            let view = proxy.view.ok_or(Error::CameraDoesNotExist(camera))?;

            // TODO: proper viewport & camera system
            let proj = {
                // `width` & `height` aren't large enough for this to matter
                #[allow(clippy::cast_precision_loss)]
                let aspect = settings.extent.width as f32 / settings.extent.height.max(1) as f32;
                glam::camera::rh::proj::vulkan::perspective(
                    settings.fov_y_radians,
                    aspect,
                    settings.near,
                    settings.far,
                )
            };

            let scene_ubo = SceneUbo { view, proj };
            scene.ubo[frame].write(&scene_ubo)?;
        };

        for proxy in &proxies {
            proxy.write_ubo(frame)?;
        }

        let mut ctx = self.graphics_pipeline.bind(cmd);

        // the window size never gets anywhere near 2^23
        #[allow(clippy::cast_precision_loss)]
        ctx.command().rhi_mut().set_viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: settings.extent.width as f32,
            height: settings.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        ctx.command().rhi_mut().set_scissor(Rect {
            x: 0,
            y: 0,
            width: settings.extent.width,
            height: settings.extent.height,
        });

        for proxy in &proxies {
            let Some(ref model) = proxy.model else {
                continue;
            };

            match models.render_model(model, &scene.sets[frame], &proxy.sets[frame], &mut ctx) {
                Ok(()) | Err(dirk_assets::Error::NotFound(_)) => (),
                Err(err) => return Err(err.into()),
            }
        }

        Ok(())
    }
    pub fn create_scene(&mut self, world: WorldId) -> Result<()> {
        let scene = Scene::build(self)?;
        self.scenes.insert(world, scene);
        Ok(())
    }
    pub fn destroy_scene(&mut self, world: WorldId) {
        self.scenes.remove(&world);
    }
    pub fn entity_world(&self, entity: Entity) -> Option<WorldId> {
        self.entities.get(&entity).copied()
    }
    pub fn create_proxy(&mut self, entity: Entity, world: WorldId) -> Result<()> {
        let proxy = SceneProxy::build(self)?;
        self.proxies.insert(entity, proxy);

        self.entities.insert(entity, world);
        self.scenes
            .get_mut(&world)
            .ok_or(Error::WorldDoesNotExist(world))?
            .entities
            .insert(entity);
        Ok(())
    }
    pub fn get_proxy_mut(&mut self, entity: Entity) -> Option<&mut SceneProxy> {
        self.proxies.get_mut(&entity)
    }
    pub fn send_proxy(&mut self, entity: Entity, to: WorldId) -> Result<()> {
        let world = self
            .entities
            .get(&entity)
            .copied()
            .ok_or(Error::EntityDoesNotExist(entity))?;

        if !self.scenes.contains_key(&to) {
            return Err(Error::WorldDoesNotExist(to));
        }

        let old = self
            .scenes
            .get_mut(&world)
            .ok_or(Error::WorldDoesNotExist(world))?;

        old.entities.remove(&entity);

        let new = self
            .scenes
            .get_mut(&to)
            .ok_or(Error::WorldDoesNotExist(to))?;
        new.entities.insert(entity);

        self.entities.insert(entity, to);
        Ok(())
    }
    pub fn destroy_proxy(&mut self, entity: Entity) -> Result<()> {
        let world = self
            .entities
            .get(&entity)
            .ok_or(Error::EntityDoesNotExist(entity))?;

        self.scenes
            .get_mut(world)
            .ok_or(Error::WorldDoesNotExist(*world))?
            .entities
            .remove(&entity);
        self.proxies.remove(&entity);
        self.entities.remove(&entity);
        Ok(())
    }
}

impl Drop for SceneManager {
    fn drop(&mut self) {
        // Clear collections before allocators are dropped. Scene and
        // SceneProxy hold DescriptorSet values whose Drop impls enqueue descriptor
        // set frees; the allocators enqueue descriptor pool destroys.
        self.scenes.clear();
        self.entities.clear();
        self.proxies.clear();
    }
}

/// Renderer representation of a [`World`].
struct Scene {
    entities: HashSet<Entity>,

    ubo: [UniformBuffer; MAX_FRAMES_IN_FLIGHT],
    sets: [DescriptorSet<SceneSet>; MAX_FRAMES_IN_FLIGHT],
}

impl Scene {
    /// Builds a [Scene].
    /// Constructs the renderer stuff like command pools, descriptor sets, ... from
    /// the [Renderer].
    pub fn build(manager: &mut SceneManager) -> Result<Self> {
        let ubo_size = size_of::<SceneUbo>() as u64;
        let build_ubo = || UniformBuffer::create(&manager.device, ubo_size, MemoryDomain::Upload);
        let ubo = [build_ubo()?, build_ubo()?];
        let sets = [
            manager
                .scene_alloc
                .uniform_buffer(0, ubo[0].rhi(), ubo_size)?,
            manager
                .scene_alloc
                .uniform_buffer(0, ubo[1].rhi(), ubo_size)?,
        ];

        Ok(Self {
            entities: HashSet::new(),
            ubo,
            sets,
        })
    }
}

pub struct SceneProxy {
    /// The model matrix used for rendering. Constructed from the
    /// [`world::components::Transform`] of the entity.
    model_matrix: Option<glam::Mat4>,
    /// The view matrix used for rendering as camera
    view: Option<glam::Mat4>,
    /// The name of the model. Used to request a [`crate::model::Model`] from the
    /// renderer at render time.
    model: Option<dirk_assets::AssetHandle>,

    // Per frame render stuff
    ubo: [UniformBuffer; MAX_FRAMES_IN_FLIGHT],
    sets: [DescriptorSet<ObjectSet>; MAX_FRAMES_IN_FLIGHT],
}

impl SceneProxy {
    pub fn build(manager: &mut SceneManager) -> Result<Self> {
        let size = size_of::<ProxyUbo>() as u64;
        let build_ubo = || UniformBuffer::create(&manager.device, size, MemoryDomain::Upload);
        let ubo = [build_ubo()?, build_ubo()?];
        let sets = [
            manager.proxy_alloc.uniform_buffer(0, ubo[0].rhi(), size)?,
            manager.proxy_alloc.uniform_buffer(0, ubo[1].rhi(), size)?,
        ];

        Ok(Self {
            model: None,
            model_matrix: None,
            view: None,
            ubo,
            sets,
        })
    }
    pub fn set_model(&mut self, model: Option<dirk_assets::AssetHandle>) {
        self.model = model;
    }
    pub fn set_model_matrix(&mut self, mat: Option<glam::Mat4>) -> Result<()> {
        self.model_matrix = mat;

        if let Some(mat) = mat {
            let proxy_ubo = ProxyUbo { model: mat };
            for ubo in &self.ubo {
                ubo.write(&proxy_ubo)?;
            }
        }
        Ok(())
    }
    pub fn set_view(&mut self, view: Option<glam::Mat4>) {
        self.view = view;
    }
    pub fn write_ubo(&self, frame: usize) -> Result<()> {
        let Some(model) = self.model_matrix else {
            return Ok(());
        };

        let data = ProxyUbo { model };
        self.ubo[frame].write(&data)
    }
}
