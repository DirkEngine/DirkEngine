use std::collections::{HashMap, HashSet};

use dirk_rhi::{
    Buffer, BufferDesc, BufferUsage, Capabilities, ColorTarget, CompareOp, CullMode, DepthBias,
    DepthState, Format, Frame, FrontFace, GraphicsPipeline, GraphicsPipelineDesc, ImageUsage,
    IndexFormat, Indices, Memory, PerFrame, RenderPass, Rhi, SampleCount, Topology,
};
use dirk_shaders::types::{MeshPush, ObjectData};
use dirk_universe::{Entity, WorldId};

use crate::{
    COLOR_FORMAT, Error, Result, StorageBuffer,
    models::ModelRegistry,
    shaders::{MeshFs, MeshVs},
};

/// Objects the per-frame object buffers hold at first.
const INITIAL_OBJECTS: u64 = 64;

/// The renderer's proxy of the universe: its worlds, the entities in them,
/// and what renders them.
pub struct SceneManager {
    /// The entities of each world.
    scenes: HashMap<WorldId, HashSet<Entity>>,
    entities: HashMap<Entity, WorldId>,
    proxies: HashMap<Entity, SceneProxy>,

    targets: SceneTargets,
    pipeline: GraphicsPipeline<MeshPush>,
    /// Each frame's [`ObjectData`], indexed by the proxies' `object`.
    objects: PerFrame<Buffer>,
}

/// The attachments scenes render to besides their color image, which has
/// [`COLOR_FORMAT`]: a depth image, and the sample count of both.
#[derive(Clone, Copy, Debug)]
pub struct SceneTargets {
    pub depth: Format,
    /// Above one, a multisampled color image is rendered to and resolved.
    pub samples: SampleCount,
}

/// What a viewport sees of a scene this frame.
pub struct SceneView<'a> {
    entities: &'a HashSet<Entity>,
    view_projection: glam::Mat4,
}

impl SceneManager {
    /// Creates an empty scene manager and its mesh pipeline.
    pub fn new(rhi: &Rhi) -> Result<Self> {
        let targets = SceneTargets::select(rhi.capabilities())?;
        let vertex = rhi.create_shader::<MeshVs, MeshPush>()?;
        let fragment = rhi.create_shader::<MeshFs, MeshPush>()?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: "mesh",
            vertex: &vertex,
            fragment: Some(&fragment),
            topology: Topology::TriangleList,
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            color_targets: &[ColorTarget::new(COLOR_FORMAT)],
            depth: Some(DepthState {
                format: targets.depth,
                write: true,
                compare: CompareOp::Less,
                bias: DepthBias::default(),
            }),
            samples: targets.samples,
        })?;
        let objects = PerFrame::try_new(rhi, |_| {
            Self::create_objects(rhi, INITIAL_OBJECTS * size_of::<ObjectData>() as u64)
        })?;
        Ok(Self {
            scenes: HashMap::new(),
            entities: HashMap::new(),
            proxies: HashMap::new(),
            targets,
            pipeline,
            objects,
        })
    }

    /// The attachments scenes render to.
    pub fn targets(&self) -> SceneTargets {
        self.targets
    }

    /// Writes the data of every drawable object to `frame`'s object buffer,
    /// growing it if needed, and gives each its index for this frame.
    pub fn write_objects(&mut self, frame: &Frame<'_>) -> Result<()> {
        let mut data = Vec::with_capacity(self.proxies.len());
        for proxy in self.proxies.values_mut() {
            proxy.object = match (&proxy.model, proxy.model_matrix) {
                (Some(_), Some(model)) => {
                    data.push(ObjectData {
                        model: model.to_cols_array_2d(),
                    });
                    u32::try_from(data.len() - 1).ok()
                }
                _ => None,
            };
        }
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let size = bytes.len() as u64;
        let buffer = self.objects.get_mut(frame);
        if size > buffer.info().size {
            *buffer = Self::create_objects(frame.rhi(), size.next_power_of_two())?;
        }
        if !bytes.is_empty() {
            self.objects.write(frame, 0, bytes)?;
        }
        Ok(())
    }

    /// What `camera` sees of `world`, projected onto an image of `aspect`
    /// width per height.
    pub fn view(
        &self,
        world: WorldId,
        camera: Entity,
        aspect: f32,
        projection: Projection,
    ) -> Result<SceneView<'_>> {
        let entities = self
            .scenes
            .get(&world)
            .ok_or(Error::WorldDoesNotExist(world))?;
        let view = self
            .proxies
            .get(&camera)
            .and_then(|proxy| proxy.view)
            .ok_or(Error::CameraDoesNotExist(camera))?;
        let projection = glam::camera::rh::proj::vulkan::perspective(
            projection.fov_y_radians,
            aspect,
            projection.near,
            projection.far,
        );
        Ok(SceneView {
            entities,
            view_projection: projection * view,
        })
    }

    /// Draws every object `view` sees with the data
    /// [`write_objects`](Self::write_objects) wrote to `frame`'s buffer, in
    /// a pass with [`targets`](Self::targets).
    pub fn draw(
        &self,
        pass: &mut RenderPass<'_, '_>,
        objects: &Buffer,
        models: &ModelRegistry,
        view: &SceneView<'_>,
    ) -> dirk_rhi::Result<()> {
        let objects = objects.storage()?;
        let view_projection = view.view_projection.to_cols_array_2d();
        for proxy in view.entities.iter().filter_map(|e| self.proxies.get(e)) {
            let (Some(model), Some(object)) = (&proxy.model, proxy.object) else {
                continue;
            };
            for primitive in models.primitives(model).unwrap_or_default() {
                let push = MeshPush {
                    view_projection,
                    objects,
                    vertices: primitive.vertices.storage()?,
                    texture: primitive.texture,
                    sampler: primitive.sampler,
                };
                let indices = Indices {
                    buffer: &primitive.indices,
                    format: IndexFormat::U32,
                    offset: 0,
                    range: 0..primitive.index_count,
                    base_vertex: 0,
                };
                pass.draw_indexed(&self.pipeline, &push, indices, object..object + 1)?;
            }
        }
        Ok(())
    }

    /// `frame`'s object buffer.
    pub fn objects(&self, frame: &Frame<'_>) -> &Buffer {
        self.objects.get(frame)
    }

    pub fn create_scene(&mut self, world: WorldId) {
        self.scenes.insert(world, HashSet::new());
    }
    pub fn destroy_scene(&mut self, world: WorldId) {
        self.scenes.remove(&world);
    }
    pub fn entity_world(&self, entity: Entity) -> Option<WorldId> {
        self.entities.get(&entity).copied()
    }
    pub fn create_proxy(&mut self, entity: Entity, world: WorldId) -> Result<()> {
        self.scenes
            .get_mut(&world)
            .ok_or(Error::WorldDoesNotExist(world))?
            .insert(entity);
        self.proxies.insert(entity, SceneProxy::default());
        self.entities.insert(entity, world);
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
        self.scenes
            .get_mut(&world)
            .ok_or(Error::WorldDoesNotExist(world))?
            .remove(&entity);
        self.scenes
            .get_mut(&to)
            .ok_or(Error::WorldDoesNotExist(to))?
            .insert(entity);
        self.entities.insert(entity, to);
        Ok(())
    }
    pub fn destroy_proxy(&mut self, entity: Entity) -> Result<()> {
        let world = self
            .entities
            .remove(&entity)
            .ok_or(Error::EntityDoesNotExist(entity))?;
        self.scenes
            .get_mut(&world)
            .ok_or(Error::WorldDoesNotExist(world))?
            .remove(&entity);
        self.proxies.remove(&entity);
        Ok(())
    }

    /// Creates an object buffer of `size` bytes.
    fn create_objects(rhi: &Rhi, size: u64) -> dirk_rhi::Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "objects",
            size,
            usage: BufferUsage::STORAGE,
            memory: Memory::Upload,
        })
    }
}

impl SceneTargets {
    /// The most precise depth format the device renders to, and the most
    /// samples it supports for it and [`COLOR_FORMAT`], up to four, which
    /// smooths edges well for far less memory and bandwidth than eight.
    fn select(capabilities: &Capabilities) -> Result<Self> {
        let depth = [
            Format::Depth32Float,
            Format::Depth32FloatStencil8,
            Format::Depth24UnormStencil8,
        ]
        .into_iter()
        .find(|&format| {
            capabilities
                .format(format)
                .usage
                .contains(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or(Error::NoDepthFormat)?;
        let supported = capabilities.format(depth).sample_counts
            & capabilities.format(COLOR_FORMAT).sample_counts;
        let samples = [SampleCount::X4, SampleCount::X2]
            .into_iter()
            .find(|samples| supported.contains(samples.flag()))
            .unwrap_or(SampleCount::X1);
        Ok(Self { depth, samples })
    }
}

/// How a viewport projects what its camera sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

/// The renderer's copy of an entity's rendered components.
#[derive(Default)]
pub struct SceneProxy {
    /// Transforms model space to world space, from the entity's transform.
    model_matrix: Option<glam::Mat4>,
    /// Transforms world space to view space, for cameras.
    view: Option<glam::Mat4>,
    /// The model drawn.
    model: Option<dirk_assets::AssetHandle>,
    /// The entity's index in this frame's object buffer, if drawn.
    object: Option<u32>,
}

impl SceneProxy {
    pub fn set_model(&mut self, model: Option<dirk_assets::AssetHandle>) {
        self.model = model;
    }
    pub fn set_model_matrix(&mut self, matrix: Option<glam::Mat4>) {
        self.model_matrix = matrix;
    }
    pub fn set_view(&mut self, view: Option<glam::Mat4>) {
        self.view = view;
    }
}
