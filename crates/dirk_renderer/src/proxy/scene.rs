//! CPU scene proxies and frame-slot GPU preparation.
use crate::{
    MAX_FRAMES_IN_FLIGHT, RendererProperties, Result,
    frame_graph::{AttachmentInfo, RenderGraph, TextureDesc, TextureHandle},
    models::ModelRegistry,
    pipeline::{MainPipelineSpec, graphics::GraphicsPipeline},
    render_commands::RenderDelta,
    resources::{
        buffer::UniformBuffer,
        descriptors::{BindingLayout, DescriptorSet, sets::ObjectSet},
    },
    viewport::Viewport,
};
use dirk_player::PlayerId;
use dirk_rhi::{Extent3d, Rect, Rhi, SampleCount, TextureFormat};
use dirk_shaders::types::ProxyUbo;
use dirk_universe::{Entity, WorldId};
use std::collections::{HashMap, HashSet};

pub(crate) struct SceneRenderSettings {
    pub extent: Extent3d,
    pub format: TextureFormat,
    pub clear_color: [f32; 4],
}
pub struct SceneManager {
    proxies: HashMap<Entity, SceneProxy>,
    graphics_pipeline: GraphicsPipeline<MainPipelineSpec>,
    proxy_alloc: BindingLayout<ObjectSet>,
    properties: RendererProperties,
    ambiguous: HashMap<PlayerId, HashSet<Entity>>,
}
impl SceneManager {
    pub fn init(rhi: &Rhi, properties: RendererProperties) -> Result<Self> {
        Ok(Self {
            proxies: HashMap::new(),
            graphics_pipeline: GraphicsPipeline::build(
                rhi,
                MainPipelineSpec::settings(properties),
            )?,
            proxy_alloc: BindingLayout::new(rhi)?,
            properties,
            ambiguous: HashMap::new(),
        })
    }
    pub fn entities(&self) -> impl Iterator<Item = (Entity, WorldId)> + '_ {
        self.proxies.iter().map(|(e, p)| (*e, p.world))
    }
    pub fn apply(&mut self, deltas: Vec<RenderDelta>) {
        for delta in deltas {
            let Some(data) = delta
                .state
                .filter(|data| data.model.is_some() || data.player.is_some())
            else {
                self.proxies.remove(&delta.entity);
                continue;
            };
            let matrix = data
                .transform
                .as_ref()
                .map(dirk_world::components::Transform::matrix)
                .filter(|m| {
                    let determinant = m.determinant();
                    m.is_finite() && determinant.is_finite() && determinant != 0.0
                });
            let view = data
                .transform
                .as_ref()
                .filter(|t| {
                    t.location.is_finite() && t.rotation.is_finite() && t.rotation.is_normalized()
                })
                .map(dirk_world::components::Transform::view);
            let proxy = self
                .proxies
                .entry(delta.entity)
                .or_insert_with(|| SceneProxy {
                    world: data.world,
                    model: None,
                    model_matrix: None,
                    view: None,
                    player: None,
                    gpu: None,
                });
            if data.model.is_some()
                && matrix.is_none()
                && (proxy.model.is_none() || proxy.model_matrix.is_some())
            {
                tracing::warn!(entity = ?delta.entity, "renderable has no valid transform; skipping it");
            }
            if data.player.is_some()
                && view.is_none()
                && (proxy.player.is_none() || proxy.view.is_some())
            {
                tracing::warn!(entity = ?delta.entity, "camera has no valid transform; viewport unavailable");
            }
            proxy.world = data.world;
            proxy.model = data.model;
            proxy.model_matrix = matrix;
            proxy.view = view;
            proxy.player = data.player;
        }
    }
    pub fn reconcile_views(&mut self, viewports: &mut HashMap<PlayerId, Viewport>) {
        let mut cameras: HashMap<PlayerId, HashSet<Entity>> = HashMap::new();
        for (entity, proxy) in &self.proxies {
            if let Some(player) = proxy.player {
                cameras.entry(player).or_default().insert(*entity);
            }
        }
        let mut ambiguous = HashMap::new();
        for (player, viewport) in viewports {
            let candidates = cameras.get(player);
            let camera = candidates
                .filter(|c| c.len() == 1)
                .and_then(|c| c.iter().next())
                .copied();
            if let Some(candidates) = candidates.filter(|c| c.len() > 1) {
                if self.ambiguous.get(player) != Some(candidates) {
                    tracing::warn!(
                        ?player,
                        ?candidates,
                        "multiple cameras assigned to one player; viewport unavailable"
                    );
                }
                ambiguous.insert(*player, candidates.clone());
            }
            let valid = camera.and_then(|e| {
                self.proxies
                    .get(&e)
                    .filter(|p| p.view.is_some())
                    .map(|p| (e, p.world))
            });
            let previous = (viewport.camera, viewport.world);
            viewport.camera = valid.map(|(e, _)| e);
            viewport.world = valid.map(|(_, w)| w);
            if previous != (viewport.camera, viewport.world) || valid.is_none() {
                viewport.invalidate();
            }
        }
        self.ambiguous = ambiguous;
    }
    /// Called only after the renderer has waited for this frame slot.
    pub fn prepare(
        &mut self,
        rhi: &Rhi,
        frame: usize,
        viewports: &mut HashMap<PlayerId, Viewport>,
    ) -> Result<()> {
        for proxy in self.proxies.values_mut() {
            let Some(model) = proxy.model_matrix.filter(|_| proxy.model.is_some()) else {
                continue;
            };
            if proxy.gpu.is_none() {
                proxy.gpu = Some(ProxyGpu::new(rhi, &self.proxy_alloc)?);
            }
            // SAFETY: this frame slot's previous submission has completed.
            if let Some(gpu) = &mut proxy.gpu {
                unsafe {
                    gpu.ubo[frame].write(&ProxyUbo {
                        model,
                        normal: model.inverse().transpose(),
                    })?;
                }
            }
        }
        for viewport in viewports.values_mut() {
            if let Some(view) = viewport
                .camera
                .and_then(|e| self.proxies.get(&e))
                .and_then(|p| p.view)
            {
                viewport.prepare_camera(frame, view)?;
            }
        }
        Ok(())
    }
    pub fn render<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        models: &'a ModelRegistry,
        viewport: &'a Viewport,
        target: TextureHandle,
        frame: usize,
    ) -> Result<()> {
        let Some(world) = viewport.world else {
            return Ok(());
        };
        let scene_set = viewport.camera_set(frame);
        let settings = SceneRenderSettings {
            extent: viewport.settings().extent,
            format: viewport.settings().format,
            clear_color: viewport.settings().clear_color,
        };
        let uniform_state = dirk_rhi::ImageState::Uniform(dirk_rhi::ShaderStages::VERTEX);
        let mut uniforms = vec![graph.import_buffer(crate::frame_graph::ImportedBuffer {
            buffer: viewport.camera_buffer(frame),
            initial_state: uniform_state,
            final_state: uniform_state,
        })?];
        for proxy in self
            .proxies
            .values()
            .filter(|p| p.world == world && p.model_matrix.is_some())
        {
            if let Some(gpu) = &proxy.gpu {
                uniforms.push(graph.import_buffer(crate::frame_graph::ImportedBuffer {
                    buffer: gpu.ubo[frame].buffer(),
                    initial_state: uniform_state,
                    final_state: uniform_state,
                })?);
            }
        }
        let depth = graph.create_texture(TextureDesc {
            width: settings.extent.width,
            height: settings.extent.height,
            format: self.properties.depth_format,
            samples: self.properties.msaa_samples,
            imported: None,
        });
        let color = (self.properties.msaa_samples != SampleCount::One).then(|| {
            graph.create_texture(TextureDesc {
                width: settings.extent.width,
                height: settings.extent.height,
                format: settings.format,
                samples: self.properties.msaa_samples,
                imported: None,
            })
        });
        let mut pass = graph.add_pass("scene");
        for uniform in uniforms {
            pass.read_buffer(uniform, uniform_state);
        }
        let [r, g, b, a] = settings.clear_color;
        if let Some(color) = color {
            pass.write_color_attachment_with_resolve(
                color,
                target,
                AttachmentInfo::clear_color(r, g, b, a),
            );
        } else {
            pass.write_color_attachment(target, AttachmentInfo::clear_color(r, g, b, a));
        }
        pass.write_depth_attachment(depth, AttachmentInfo::clear_discard_depth(1.0, 0));
        // SAFETY: model vertices, indices and textures are immutable after the upload acquisition.
        unsafe {
            pass.external_reads(Box::new(move |cmd, _| {
                // SAFETY: renderer owns the immutable pipeline and model resources through GPU use;
                // uploads acquire their resources before this graph and uniforms belong to the waited slot.
                let mut ctx = self.graphics_pipeline.bind(cmd)?;
                #[allow(clippy::cast_precision_loss)]
                ctx.command().set_viewport(dirk_rhi::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: settings.extent.width as f32,
                    height: settings.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                })?;
                ctx.command().set_scissor(Rect {
                    x: 0,
                    y: 0,
                    width: settings.extent.width,
                    height: settings.extent.height,
                })?;
                for proxy in self
                    .proxies
                    .values()
                    .filter(|p| p.world == world && p.model_matrix.is_some())
                {
                    if let (Some(model), Some(gpu)) = (&proxy.model, &proxy.gpu) {
                        match models.render_model(model, scene_set, &gpu.sets[frame], &mut ctx) {
                            Ok(())
                            | Err(crate::Error::AssetError(dirk_assets::Error::NotFound(_))) => {}
                            Err(error) => return Err(error.into()),
                        }
                    }
                }
                Ok(())
            }));
        }
        Ok(())
    }
}
struct SceneProxy {
    world: WorldId,
    model: Option<dirk_assets::AssetHandle>,
    model_matrix: Option<glam::Mat4>,
    view: Option<glam::Mat4>,
    player: Option<PlayerId>,
    gpu: Option<ProxyGpu>,
}
struct ProxyGpu {
    ubo: [UniformBuffer<ProxyUbo>; MAX_FRAMES_IN_FLIGHT],
    sets: [DescriptorSet<ObjectSet>; MAX_FRAMES_IN_FLIGHT],
}
impl ProxyGpu {
    fn new(rhi: &Rhi, allocator: &BindingLayout<ObjectSet>) -> Result<Self> {
        let ubo = [UniformBuffer::new(rhi)?, UniformBuffer::new(rhi)?];
        let sets = [
            allocator.uniform_buffer(rhi, 0, ubo[0].buffer(), size_of::<ProxyUbo>() as u64)?,
            allocator.uniform_buffer(rhi, 0, ubo[1].buffer(), size_of::<ProxyUbo>() as u64)?,
        ];
        Ok(Self { ubo, sets })
    }
}
