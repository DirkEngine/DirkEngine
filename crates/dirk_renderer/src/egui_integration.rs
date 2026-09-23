use dirk_render_utils::upload::ImageUpload;
use std::{collections::HashMap, mem::size_of, time::Instant};

use dirk_input::{ButtonState, ImeEvent, InputEvent};
use dirk_platform::{Theme, WindowId, WindowInputEvent};
use dirk_rhi::{
    AddressMode, BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor, BlendOp,
    BlendState, BufferUsages, CullMode, Extent3d, FilterMode, FrontFace, ImageState, ImageUsages,
    IndexFormat, InvalidResourceKind as Ir, MemoryDomain, Origin3d, PrimitiveTopology, RasterState,
    Rect, SampleCount, SamplerDesc, ShaderStages, Viewport,
};
use dirk_shaders::types::EguiUbo;
use egui::{
    ClippedPrimitive, Context, TextureFilter, TextureId, TextureOptions, TextureWrapMode,
    TexturesDelta, ViewportId, ViewportInfo,
    epaint::{ImageData, Primitive},
};
use tracing::warn;

use crate::{
    MAX_FRAMES_IN_FLIGHT, Result,
    pipeline::graphics::{GraphicsPipeline, GraphicsPipelineSpec},
    resources::{
        CommandEncoder as CommandBuffer, ImageView, RenderPass, Rhi,
        buffer::UniformBuffer,
        descriptors::{BindingLayout, DescriptorSet, layouts::SetLayout},
        image::{Image, ImageCreateInfo},
    },
    shaders::{EguiFS, EguiVS},
};

pub struct EguiState {
    ctx: Context,
    target_format: dirk_rhi::TextureFormat,
    output_is_srgb: bool,
    modifiers: egui::Modifiers,
    pipeline: GraphicsPipeline<EguiPipelineSpec>,
    texture_allocator: BindingLayout<EguiTextureSet>,
    user_sampler: crate::resources::Sampler,
    textures: HashMap<TextureId, EguiTexture>,
    frames: [EguiFrameResources; MAX_FRAMES_IN_FLIGHT],
    next_user_texture: u64,
    start_time: Instant,
    pending: Option<EguiPaintData>,
    prepared: Option<PreparedFrame>,
    textures_to_free: [Vec<TextureId>; MAX_FRAMES_IN_FLIGHT],
}

pub struct EguiFrameInput {
    pub window_id: WindowId,
    pub extent: Extent3d,
    pub native_pixels_per_point: f32,
    pub focused: bool,
    pub theme: Option<Theme>,
    pub events: Vec<WindowInputEvent>,
}

/// Builds a typed invalid-resource error for egui bookkeeping violations.
fn invalid_resource(kind: Ir) -> dirk_rhi::Error {
    kind.into()
}

impl EguiState {
    pub fn new(device: &Rhi, properties: crate::RendererProperties) -> Result<Self> {
        let frame_allocator = BindingLayout::new(device)?;
        let texture_allocator = BindingLayout::new(device)?;
        let user_sampler = create_sampler(device, TextureOptions::LINEAR)?;
        let frames = [
            EguiFrameResources::new(device, &frame_allocator)?,
            EguiFrameResources::new(device, &frame_allocator)?,
        ];

        Ok(Self {
            ctx: Context::default(),
            target_format: properties.surface_format,
            output_is_srgb: is_srgb_format(properties.surface_format),
            modifiers: egui::Modifiers::default(),
            pipeline: Self::pipeline_for_format(device, properties.surface_format)?,
            texture_allocator,
            user_sampler,
            textures: HashMap::new(),
            frames,
            next_user_texture: 0,
            start_time: Instant::now(),
            pending: None,
            prepared: None,
            textures_to_free: std::array::from_fn(|_| Vec::new()),
        })
    }

    /// The active egui window may use a different surface format from the first window.
    pub fn set_target_format(
        &mut self,
        device: &Rhi,
        format: dirk_rhi::TextureFormat,
    ) -> Result<()> {
        if self.target_format == format {
            return Ok(());
        }
        let pipeline = Self::pipeline_for_format(device, format)?;
        self.pipeline = pipeline;
        self.target_format = format;
        self.output_is_srgb = is_srgb_format(format);
        Ok(())
    }

    fn pipeline_for_format(
        device: &Rhi,
        format: dirk_rhi::TextureFormat,
    ) -> Result<GraphicsPipeline<EguiPipelineSpec>> {
        Ok(GraphicsPipeline::build(
            device,
            crate::pipeline::graphics::PipelineSettings {
                color_format: format,
                depth: None,
                samples: SampleCount::One,
            },
        )?)
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn begin_frame(&mut self, input: &EguiFrameInput) -> Context {
        let native_pixels_per_point = input.native_pixels_per_point.max(f32::EPSILON);
        let screen_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                input.extent.width as f32 / native_pixels_per_point,
                input.extent.height as f32 / native_pixels_per_point,
            ),
        );
        let events = translate_events(
            input.window_id,
            glam::UVec2 {
                x: input.extent.width,
                y: input.extent.height,
            },
            native_pixels_per_point,
            input.events.as_slice(),
        );
        for event in &input.events {
            if event.window != input.window_id {
                continue;
            }
            match &event.event {
                InputEvent::ModifiersChanged(modifiers)
                | InputEvent::Key { modifiers, .. }
                | InputEvent::PointerButton { modifiers, .. }
                | InputEvent::Scroll { modifiers, .. } => {
                    self.modifiers = (*modifiers).into();
                }
                _ => {}
            }
        }
        if !input.focused {
            self.modifiers = egui::Modifiers::default();
        }
        let system_theme = input.theme.map(|theme| match theme {
            Theme::Dark => egui::Theme::Dark,
            Theme::Light => egui::Theme::Light,
        });
        let mut raw_input = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(self.start_time.elapsed().as_secs_f64()),
            focused: input.focused,
            modifiers: self.modifiers,
            system_theme,
            events,
            ..egui::RawInput::default()
        };
        raw_input.viewports.insert(
            ViewportId::ROOT,
            ViewportInfo {
                native_pixels_per_point: Some(native_pixels_per_point),
                inner_rect: Some(screen_rect),
                focused: Some(input.focused),
                ..ViewportInfo::default()
            },
        );

        self.ctx.begin_pass(raw_input);
        self.ctx.clone()
    }

    pub fn end_frame(&mut self) -> (egui::PlatformOutput, f32) {
        let output = self.ctx.end_pass();
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        let mut textures_delta = self
            .pending
            .take()
            .map_or_else(TexturesDelta::default, |p| p.textures_delta);
        textures_delta.append(output.textures_delta);
        self.pending = Some(EguiPaintData {
            textures_delta,
            primitives,
            pixels_per_point: output.pixels_per_point,
        });
        (output.platform_output, output.pixels_per_point)
    }

    pub fn free_textures_for_frame(&mut self, frame: usize) {
        for texture in std::mem::take(&mut self.textures_to_free[frame]) {
            self.textures.remove(&texture);
        }
    }

    pub fn add_user_texture(&mut self, device: &Rhi, view: &ImageView) -> Result<TextureId> {
        let id = loop {
            let id = TextureId::User(self.next_user_texture);
            self.next_user_texture = self
                .next_user_texture
                .checked_add(1)
                .ok_or(invalid_resource(Ir::OutOfRange))?;
            if !self.textures.contains_key(&id) {
                break id;
            }
        };
        let binding = self
            .texture_allocator
            .sampled_image(device, 0, view, &self.user_sampler)?;
        self.textures.insert(id, EguiTexture::User { binding });
        Ok(id)
    }

    pub fn remove_user_texture(&mut self, id: TextureId) {
        self.textures.remove(&id);
    }

    #[allow(clippy::cast_precision_loss)]
    pub unsafe fn prepare(
        &mut self,
        device: &Rhi,
        cmd: &mut CommandBuffer,
        extent: Extent3d,
        frame: usize,
    ) -> Result<()> {
        unsafe {
            let Some(pending) = self.pending.take() else {
                return Ok(());
            };

            for (id, delta) in &pending.textures_delta.set {
                self.set_texture(device, cmd, *id, delta)?;
            }

            let mesh = flatten_meshes(&pending.primitives)?;
            self.frames[frame].prepare(device, &mesh)?;
            let screen_size = glam::vec2(
                extent.width as f32 / pending.pixels_per_point,
                extent.height as f32 / pending.pixels_per_point,
            );
            self.frames[frame].uniform.write(&EguiUbo {
                screen_size,
                output_is_srgb: f32::from(self.output_is_srgb),
                padding: 0.0,
            })?;
            self.textures_to_free[frame].extend(pending.textures_delta.free);
            self.prepared = Some(PreparedFrame {
                frame,
                pixels_per_point: pending.pixels_per_point,
            });
            Ok(())
        }
    }

    pub fn add_pass<'a>(
        &'a self,
        graph: &mut crate::frame_graph::RenderGraph<'a>,
        target: crate::frame_graph::TextureHandle,
        extent: Extent3d,
        frame: usize,
        viewport_textures: &[crate::frame_graph::TextureHandle],
    ) -> Result<()> {
        use crate::frame_graph::{AttachmentInfo, ImportedBuffer, ImportedTexture};
        let mut images = viewport_textures.to_vec();
        let sampled = ImageState::ShaderRead(ShaderStages::FRAGMENT);
        for texture in self.textures.values() {
            if let EguiTexture::Managed { image, .. } = texture {
                images.push(graph.import_texture(ImportedTexture {
                    image: image.rhi_image(),
                    view: image.rhi_view(),
                    initial_state: sampled,
                    final_state: sampled,
                })?);
            }
        }
        let resources = &self.frames[frame];
        let mut buffers = Vec::new();
        for (buffer, access) in [
            (
                Some(resources.uniform.buffer()),
                ImageState::Uniform(ShaderStages::VERTEX),
            ),
            (resources.vertices.as_ref(), ImageState::Vertex),
            (resources.indices.as_ref(), ImageState::Index),
        ] {
            if let Some(buffer) = buffer {
                buffers.push((
                    graph.import_buffer(ImportedBuffer {
                        buffer,
                        initial_state: access,
                        final_state: access,
                    })?,
                    access,
                ));
            }
        }
        let mut pass = graph.add_pass("egui");
        pass.write_color_attachment(target, AttachmentInfo::load_store());
        for image in images {
            pass.read_sampled(image, ShaderStages::FRAGMENT);
        }
        for (buffer, access) in buffers {
            pass.read_buffer(buffer, access);
        }
        // SAFETY: retired viewport registrations are read-only and their native owners remain in the current retirement cycle.
        unsafe {
            pass.external_reads(Box::new(move |cmd, _| {
                self.render(cmd, extent, frame)?;
                Ok(())
            }));
        }
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    pub unsafe fn render(
        &self,
        cmd: &mut RenderPass<'_>,
        extent: Extent3d,
        frame: usize,
    ) -> Result<()> {
        unsafe {
            let Some(prepared) = self.prepared.as_ref() else {
                return Ok(());
            };
            if prepared.frame != frame {
                return Err(invalid_resource(Ir::BadState).into());
            }

            let resources = &self.frames[frame];
            let (Some(vertices), Some(indices)) = (&resources.vertices, &resources.indices) else {
                return Ok(());
            };
            let mut rendering = self.pipeline.bind(cmd)?;
            rendering.command().set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            rendering.command().bind_vertex_buffer(0, vertices, 0)?;
            rendering
                .command()
                .bind_index_buffer(indices, 0, IndexFormat::Uint32)?;

            for draw in &resources.draws {
                let Some(scissor) = clip_scissor(draw.clip_rect, prepared.pixels_per_point, extent)
                else {
                    continue;
                };
                let texture = self
                    .textures
                    .get(&draw.texture)
                    .ok_or(invalid_resource(Ir::BadState))?;
                rendering.bind_descriptor_sets(&(&resources.set, texture.binding()))?;
                rendering.command().set_scissor(scissor)?;
                rendering.command().draw_indexed(
                    draw.index_count,
                    1,
                    draw.first_index,
                    draw.vertex_offset,
                    0,
                )?;
            }
            Ok(())
        }
    }

    fn set_texture(
        &mut self,
        device: &Rhi,
        cmd: &mut CommandBuffer,
        id: TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> Result<()> {
        let [width, height] = delta.image.size();
        let extent = Extent3d::new_2d(
            u32::try_from(width).map_err(|_| invalid_resource(Ir::OutOfRange))?,
            u32::try_from(height).map_err(|_| invalid_resource(Ir::OutOfRange))?,
        );
        let origin = if let Some([x, y]) = delta.pos {
            Origin3d {
                x: u32::try_from(x).map_err(|_| invalid_resource(Ir::OutOfRange))?,
                y: u32::try_from(y).map_err(|_| invalid_resource(Ir::OutOfRange))?,
                z: 0,
            }
        } else {
            let image = Image::create_image(
                device,
                &ImageCreateInfo {
                    extent,
                    format: dirk_rhi::TextureFormat::Rgba8Srgb,
                    usage: ImageUsages::COPY_DST | ImageUsages::SAMPLED,
                    mip_levels: 1,
                    samples: SampleCount::One,
                },
            )?;
            let sampler = create_sampler(device, delta.options)?;
            let binding =
                self.texture_allocator
                    .sampled_image(device, 0, image.rhi_view(), &sampler)?;
            self.textures.insert(
                id,
                EguiTexture::Managed {
                    image: Box::new(image),
                    binding,
                    options: delta.options,
                    sampler,
                },
            );
            Origin3d::default()
        };
        let texture = self
            .textures
            .get_mut(&id)
            .ok_or(invalid_resource(Ir::BadState))?;
        let EguiTexture::Managed {
            image,
            binding,
            options,
            sampler,
        } = texture
        else {
            return Err(invalid_resource(Ir::BadState).into());
        };
        let pixels = match &delta.image {
            ImageData::Color(image) => image
                .pixels
                .iter()
                .flat_map(egui::Color32::to_array)
                .collect::<Vec<_>>(),
        };
        let old_state = if delta.pos.is_some() {
            ImageState::ShaderRead(ShaderStages::FRAGMENT)
        } else {
            ImageState::Undefined
        };
        // SAFETY: graphics queue ordering covers prior reads, and managed textures are renderer-owned.
        unsafe {
            Self::texture_barrier(
                cmd,
                image.rhi_image(),
                old_state,
                ImageState::CopyDestination,
            )?;
            ImageUpload {
                image: image.rhi_image(),
                mip: 0,
                origin,
                extent,
                pixels: &pixels,
            }
            .record(device, cmd)?;
            Self::texture_barrier(
                cmd,
                image.rhi_image(),
                ImageState::CopyDestination,
                ImageState::ShaderRead(ShaderStages::FRAGMENT),
            )?;
        }
        if *options != delta.options {
            let replacement = create_sampler(device, delta.options)?;
            *binding =
                self.texture_allocator
                    .sampled_image(device, 0, image.rhi_view(), &replacement)?;
            *sampler = replacement;
            *options = delta.options;
        }
        Ok(())
    }
    unsafe fn texture_barrier(
        cmd: &mut CommandBuffer,
        image: &dirk_rhi::Image,
        old_state: ImageState,
        new_state: ImageState,
    ) -> Result<()> {
        unsafe {
            cmd.barrier(&dirk_rhi::DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[dirk_rhi::ImageBarrier {
                    image,
                    old_state,
                    new_state,
                    aspects: dirk_rhi::ImageAspects::COLOR,
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    queue_transfer: None,
                }],
            })?;
        }
        Ok(())
    }
}

struct EguiPaintData {
    textures_delta: TexturesDelta,
    primitives: Vec<ClippedPrimitive>,
    pixels_per_point: f32,
}

struct PreparedFrame {
    frame: usize,
    pixels_per_point: f32,
}

struct EguiFrameResources {
    uniform: UniformBuffer<EguiUbo>,
    set: DescriptorSet<EguiFrameSet>,
    vertices: Option<dirk_rhi::Buffer>,
    indices: Option<dirk_rhi::Buffer>,
    vertex_capacity: usize,
    index_capacity: usize,
    draws: Vec<EguiDraw>,
}

impl EguiFrameResources {
    fn new(device: &Rhi, allocator: &BindingLayout<EguiFrameSet>) -> Result<Self> {
        let uniform_size =
            u64::try_from(size_of::<EguiUbo>()).map_err(|_| invalid_resource(Ir::OutOfRange))?;
        let uniform = UniformBuffer::new(device)?;
        let set = allocator.uniform_buffer(device, 0, uniform.buffer(), uniform_size)?;
        Ok(Self {
            uniform,
            set,
            vertices: None,
            indices: None,
            vertex_capacity: 0,
            index_capacity: 0,
            draws: Vec::new(),
        })
    }

    unsafe fn prepare(&mut self, device: &Rhi, mesh: &FlattenedMesh) -> Result<()> {
        unsafe {
            ensure_buffer(
                device,
                &mut self.vertices,
                &mut self.vertex_capacity,
                std::mem::size_of_val(mesh.vertices.as_slice()),
                BufferUsages::VERTEX,
            )?;
            ensure_buffer(
                device,
                &mut self.indices,
                &mut self.index_capacity,
                std::mem::size_of_val(mesh.indices.as_slice()),
                BufferUsages::INDEX,
            )?;
            if let Some(vertices) = &mut self.vertices {
                vertices.write(0, bytemuck::cast_slice(&mesh.vertices))?;
            }
            if let Some(indices) = &mut self.indices {
                indices.write(0, bytemuck::cast_slice(&mesh.indices))?;
            }
            self.draws.clone_from(&mesh.draws);
            Ok(())
        }
    }
}

enum EguiTexture {
    Managed {
        image: Box<Image>,
        binding: DescriptorSet<EguiTextureSet>,
        options: TextureOptions,
        sampler: crate::resources::Sampler,
    },
    User {
        binding: DescriptorSet<EguiTextureSet>,
    },
}

impl EguiTexture {
    fn binding(&self) -> &DescriptorSet<EguiTextureSet> {
        match self {
            Self::Managed { binding, .. } | Self::User { binding } => binding,
        }
    }
}

struct EguiFrameSet;

impl SetLayout for EguiFrameSet {
    const BINDINGS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        ty: BindingType::UniformBuffer {
            dynamic_offset: false,
        },
        visibility: ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
    }];
}

struct EguiTextureSet;

impl SetLayout for EguiTextureSet {
    const BINDINGS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        ty: BindingType::SampledImage,
        visibility: ShaderStages::FRAGMENT,
    }];
}

struct EguiPipelineSpec;

impl GraphicsPipelineSpec for EguiPipelineSpec {
    type VertexShader = EguiVS;
    type FragmentShader = EguiFS;
    type Input = EguiVertex;
    type DescriptorSets = (EguiFrameSet, EguiTextureSet);

    const NAME: &'static str = "egui";

    fn raster() -> RasterState {
        RasterState {
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Clockwise,
            cull_mode: CullMode::None,
        }
    }

    fn blend() -> Option<BlendState> {
        Some(BlendState {
            color: BlendComponent {
                source: BlendFactor::One,
                destination: BlendFactor::OneMinusSourceAlpha,
                operation: BlendOp::Add,
            },
            alpha: BlendComponent {
                source: BlendFactor::OneMinusDestinationAlpha,
                destination: BlendFactor::One,
                operation: BlendOp::Add,
            },
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EguiVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: [f32; 4],
}

impl crate::shaders::metadata::VertexInput for EguiVertex {
    #[allow(clippy::cast_possible_truncation)]
    const ATTRIBUTES: &'static [dirk_rhi::VertexAttribute] = &[
        dirk_rhi::VertexAttribute {
            location: 0,
            format: dirk_rhi::VertexFormat::Float32x2,
            offset: std::mem::offset_of!(Self, position) as u32,
        },
        dirk_rhi::VertexAttribute {
            location: 1,
            format: dirk_rhi::VertexFormat::Float32x2,
            offset: std::mem::offset_of!(Self, tex_coord) as u32,
        },
        dirk_rhi::VertexAttribute {
            location: 2,
            format: dirk_rhi::VertexFormat::Float32x4,
            offset: std::mem::offset_of!(Self, color) as u32,
        },
    ];
}

struct FlattenedMesh {
    vertices: Vec<EguiVertex>,
    indices: Vec<u32>,
    draws: Vec<EguiDraw>,
}

#[derive(Clone)]
struct EguiDraw {
    clip_rect: egui::Rect,
    texture: TextureId,
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
}

fn flatten_meshes(primitives: &[ClippedPrimitive]) -> Result<FlattenedMesh> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut draws = Vec::new();
    for primitive in primitives {
        let Primitive::Mesh(mesh) = &primitive.primitive else {
            warn!("egui callback primitives are not supported");
            continue;
        };
        let first_index =
            u32::try_from(indices.len()).map_err(|_| invalid_resource(Ir::OutOfRange))?;
        let index_count =
            u32::try_from(mesh.indices.len()).map_err(|_| invalid_resource(Ir::OutOfRange))?;
        let vertex_offset =
            i32::try_from(vertices.len()).map_err(|_| invalid_resource(Ir::OutOfRange))?;
        vertices.extend(mesh.vertices.iter().map(|vertex| {
            let color = vertex.color.to_array();
            EguiVertex {
                position: [vertex.pos.x, vertex.pos.y],
                tex_coord: [vertex.uv.x, vertex.uv.y],
                color: color.map(|channel| f32::from(channel) / 255.0),
            }
        }));
        indices.extend_from_slice(&mesh.indices);
        draws.push(EguiDraw {
            clip_rect: primitive.clip_rect,
            texture: mesh.texture_id,
            first_index,
            index_count,
            vertex_offset,
        });
    }
    Ok(FlattenedMesh {
        vertices,
        indices,
        draws,
    })
}

fn ensure_buffer(
    device: &Rhi,
    buffer: &mut Option<dirk_rhi::Buffer>,
    capacity: &mut usize,
    required: usize,
    usage: BufferUsages,
) -> Result<()> {
    if required == 0 || required <= *capacity {
        return Ok(());
    }
    let new_capacity = required.next_power_of_two();
    *buffer = Some(device.create_buffer(&dirk_rhi::BufferDesc {
        label: "egui frame buffer",
        size: u64::try_from(new_capacity).map_err(|_| invalid_resource(Ir::OutOfRange))?,
        usage,
        memory: MemoryDomain::Upload,
    })?);
    *capacity = new_capacity;
    Ok(())
}

fn create_sampler(device: &Rhi, options: TextureOptions) -> Result<crate::resources::Sampler> {
    let filter = |filter| match filter {
        TextureFilter::Nearest => FilterMode::Nearest,
        TextureFilter::Linear => FilterMode::Linear,
    };
    let address = match options.wrap_mode {
        TextureWrapMode::ClampToEdge => AddressMode::ClampToEdge,
        TextureWrapMode::Repeat => AddressMode::Repeat,
        TextureWrapMode::MirroredRepeat => AddressMode::MirrorRepeat,
    };
    Ok(device.create_sampler(&SamplerDesc {
        label: "egui texture sampler",
        mag_filter: filter(options.magnification),
        min_filter: filter(options.minification),
        mip_filter: filter(options.mipmap_mode.unwrap_or(options.minification)),
        address_u: address,
        address_v: address,
        address_w: address,
        max_anisotropy: 1,
        lod_min: 0.0,
        lod_max: 1.0,
    })?)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn clip_scissor(clip: egui::Rect, pixels_per_point: f32, extent: Extent3d) -> Option<Rect> {
    let width = extent.width as f32;
    let height = extent.height as f32;
    let min_x = (clip.min.x * pixels_per_point).floor().clamp(0.0, width);
    let min_y = (clip.min.y * pixels_per_point).floor().clamp(0.0, height);
    let max_x = (clip.max.x * pixels_per_point).ceil().clamp(min_x, width);
    let max_y = (clip.max.y * pixels_per_point).ceil().clamp(min_y, height);
    let scissor = Rect {
        x: min_x as i32,
        y: min_y as i32,
        width: (max_x - min_x) as u32,
        height: (max_y - min_y) as u32,
    };
    (scissor.width > 0 && scissor.height > 0).then_some(scissor)
}

fn is_srgb_format(format: dirk_rhi::TextureFormat) -> bool {
    matches!(
        format,
        dirk_rhi::TextureFormat::Rgba8Srgb | dirk_rhi::TextureFormat::Bgra8Srgb
    )
}

fn translate_events(
    window_id: WindowId,
    extent: glam::UVec2,
    native_pixels_per_point: f32,
    events: &[WindowInputEvent],
) -> Vec<egui::Event> {
    let mut translated = Vec::new();
    for event in events {
        if event.window != window_id {
            continue;
        }
        append_translated_event(
            &mut translated,
            extent,
            native_pixels_per_point,
            &event.event,
        );
    }
    translated
}

fn append_translated_event(
    out: &mut Vec<egui::Event>,
    extent: glam::UVec2,
    native_pixels_per_point: f32,
    event: &InputEvent,
) {
    match event {
        InputEvent::Key {
            key,
            state,
            repeat,
            modifiers,
        } => {
            let modifiers = egui::Modifiers::from(*modifiers);
            if let Some(key) = key.to_egui() {
                out.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: *state == ButtonState::Pressed,
                    repeat: *repeat,
                    modifiers,
                });
            }
        }
        InputEvent::Text(text) => out.push(egui::Event::Text(text.clone())),
        InputEvent::Ime(event) => out.push(egui::Event::Ime(match event {
            ImeEvent::Enabled => egui::ImeEvent::Enabled,
            ImeEvent::Preedit(text) => egui::ImeEvent::Preedit(text.clone()),
            ImeEvent::Commit(text) => egui::ImeEvent::Commit(text.clone()),
            ImeEvent::Disabled => egui::ImeEvent::Disabled,
        })),
        InputEvent::ModifiersChanged(_) | InputEvent::PointerEntered => {}
        InputEvent::PointerMoved { position, .. } => {
            out.push(egui::Event::PointerMoved(
                position.to_egui(extent, native_pixels_per_point),
            ));
        }
        InputEvent::PointerLeft => {
            out.push(egui::Event::PointerGone);
        }
        InputEvent::PointerButton {
            button,
            state,
            position,
            modifiers,
        } => {
            out.push(egui::Event::PointerButton {
                pos: position.to_egui(extent, native_pixels_per_point),
                button: egui::PointerButton::from(*button),
                pressed: *state == ButtonState::Pressed,
                modifiers: egui::Modifiers::from(*modifiers),
            });
        }
        InputEvent::Scroll {
            delta,
            unit,
            modifiers,
        } => {
            out.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::from(*unit),
                delta: delta.to_egui(extent, native_pixels_per_point),
                modifiers: egui::Modifiers::from(*modifiers),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirk_input::{
        LogicalKey, Modifiers, NamedKey, NormalizedDelta, NormalizedPosition, PointerButton,
        ScrollUnit,
    };
    use egui::{Pos2, Vec2};

    fn window_id(raw: usize) -> WindowId {
        WindowId::from_raw(raw)
    }

    #[test]
    fn detects_common_srgb_formats() {
        assert!(is_srgb_format(dirk_rhi::TextureFormat::Rgba8Srgb));
        assert!(is_srgb_format(dirk_rhi::TextureFormat::Bgra8Srgb));
        assert!(!is_srgb_format(dirk_rhi::TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn pipeline_matches_reflected_shaders() {
        EguiPipelineSpec::validate().expect("egui pipeline metadata should match its shaders");
    }

    #[test]
    fn clips_scissors_to_the_render_target() {
        assert_eq!(
            clip_scissor(
                egui::Rect::from_min_max(egui::pos2(-5.0, 2.0), egui::pos2(80.0, 40.0)),
                2.0,
                Extent3d::new_2d(100, 50),
            ),
            Some(Rect {
                x: 0,
                y: 4,
                width: 100,
                height: 46,
            })
        );
    }

    #[test]
    fn pointer_positions_are_scaled_from_normalized_to_points() {
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 200, y: 100 },
            2.0,
            &[WindowInputEvent {
                window: window_id(1),
                event: InputEvent::PointerMoved {
                    position: NormalizedPosition::new(glam::vec2(0.5, 0.5)),
                    delta: NormalizedDelta(glam::Vec2::ZERO),
                },
            }],
        );

        assert_eq!(
            events,
            vec![egui::Event::PointerMoved(Pos2::new(50.0, 25.0))]
        );
    }

    #[test]
    fn committed_text_preserves_case_separately_from_binding_key() {
        let modifiers = Modifiers::default();
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Key {
                        key: LogicalKey::character("A"),
                        state: ButtonState::Pressed,
                        repeat: false,
                        modifiers,
                    },
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Text("A".to_owned()),
                },
            ],
        );

        assert_eq!(
            events,
            vec![
                egui::Event::Key {
                    key: egui::Key::A,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: modifiers.into(),
                },
                egui::Event::Text("A".to_owned()),
            ]
        );
    }

    #[test]
    fn space_key_press_emits_text_event() {
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Key {
                        key: LogicalKey::Named(NamedKey::Space),
                        state: ButtonState::Pressed,
                        repeat: false,
                        modifiers: Modifiers::default(),
                    },
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Text(" ".to_owned()),
                },
            ],
        );

        assert!(events.contains(&egui::Event::Text(" ".to_owned())));
    }

    #[test]
    fn key_events_never_invent_text_but_repeated_commits_are_inserted() {
        let ctrl_modifiers = Modifiers {
            ctrl: true,
            ..Modifiers::default()
        };
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Text("a".to_owned()),
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Key {
                        key: LogicalKey::character("a"),
                        state: ButtonState::Pressed,
                        repeat: true,
                        modifiers: Modifiers::default(),
                    },
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Key {
                        key: LogicalKey::character("b"),
                        state: ButtonState::Released,
                        repeat: false,
                        modifiers: Modifiers::default(),
                    },
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Key {
                        key: LogicalKey::character("c"),
                        state: ButtonState::Pressed,
                        repeat: false,
                        modifiers: ctrl_modifiers,
                    },
                },
            ],
        );

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, egui::Event::Text(_)))
                .collect::<Vec<_>>(),
            vec![&egui::Event::Text("a".to_owned())]
        );
    }

    #[test]
    fn ime_composition_and_commit_reach_egui() {
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Ime(ImeEvent::Enabled),
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Ime(ImeEvent::Preedit("あ".to_owned())),
                },
                WindowInputEvent {
                    window: window_id(1),
                    event: InputEvent::Ime(ImeEvent::Commit("あ".to_owned())),
                },
            ],
        );
        assert_eq!(
            events,
            vec![
                egui::Event::Ime(egui::ImeEvent::Enabled),
                egui::Event::Ime(egui::ImeEvent::Preedit("あ".to_owned())),
                egui::Event::Ime(egui::ImeEvent::Commit("あ".to_owned())),
            ]
        );
    }

    #[test]
    fn pointer_buttons_preserve_modifiers() {
        let modifiers = Modifiers {
            alt: true,
            ctrl: true,
            shift: true,
            super_key: false,
        };
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[WindowInputEvent {
                window: window_id(1),
                event: InputEvent::PointerButton {
                    button: PointerButton::Primary,
                    state: ButtonState::Pressed,
                    position: NormalizedPosition::new(glam::vec2(0.25, 0.75)),
                    modifiers,
                },
            }],
        );

        assert_eq!(
            events,
            vec![egui::Event::PointerButton {
                pos: Pos2::new(25.0, 75.0),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: modifiers.into(),
            }]
        );
    }

    #[test]
    fn scroll_events_preserve_unit_and_modifiers() {
        let modifiers = Modifiers {
            alt: false,
            ctrl: true,
            shift: false,
            super_key: false,
        };
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 2, y: 4 },
            1.0,
            &[WindowInputEvent {
                window: window_id(1),
                event: InputEvent::Scroll {
                    delta: NormalizedDelta(glam::vec2(0.5, 0.25)),
                    unit: ScrollUnit::Line,
                    modifiers,
                },
            }],
        );

        assert_eq!(
            events,
            vec![egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: Vec2::new(1.0, 1.0),
                modifiers: modifiers.into(),
            }]
        );
    }
}
