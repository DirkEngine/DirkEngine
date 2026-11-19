//! The editor's immediate-mode UI: egui's input, frames, and rendering on
//! the RHI.
//!
//! egui's colors are sRGB-encoded with premultiplied alpha. The renderer
//! turns them into linear, premultiplied colors (unpremultiplying before
//! decoding, as `ecolor` does) and composites the UI in linear space, into
//! an image the window's presentation encodes once.

use std::{collections::HashMap, ops::Range, time::Instant};

use dirk_engine::editor::{EditorRenderContext, EditorServices};
use dirk_input::{ButtonState, InputEvent};
use dirk_platform::{CursorIcon, ImeArea, Theme, UiState, WindowId, WindowInputEvent};
use dirk_render_utils::{Handle, RenderGraph, Uploads};
use dirk_rhi::{
    AddressMode, BlendState, Buffer, BufferDesc, BufferUsage, ColorTarget, CullMode, Extent3d,
    Filter, Frame, FrontFace, GraphicsPipeline, GraphicsPipelineDesc, Image, ImageAccess,
    ImageDesc, ImageDimension, ImageRegion, ImageUsage, IndexFormat, Indices, LoadOp, Memory,
    Origin3d, PerFrame, Rect, Rhi, SampleCount, Sampler, SamplerDesc, SamplerIndex, ShaderStages,
    TextureIndex, Topology,
};
use dirk_shaders::types::{EguiPush, EguiVertex};
use dirk_universe::Universe;
use egui::{
    ClippedPrimitive, Color32, Context, Rgba, TextureFilter, TextureId, TextureOptions,
    TextureWrapMode, TexturesDelta, ViewportId, ViewportInfo,
    epaint::{ImageData, ImageDelta, Primitive, Vertex},
};

use crate::{
    COLOR_FORMAT, Error, Result, Sampled, StorageBuffer,
    shaders::{EguiFs, EguiVs},
};

// egui's vertices are uploaded as they are, so their layout must be the one
// the shaders read.
const _: () = assert!(
    size_of::<Vertex>() == size_of::<EguiVertex>()
        && std::mem::offset_of!(Vertex, pos) == std::mem::offset_of!(EguiVertex, position)
        && std::mem::offset_of!(Vertex, uv) == std::mem::offset_of!(EguiVertex, uv)
        && std::mem::offset_of!(Vertex, color) == std::mem::offset_of!(EguiVertex, color)
);

/// Bytes the per-frame vertex and index buffers hold at first.
const INITIAL_BUFFER_SIZE: u64 = 1 << 16;

/// How the UI's passes leave egui's textures between frames.
const SAMPLED: ImageAccess = ImageAccess::Sampled(ShaderStages::FRAGMENT);

/// The editor UI: egui's context, the input it is fed, and its rendering.
pub struct EguiState {
    ctx: Context,
    start_time: Instant,
    /// What the last frame painted, until rendered.
    pending: Option<EguiPaintData>,
    input: EguiInputState,
    renderer: EguiRenderer,
}

pub struct EguiFrameInput {
    pub window_id: WindowId,
    /// Size of the window in pixels.
    pub extent: (u32, u32),
    pub native_pixels_per_point: f32,
    pub focused: bool,
    pub theme: Option<Theme>,
    pub events: Vec<WindowInputEvent>,
}

/// What an egui frame painted.
struct EguiPaintData {
    textures_delta: TexturesDelta,
    primitives: Vec<ClippedPrimitive>,
    pixels_per_point: f32,
    /// Size of the window it painted, in pixels.
    extent: (u32, u32),
}

impl EguiState {
    pub fn new(rhi: &Rhi) -> Result<Self> {
        Ok(Self {
            ctx: Context::default(),
            start_time: Instant::now(),
            pending: None,
            input: EguiInputState::default(),
            renderer: EguiRenderer::new(rhi)?,
        })
    }

    /// Runs the editor UI for one frame, keeping its paint data for
    /// rendering, and returns the cursor and input method state the UI
    /// requests from its window.
    #[expect(
        clippy::cast_precision_loss,
        reason = "window sizes are far below 2^24"
    )]
    pub fn run_frame(
        &mut self,
        input: &EguiFrameInput,
        editor: &EditorServices,
        context: &EditorRenderContext<'_>,
        universe: &Universe,
    ) -> anyhow::Result<UiState> {
        let native_pixels_per_point = input.native_pixels_per_point.max(f32::EPSILON);
        let (width, height) = input.extent;
        let screen_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                width as f32 / native_pixels_per_point,
                height as f32 / native_pixels_per_point,
            ),
        );
        let events = self.input.translate_events(
            input.window_id,
            glam::uvec2(width, height),
            native_pixels_per_point,
            input.events.as_slice(),
        );
        let system_theme = input.theme.map(|theme| match theme {
            Theme::Dark => egui::Theme::Dark,
            Theme::Light => egui::Theme::Light,
        });
        let mut raw_input = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(self.start_time.elapsed().as_secs_f64()),
            focused: input.focused,
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

        let output = editor.render_ui(&self.ctx, raw_input, context, universe)?;
        let pixels_per_point = output.pixels_per_point;
        // Texture changes of a frame that was never rendered still apply.
        let mut textures_delta = self
            .pending
            .take()
            .map_or_default(|pending| pending.textures_delta);
        textures_delta.append(output.textures_delta);
        self.pending = Some(EguiPaintData {
            textures_delta,
            primitives: self.ctx.tessellate(output.shapes, pixels_per_point),
            pixels_per_point,
            extent: input.extent,
        });

        let platform = output.platform_output;
        Ok(UiState {
            cursor: platform_cursor(platform.cursor_icon),
            ime: platform.ime.map(|ime| {
                let rect = ime.cursor_rect * pixels_per_point;
                ImeArea {
                    x: f64::from(rect.min.x),
                    y: f64::from(rect.min.y),
                    width: f64::from(rect.width()),
                    height: f64::from(rect.height()),
                }
            }),
        })
    }

    /// Uploads the textures the last frame set, with `uploads`. Done every
    /// frame, even when the UI is not shown, so no change is lost.
    pub fn update_textures(&mut self, rhi: &Rhi, uploads: &mut Uploads<'_>) -> Result<()> {
        match &mut self.pending {
            Some(pending) => self
                .renderer
                .set_textures(rhi, uploads, &mut pending.textures_delta),
            None => Ok(()),
        }
    }

    /// Adds the pass drawing the last frame's UI to `graph`, onto a new
    /// image of `extent` in linear colors, which it returns. `viewports`
    /// are the outputs `graph` renders, which the UI may show.
    pub fn record<'a>(
        &'a mut self,
        graph: &mut RenderGraph<'a>,
        frame: &Frame<'_>,
        extent: (u32, u32),
        viewports: impl IntoIterator<Item = Handle<Image>>,
    ) -> Result<Handle<Image>> {
        let (primitives, pixels_per_point, painted) = match &mut self.pending {
            Some(pending) => (
                std::mem::take(&mut pending.primitives),
                pending.pixels_per_point,
                pending.extent,
            ),
            None => (Vec::new(), 1.0, extent),
        };
        self.renderer
            .prepare(frame, &primitives, pixels_per_point, painted, extent)?;
        self.renderer.record(graph, frame, extent, viewports)
    }

    /// Frees the textures the last frame no longer uses, once its UI is
    /// recorded or skipped.
    pub fn finish_frame(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.renderer.free_textures(&pending.textures_delta);
        }
    }
}

/// Draws egui's meshes with the bindless heap.
pub struct EguiRenderer {
    pipeline: GraphicsPipeline<EguiPush>,
    textures: HashMap<TextureId, EguiTexture>,
    /// One sampler per texture options in use.
    samplers: HashMap<TextureOptions, Sampler>,
    /// Each frame's vertices and indices, every mesh after the other.
    vertices: PerFrame<Buffer>,
    indices: PerFrame<Buffer>,
    /// The draws [`prepare`](Self::prepare) set up for its frame.
    draws: Vec<EguiDraw>,
    /// Size of the screen in points, for the draws.
    screen_size: [f32; 2],
}

/// A texture egui manages.
struct EguiTexture {
    image: Image,
    sampler: SamplerIndex,
}

/// A mesh, drawn from the frame's vertex and index buffers.
struct EguiDraw {
    scissor: Rect,
    texture: TextureIndex,
    /// Whether `texture` comes from a user texture id, which is only drawn
    /// if it names a viewport the pass samples.
    user: bool,
    sampler: SamplerIndex,
    indices: Range<u32>,
    base_vertex: i32,
}

impl EguiRenderer {
    pub fn new(rhi: &Rhi) -> Result<Self> {
        let vertex = rhi.create_shader::<EguiVs, EguiPush>()?;
        let fragment = rhi.create_shader::<EguiFs, EguiPush>()?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: "egui",
            vertex: &vertex,
            fragment: Some(&fragment),
            topology: Topology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            color_targets: &[ColorTarget {
                blend: Some(BlendState::PREMULTIPLIED_ALPHA),
                ..ColorTarget::new(COLOR_FORMAT)
            }],
            depth: None,
            samples: SampleCount::X1,
        })?;
        let buffers = |label, usage| {
            PerFrame::try_new(rhi, |_| {
                Self::create_buffer(rhi, label, usage, INITIAL_BUFFER_SIZE)
            })
        };
        let mut renderer = Self {
            pipeline,
            textures: HashMap::new(),
            samplers: HashMap::new(),
            vertices: buffers("egui vertices", BufferUsage::STORAGE)?,
            indices: buffers("egui indices", BufferUsage::INDEX)?,
            draws: Vec::new(),
            screen_size: [1.0; 2],
        };
        // User textures, such as viewports, use linear filtering.
        renderer.sampler(rhi, TextureOptions::LINEAR)?;
        Ok(renderer)
    }

    /// Applies the texture changes in `delta` with `uploads`, creating new
    /// textures and writing updated parts into existing ones, and leaves no
    /// changes in it.
    pub fn set_textures(
        &mut self,
        rhi: &Rhi,
        uploads: &mut Uploads<'_>,
        delta: &mut TexturesDelta,
    ) -> Result<()> {
        for (id, changes) in std::mem::take(&mut delta.set) {
            for ImageDelta {
                image: ImageData::Color(image),
                options,
                pos,
            } in changes
            {
                let texels = EguiTexture::texels(&image.pixels);
                let sampler = self.sampler(rhi, options)?;
                let Some([x, y]) = pos else {
                    let texture = EguiTexture {
                        image: EguiTexture::upload(uploads, image.size, &texels)?,
                        sampler,
                    };
                    // The replaced image is freed once the GPU is done with it.
                    self.textures.insert(id, texture);
                    continue;
                };
                let texture = self
                    .textures
                    .get_mut(&id)
                    .ok_or(Error::UnknownEguiTexture(id))?;
                texture.sampler = sampler;
                let [width, height] = image.size.map(EguiTexture::texel_count);
                let region = ImageRegion {
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: 1,
                    origin: Origin3d {
                        x: EguiTexture::texel_count(x),
                        y: EguiTexture::texel_count(y),
                        z: 0,
                    },
                    extent: Extent3d::new_2d(width, height),
                };
                uploads.write_image(&texture.image, SAMPLED, region, &texels)?;
            }
        }
        Ok(())
    }

    /// Drops the textures `delta` frees.
    pub fn free_textures(&mut self, delta: &TexturesDelta) {
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    /// Writes `primitives`, painted at `pixels_per_point` for a window of
    /// `painted` pixels, to `frame`'s buffers and sets up their draws onto
    /// a target of `extent` pixels.
    pub fn prepare(
        &mut self,
        frame: &Frame<'_>,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
        painted: (u32, u32),
        extent: (u32, u32),
    ) -> Result<()> {
        #[expect(
            clippy::cast_precision_loss,
            reason = "window sizes are far below 2^24"
        )]
        let screen_size = [painted.0 as f32, painted.1 as f32].map(|size| size / pixels_per_point);
        self.screen_size = screen_size;
        self.draws.clear();
        let meshes = primitives
            .iter()
            .filter_map(|primitive| match &primitive.primitive {
                Primitive::Mesh(mesh) => Some((primitive.clip_rect, mesh)),
                Primitive::Callback(_) => None,
            });
        let vertex_bytes: usize = meshes
            .clone()
            .map(|(_, mesh)| size_of_val(mesh.vertices.as_slice()))
            .sum();
        let index_bytes: usize = meshes
            .clone()
            .map(|(_, mesh)| size_of_val(mesh.indices.as_slice()))
            .sum();
        Self::reserve(
            frame,
            &mut self.vertices,
            "egui vertices",
            BufferUsage::STORAGE,
            vertex_bytes,
        )?;
        Self::reserve(
            frame,
            &mut self.indices,
            "egui indices",
            BufferUsage::INDEX,
            index_bytes,
        )?;

        let (mut vertex_offset, mut index_offset) = (0_u32, 0_u32);
        for (clip, mesh) in meshes {
            let vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(u32::MAX);
            let index_count = u32::try_from(mesh.indices.len()).unwrap_or(u32::MAX);
            self.vertices.write(
                frame,
                u64::from(vertex_offset) * size_of::<Vertex>() as u64,
                bytemuck::cast_slice(&mesh.vertices),
            )?;
            self.indices.write(
                frame,
                u64::from(index_offset) * size_of::<u32>() as u64,
                bytemuck::cast_slice(&mesh.indices),
            )?;
            let texture = match mesh.texture_id {
                TextureId::Managed(_) => self
                    .textures
                    .get(&mesh.texture_id)
                    .map(|texture| {
                        texture
                            .image
                            .texture()
                            .map(|index| (index, texture.sampler))
                    })
                    .transpose()?,
                // User textures are the bindless indices of viewport outputs,
                // checked when the pass records.
                TextureId::User(slot) => u32::try_from(slot).ok().map(|slot| {
                    (
                        TextureIndex::new(slot),
                        self.samplers[&TextureOptions::LINEAR].index(),
                    )
                }),
            };
            if let (Some((texture, sampler)), Some(scissor)) =
                (texture, Self::scissor(clip, pixels_per_point, extent))
            {
                self.draws.push(EguiDraw {
                    scissor,
                    texture,
                    user: matches!(mesh.texture_id, TextureId::User(_)),
                    sampler,
                    indices: index_offset..index_offset + index_count,
                    base_vertex: i32::try_from(vertex_offset).unwrap_or(i32::MAX),
                });
            }
            vertex_offset += vertex_count;
            index_offset += index_count;
        }
        Ok(())
    }

    /// Adds the pass drawing what [`prepare`](Self::prepare) set up for
    /// `frame` to `graph`, onto a new image of `extent` cleared to black,
    /// which it returns. The pass samples `viewports`, outputs `graph`
    /// renders that user textures may show; meshes whose user texture names
    /// none of them, such as a stale one, are skipped.
    pub fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame: &Frame<'_>,
        (width, height): (u32, u32),
        viewports: impl IntoIterator<Item = Handle<Image>>,
    ) -> Result<Handle<Image>> {
        let target = graph.create_image(&ImageDesc {
            label: "ui",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(width, height),
            format: COLOR_FORMAT,
            usage: ImageUsage::empty(),
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
        });
        let vertices = self.vertices.get(frame);
        let indices = self.indices.get(frame);
        let viewports: Vec<_> = viewports.into_iter().collect();
        viewports
            .iter()
            .fold(graph.pass("egui"), |pass, &viewport| {
                pass.sampled(viewport, ShaderStages::FRAGMENT)
            })
            .color(target, LoadOp::Clear([0.0, 0.0, 0.0, 1.0]))
            .record(move |pass| {
                let vertices = vertices.storage()?;
                let shown = viewports
                    .iter()
                    .map(|&viewport| pass.image(viewport)?.texture())
                    .collect::<dirk_rhi::Result<Vec<_>>>()?;
                for draw in &self.draws {
                    if draw.user && !shown.contains(&draw.texture) {
                        continue;
                    }
                    pass.set_scissor(draw.scissor)?;
                    let push = EguiPush {
                        screen_size: self.screen_size,
                        vertices,
                        texture: draw.texture,
                        sampler: draw.sampler,
                    };
                    let indices = Indices {
                        buffer: indices,
                        format: IndexFormat::U32,
                        offset: 0,
                        range: draw.indices.clone(),
                        base_vertex: draw.base_vertex,
                    };
                    pass.draw_indexed(&self.pipeline, &push, indices, 0..1)?;
                }
                Ok(())
            })?;
        Ok(target)
    }

    /// The sampler for textures with `options`, created on first use.
    fn sampler(&mut self, rhi: &Rhi, options: TextureOptions) -> Result<SamplerIndex> {
        if let Some(sampler) = self.samplers.get(&options) {
            return Ok(sampler.index());
        }
        let filter = |filter| match filter {
            TextureFilter::Nearest => Filter::Nearest,
            TextureFilter::Linear => Filter::Linear,
        };
        let address = match options.wrap_mode {
            TextureWrapMode::ClampToEdge => AddressMode::ClampToEdge,
            TextureWrapMode::Repeat => AddressMode::Repeat,
            TextureWrapMode::MirroredRepeat => AddressMode::MirrorRepeat,
        };
        let sampler = rhi.create_sampler(&SamplerDesc {
            label: "egui",
            mag_filter: filter(options.magnification),
            min_filter: filter(options.minification),
            mipmap_filter: filter(options.mipmap_mode.unwrap_or(TextureFilter::Nearest)),
            address_modes: [address; 3],
            ..SamplerDesc::default()
        })?;
        let index = sampler.index();
        self.samplers.insert(options, sampler);
        Ok(index)
    }

    /// The pixel rectangle of `clip`, in points at `pixels_per_point`,
    /// inside a target of `extent`, or `None` if empty.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the corners are rounded and clamped to the target's size"
    )]
    fn scissor(
        clip: egui::Rect,
        pixels_per_point: f32,
        (width, height): (u32, u32),
    ) -> Option<Rect> {
        let corner = |point: egui::Pos2| {
            let x = (point.x * pixels_per_point)
                .round()
                .clamp(0.0, width as f32) as u32;
            let y = (point.y * pixels_per_point)
                .round()
                .clamp(0.0, height as f32) as u32;
            (x, y)
        };
        let (min, max) = (corner(clip.min), corner(clip.max));
        (max.0 > min.0 && max.1 > min.1).then(|| Rect {
            x: min.0,
            y: min.1,
            width: max.0 - min.0,
            height: max.1 - min.1,
        })
    }

    /// Makes `frame`'s buffer in `buffers` hold at least `size` bytes.
    fn reserve(
        frame: &Frame<'_>,
        buffers: &mut PerFrame<Buffer>,
        label: &str,
        usage: BufferUsage,
        size: usize,
    ) -> Result<()> {
        let size = size as u64;
        let buffer = buffers.get_mut(frame);
        if size > buffer.info().size {
            *buffer = Self::create_buffer(frame.rhi(), label, usage, size.next_power_of_two())?;
        }
        Ok(())
    }

    /// Creates a host-written buffer of `size` bytes.
    fn create_buffer(
        rhi: &Rhi,
        label: &str,
        usage: BufferUsage,
        size: u64,
    ) -> dirk_rhi::Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label,
            size,
            usage,
            memory: Memory::Upload,
        })
    }
}

impl EguiTexture {
    /// Linear, premultiplied, sRGB-encoded texels of egui's `colors`.
    fn texels(colors: &[Color32]) -> Vec<u8> {
        colors
            .iter()
            .flat_map(|&color| {
                let [r, g, b, a] = Rgba::from(color).to_array();
                [
                    egui::ecolor::gamma_u8_from_linear_f32(r),
                    egui::ecolor::gamma_u8_from_linear_f32(g),
                    egui::ecolor::gamma_u8_from_linear_f32(b),
                    egui::ecolor::linear_u8_from_linear_f32(a),
                ]
            })
            .collect()
    }

    /// A count or position of egui's texels as the RHI's. Ones too large
    /// for it are rejected when uploading.
    fn texel_count(count: usize) -> u32 {
        u32::try_from(count).unwrap_or(u32::MAX)
    }

    /// Uploads `texels`, `size` texels, as a texture fragment shaders
    /// sample.
    fn upload(uploads: &mut Uploads<'_>, size: [usize; 2], texels: &[u8]) -> Result<Image> {
        let [width, height] = size.map(Self::texel_count);
        Ok(uploads.image(
            &ImageDesc {
                label: "egui texture",
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(width, height),
                format: COLOR_FORMAT,
                usage: ImageUsage::SAMPLED,
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::X1,
            },
            SAMPLED,
            &[texels],
        )?)
    }
}

/// Maps an egui cursor to the platform cursor, or `None` to hide it.
fn platform_cursor(icon: egui::CursorIcon) -> Option<CursorIcon> {
    use egui::CursorIcon as Ui;
    Some(match icon {
        Ui::None => return None,
        Ui::Default => CursorIcon::Default,
        Ui::ContextMenu => CursorIcon::ContextMenu,
        Ui::Help => CursorIcon::Help,
        Ui::PointingHand => CursorIcon::Pointer,
        Ui::Progress => CursorIcon::Progress,
        Ui::Wait => CursorIcon::Wait,
        Ui::Cell => CursorIcon::Cell,
        Ui::Crosshair => CursorIcon::Crosshair,
        Ui::Text => CursorIcon::Text,
        Ui::VerticalText => CursorIcon::VerticalText,
        Ui::Alias => CursorIcon::Alias,
        Ui::Copy => CursorIcon::Copy,
        Ui::Move => CursorIcon::Move,
        Ui::NoDrop => CursorIcon::NoDrop,
        Ui::NotAllowed => CursorIcon::NotAllowed,
        Ui::Grab => CursorIcon::Grab,
        Ui::Grabbing => CursorIcon::Grabbing,
        Ui::AllScroll => CursorIcon::AllScroll,
        Ui::ResizeHorizontal => CursorIcon::EwResize,
        Ui::ResizeNeSw => CursorIcon::NeswResize,
        Ui::ResizeNwSe => CursorIcon::NwseResize,
        Ui::ResizeVertical => CursorIcon::NsResize,
        Ui::ResizeEast => CursorIcon::EResize,
        Ui::ResizeSouthEast => CursorIcon::SeResize,
        Ui::ResizeSouth => CursorIcon::SResize,
        Ui::ResizeSouthWest => CursorIcon::SwResize,
        Ui::ResizeWest => CursorIcon::WResize,
        Ui::ResizeNorthWest => CursorIcon::NwResize,
        Ui::ResizeNorth => CursorIcon::NResize,
        Ui::ResizeNorthEast => CursorIcon::NeResize,
        Ui::ResizeColumn => CursorIcon::ColResize,
        Ui::ResizeRow => CursorIcon::RowResize,
        Ui::ZoomIn => CursorIcon::ZoomIn,
        Ui::ZoomOut => CursorIcon::ZoomOut,
    })
}

/// Input state kept across frames so focus loss can cancel native button presses.
#[derive(Default)]
struct EguiInputState {
    modifiers: egui::Modifiers,
    pressed_buttons: Vec<egui::PointerButton>,
}

impl EguiInputState {
    fn translate_events(
        &mut self,
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
            self.append_translated_event(
                &mut translated,
                extent,
                native_pixels_per_point,
                &event.event,
            );
        }
        translated
    }

    fn release_focus(&mut self, out: &mut Vec<egui::Event>) {
        self.modifiers = egui::Modifiers::default();
        out.push(egui::Event::ModifiersChanged(self.modifiers));
        if !self.pressed_buttons.is_empty() {
            // Move outside the UI before releasing, so cancellation
            // cannot turn a pending press into a click or drop.
            let pos = egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY);
            out.push(egui::Event::PointerMoved(pos));
            for button in self.pressed_buttons.drain(..) {
                out.push(egui::Event::PointerButton {
                    pos,
                    button,
                    pressed: false,
                    modifiers: self.modifiers,
                });
            }
        }
        out.push(egui::Event::PointerGone);
    }

    fn append_translated_event(
        &mut self,
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
            InputEvent::Ime(event) => out.extend(event.clone().to_egui().map(egui::Event::Ime)),
            InputEvent::PointerMoved { position, .. } => {
                out.push(egui::Event::PointerMoved(
                    position.to_egui(extent, native_pixels_per_point),
                ));
            }
            InputEvent::PointerEntered => {}
            InputEvent::ModifiersChanged(modifiers) => {
                self.modifiers = (*modifiers).into();
                out.push(egui::Event::ModifiersChanged(self.modifiers));
            }
            InputEvent::FocusChanged(focused) => {
                if !focused {
                    self.release_focus(out);
                }
                out.push(egui::Event::WindowFocused(*focused));
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
                let button = egui::PointerButton::from(*button);
                self.pressed_buttons.retain(|pressed| *pressed != button);
                if *state == ButtonState::Pressed {
                    self.pressed_buttons.push(button);
                }
                out.push(egui::Event::PointerButton {
                    pos: position.to_egui(extent, native_pixels_per_point),
                    button,
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
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::from(*modifiers),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirk_input::{
        ImeEvent, LogicalKey, Modifiers, NormalizedDelta, NormalizedPosition, PointerButton,
        ScrollUnit,
    };
    use egui::{Pos2, Vec2};

    fn window_id(raw: usize) -> WindowId {
        WindowId::from_raw(raw)
    }

    #[test]
    fn pointer_positions_are_scaled_from_normalized_to_points() {
        let events = EguiInputState::default().translate_events(
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
    fn text_comes_from_text_events_and_keeps_its_case() {
        let modifiers = Modifiers {
            shift: true,
            ..Modifiers::default()
        };
        let events = EguiInputState::default().translate_events(
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
    fn ime_composition_reaches_egui() {
        let events = EguiInputState::default().translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[
                ImeEvent::Enabled,
                ImeEvent::Preedit("あ".to_owned()),
                ImeEvent::Commit("あ".to_owned()),
                ImeEvent::Disabled,
            ]
            .map(|event| WindowInputEvent {
                window: window_id(1),
                event: InputEvent::Ime(event),
            }),
        );

        assert_eq!(
            events,
            vec![
                egui::Event::Ime(egui::ImeEvent::Preedit {
                    text: "あ".to_owned(),
                    active_range_chars: None,
                }),
                egui::Event::Ime(egui::ImeEvent::Commit("あ".to_owned())),
                egui::Event::Ime(egui::ImeEvent::Preedit {
                    text: String::new(),
                    active_range_chars: None,
                }),
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
        let events = EguiInputState::default().translate_events(
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
        let events = EguiInputState::default().translate_events(
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
                phase: egui::TouchPhase::Move,
                delta: Vec2::new(1.0, 1.0),
                modifiers: modifiers.into(),
            }]
        );
    }

    #[test]
    fn modifiers_use_last_change_for_window_and_persist_between_frames() {
        let mut input = EguiInputState::default();
        let ctrl = Modifiers {
            ctrl: true,
            ..Modifiers::default()
        };
        let change = |raw, modifiers| WindowInputEvent {
            window: window_id(raw),
            event: InputEvent::ModifiersChanged(modifiers),
        };
        let events = input.translate_events(
            window_id(1),
            glam::UVec2::ONE,
            1.0,
            &[
                change(1, Modifiers::default()),
                change(1, ctrl),
                change(2, Modifiers::default()),
                WindowInputEvent {
                    window: window_id(2),
                    event: InputEvent::FocusChanged(false),
                },
            ],
        );
        assert_eq!(input.modifiers, ctrl.into());
        let ctx = egui::Context::default();
        ctx.run_ui(
            egui::RawInput {
                events,
                ..Default::default()
            },
            |_| {},
        )
        .drop_without_applying_deltas();
        assert_eq!(ctx.input(|input| input.modifiers), ctrl.into());
        input.run_frame(&ctx, Vec::new(), true);
        assert_eq!(input.modifiers, ctrl.into());
        assert_eq!(ctx.input(|input| input.modifiers), ctrl.into());
    }

    impl EguiInputState {
        fn run_frame(&mut self, ctx: &egui::Context, events: Vec<InputEvent>, focused: bool) {
            let events = events
                .into_iter()
                .map(|event| WindowInputEvent {
                    window: window_id(1),
                    event,
                })
                .collect::<Vec<_>>();
            let events = self.translate_events(window_id(1), glam::uvec2(100, 100), 1.0, &events);
            ctx.run_ui(
                egui::RawInput {
                    events,
                    focused,
                    ..Default::default()
                },
                |_| {},
            )
            .drop_without_applying_deltas();
        }
    }

    #[test]
    fn focus_loss_cancels_keys_buttons_and_modifiers_without_clicking() {
        // Exercise presses from a previous frame and from the focus-loss batch.
        for separate_frame in [false, true] {
            let ctx = egui::Context::default();
            let mut input = EguiInputState::default();
            let modifiers = Modifiers {
                ctrl: true,
                ..Modifiers::default()
            };
            let mut events = vec![
                InputEvent::ModifiersChanged(modifiers),
                InputEvent::Key {
                    key: LogicalKey::character("w"),
                    state: ButtonState::Pressed,
                    repeat: false,
                    modifiers,
                },
            ];
            events.extend(
                [
                    PointerButton::Primary,
                    PointerButton::Secondary,
                    PointerButton::Middle,
                    PointerButton::Back,
                    PointerButton::Forward,
                ]
                .map(|button| InputEvent::PointerButton {
                    button,
                    state: ButtonState::Pressed,
                    position: NormalizedPosition::new(glam::vec2(0.5, 0.5)),
                    modifiers,
                }),
            );
            if separate_frame {
                input.run_frame(&ctx, std::mem::take(&mut events), true);
                ctx.input(|i| {
                    assert!(i.key_down(egui::Key::W));
                    assert!(i.pointer.any_down());
                });
            }
            events.push(InputEvent::FocusChanged(false));
            input.run_frame(&ctx, events, false);
            ctx.input(|i| {
                assert!(!i.key_down(egui::Key::W));
                assert!(!i.pointer.any_down());
                assert!(!i.pointer.any_click());
                assert!(i.pointer.latest_pos().is_none());
                assert!(i.pointer.delta().is_finite());
                assert_eq!(i.modifiers, egui::Modifiers::default());
            });
            input.run_frame(&ctx, vec![InputEvent::FocusChanged(true)], true);
            ctx.input(|i| {
                assert!(!i.key_down(egui::Key::W));
                assert!(!i.pointer.any_down());
            });
        }
    }

    #[test]
    fn focus_regained_in_same_batch_preserves_only_new_presses() {
        let ctx = egui::Context::default();
        let mut input = EguiInputState::default();
        let key = |text| InputEvent::Key {
            key: LogicalKey::character(text),
            state: ButtonState::Pressed,
            repeat: false,
            modifiers: Modifiers::default(),
        };
        input.run_frame(
            &ctx,
            vec![
                key("w"),
                InputEvent::PointerButton {
                    button: PointerButton::Secondary,
                    state: ButtonState::Pressed,
                    position: NormalizedPosition::new(glam::vec2(0.5, 0.5)),
                    modifiers: Modifiers::default(),
                },
                InputEvent::FocusChanged(false),
                InputEvent::FocusChanged(true),
                key("a"),
                InputEvent::PointerButton {
                    button: PointerButton::Primary,
                    state: ButtonState::Pressed,
                    position: NormalizedPosition::new(glam::vec2(0.5, 0.5)),
                    modifiers: Modifiers::default(),
                },
            ],
            true,
        );
        ctx.input(|i| {
            assert!(!i.key_down(egui::Key::W));
            assert!(i.key_down(egui::Key::A));
            assert!(i.pointer.primary_down());
            assert!(!i.pointer.secondary_down());
            assert!(i.pointer.delta().is_finite());
        });
    }

    #[test]
    fn pointer_exit_does_not_cancel_egui_drag_but_focus_loss_does() {
        let ctx = egui::Context::default();
        let mut input = EguiInputState::default();
        input.run_frame(
            &ctx,
            vec![InputEvent::PointerButton {
                button: PointerButton::Primary,
                state: ButtonState::Pressed,
                position: NormalizedPosition::new(glam::vec2(0.5, 0.5)),
                modifiers: Modifiers::default(),
            }],
            true,
        );
        input.run_frame(&ctx, vec![InputEvent::PointerLeft], true);
        assert!(ctx.input(|i| i.pointer.primary_down()));
        input.run_frame(&ctx, vec![InputEvent::FocusChanged(false)], false);
        assert!(!ctx.input(|i| i.pointer.any_down()));
    }
}
