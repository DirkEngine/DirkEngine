use std::time::Instant;

use ash::vk;
use dirk_input::{ButtonState, InputEvent};
use dirk_platform::{Theme, WindowId, WindowInputEvent};
use egui::{ClippedPrimitive, Context, TextureId, TexturesDelta, ViewportId, ViewportInfo};
use egui_ash_renderer::{DynamicRendering, Options};

use crate::{
    MAX_FRAMES_IN_FLIGHT, Result,
    resources::{command_pool::CommandBuffer, device::RenderDevice},
};

pub struct EguiState {
    ctx: Context,
    renderer: egui_ash_renderer::Renderer,
    start_time: Instant,
    pending: Option<EguiPaintData>,
    textures_to_free: [Vec<TextureId>; MAX_FRAMES_IN_FLIGHT],
}

pub struct EguiFrameInput {
    pub window_id: WindowId,
    pub extent: vk::Extent2D,
    pub native_pixels_per_point: f32,
    pub focused: bool,
    pub theme: Option<Theme>,
    pub events: Vec<WindowInputEvent>,
}

impl EguiState {
    pub fn new(device: &RenderDevice) -> Result<Self> {
        let surface_format = device.properties.surface_format.format;
        let backend = device.rhi.backend();
        let renderer = egui_ash_renderer::Renderer::with_default_allocator(
            backend.instance(),
            backend.physical_device(),
            backend.device().clone(),
            DynamicRendering {
                color_attachment_format: surface_format,
                depth_attachment_format: None,
            },
            Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                srgb_framebuffer: is_srgb_format(surface_format),
                ..Options::default()
            },
        )?;

        Ok(Self {
            ctx: Context::default(),
            renderer,
            start_time: Instant::now(),
            pending: None,
            textures_to_free: std::array::from_fn(|_| Vec::new()),
        })
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

        self.ctx.begin_pass(raw_input);
        self.ctx.clone()
    }

    pub fn end_frame(&mut self) {
        let output = self.ctx.end_pass();
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        self.pending = Some(EguiPaintData {
            textures_delta: output.textures_delta,
            primitives,
            pixels_per_point: output.pixels_per_point,
        });
    }

    pub fn free_textures_for_frame(&mut self, frame: usize) -> Result<()> {
        let textures = std::mem::take(&mut self.textures_to_free[frame]);
        self.renderer.free_textures(&textures)?;
        Ok(())
    }

    pub fn add_user_texture(&mut self, set: vk::DescriptorSet) -> TextureId {
        self.renderer.add_user_texture(set)
    }

    pub fn remove_user_texture(&mut self, id: TextureId) {
        self.renderer.remove_user_texture(id);
    }

    pub fn render(
        &mut self,
        device: &RenderDevice,
        cmd: &CommandBuffer,
        extent: vk::Extent2D,
        frame: usize,
    ) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };

        self.renderer.set_textures(
            device.rhi.backend().queue(dirk_rhi::QueueType::Graphics),
            device.graphics_pool.raw(),
            pending.textures_delta.set.as_slice(),
        )?;

        self.renderer.cmd_draw(
            **cmd,
            extent,
            pending.pixels_per_point,
            pending.primitives.as_slice(),
        )?;

        self.textures_to_free[frame].extend(pending.textures_delta.free);
        Ok(())
    }
}

struct EguiPaintData {
    textures_delta: TexturesDelta,
    primitives: Vec<ClippedPrimitive>,
    pixels_per_point: f32,
}

fn is_srgb_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
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
            if *state == ButtonState::Pressed
                && !*repeat
                && !modifiers.command
                && !modifiers.ctrl
                && let Some(text) = key.text()
            {
                out.push(egui::Event::Text(text.to_owned()));
            }
        }
        InputEvent::PointerMoved { position, .. } => {
            out.push(egui::Event::PointerMoved(
                position.to_egui(extent, native_pixels_per_point),
            ));
        }
        InputEvent::PointerEntered => {}
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
        assert!(is_srgb_format(vk::Format::R8G8B8A8_SRGB));
        assert!(is_srgb_format(vk::Format::B8G8R8A8_SRGB));
        assert!(is_srgb_format(vk::Format::A8B8G8R8_SRGB_PACK32));
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
    fn printable_key_press_emits_key_and_text_events() {
        let modifiers = Modifiers::default();
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[WindowInputEvent {
                window: window_id(1),
                event: InputEvent::Key {
                    key: LogicalKey::character("a"),
                    state: ButtonState::Pressed,
                    repeat: false,
                    modifiers,
                },
            }],
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
                egui::Event::Text("a".to_owned()),
            ]
        );
    }

    #[test]
    fn space_key_press_emits_text_event() {
        let events = translate_events(
            window_id(1),
            glam::UVec2 { x: 100, y: 100 },
            1.0,
            &[WindowInputEvent {
                window: window_id(1),
                event: InputEvent::Key {
                    key: LogicalKey::Named(NamedKey::Space),
                    state: ButtonState::Pressed,
                    repeat: false,
                    modifiers: Modifiers::default(),
                },
            }],
        );

        assert!(events.contains(&egui::Event::Text(" ".to_owned())));
    }

    #[test]
    fn printable_key_text_events_are_suppressed_for_repeats_releases_and_command_modifiers() {
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

        assert!(
            events
                .iter()
                .all(|event| !matches!(event, egui::Event::Text(_)))
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
