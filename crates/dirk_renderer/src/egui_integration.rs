use std::time::Instant;

use ash::vk;
use dirk_engine::editor::{EditorRenderContext, EditorServices};
use dirk_input::{ButtonState, InputEvent};
use dirk_platform::{Theme, WindowId, WindowInputEvent};
use dirk_universe::Universe;
use egui::{ClippedPrimitive, Context, TextureId, TexturesDelta, ViewportId, ViewportInfo};
use egui_ash_renderer::{DynamicRendering, Options, RenderMode, allocator::DefaultAllocator};

use crate::{
    MAX_FRAMES_IN_FLIGHT, Result,
    resources::{command_pool::CommandBuffer, device::RenderDevice, queues::QueueType},
};

pub struct EguiState {
    ctx: Context,
    renderer: egui_ash_renderer::Renderer<DefaultAllocator>,
    start_time: Instant,
    pending: Option<EguiPaintData>,
    textures_to_free: [Vec<TextureId>; MAX_FRAMES_IN_FLIGHT],
    input: EguiInputState,
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
        let renderer = egui_ash_renderer::Renderer::with_default_allocator(
            &device.instance,
            device.physical_device,
            device.device.clone(),
            RenderMode::DynamicRendering(DynamicRendering {
                color_attachment_format: surface_format,
                depth_attachment_format: None,
                stencil_attachment_format: None,
            }),
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
            input: EguiInputState::default(),
        })
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn run_frame(
        &mut self,
        input: &EguiFrameInput,
        editor: &EditorServices,
        context: &EditorRenderContext<'_>,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        let native_pixels_per_point = input.native_pixels_per_point.max(f32::EPSILON);
        let screen_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(
                input.extent.width as f32 / native_pixels_per_point,
                input.extent.height as f32 / native_pixels_per_point,
            ),
        );
        let events = self.input.translate_events(
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

        let output = editor.render_ui(&self.ctx, raw_input, context, universe)?;
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        self.pending = Some(EguiPaintData {
            textures_delta: output.textures_delta,
            primitives,
            pixels_per_point: output.pixels_per_point,
        });
        Ok(())
    }

    pub fn free_textures_for_frame(&mut self, frame: usize) -> Result<()> {
        let textures = std::mem::take(&mut self.textures_to_free[frame]);
        for texture in textures {
            self.renderer.free_texture(texture)?;
        }
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
        let Some(mut pending) = self.pending.take() else {
            return Ok(());
        };

        for (id, deltas) in pending.textures_delta.set.drain() {
            for delta in deltas {
                self.renderer.set_texture(
                    device.queues.raw(QueueType::Graphics),
                    device.graphics_pool.raw(),
                    id,
                    &delta,
                )?;
            }
        }

        self.textures_to_free[frame].extend(pending.textures_delta.free.drain());

        self.renderer.cmd_draw(
            **cmd,
            extent,
            pending.pixels_per_point,
            pending.primitives.as_slice(),
        )?;

        Ok(())
    }
}

struct EguiPaintData {
    textures_delta: TexturesDelta,
    primitives: Vec<ClippedPrimitive>,
    pixels_per_point: f32,
}

impl Drop for EguiPaintData {
    fn drop(&mut self) {
        // A skipped frame or rendering error can leave deltas unapplied.
        self.textures_delta.clear();
    }
}

fn is_srgb_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
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
    fn printable_key_press_emits_key_and_text_events() {
        let modifiers = Modifiers::default();
        let events = EguiInputState::default().translate_events(
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
        let events = EguiInputState::default().translate_events(
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
        let events = EguiInputState::default().translate_events(
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
