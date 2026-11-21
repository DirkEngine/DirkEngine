use super::*;
use dirk_input::{
    ImeEvent, LogicalKey, Modifiers, NormalizedDelta, NormalizedPosition, PointerButton, ScrollUnit,
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
