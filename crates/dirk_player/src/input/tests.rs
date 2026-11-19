use super::*;
use dirk_input::{
    ButtonState, LogicalKey, Modifiers, NormalizedDelta, NormalizedPosition, PointerButton,
};

#[test]
fn movement_requires_right_mouse_gate() {
    let mut input = InputContext::new();
    input.handle_event(&InputEvent::Key {
        key: LogicalKey::character("w"),
        state: ButtonState::Pressed,
        repeat: false,
        modifiers: Modifiers::default(),
    });

    assert_eq!(input.movement_input(), glam::Vec3::ZERO);
}

#[test]
fn movement_axes_read_default_bindings_when_gate_is_held() {
    let mut input = InputContext::new();
    press(
        &mut input,
        &InputEvent::PointerButton {
            button: PointerButton::Secondary,
            state: ButtonState::Pressed,
            position: NormalizedPosition::new(glam::Vec2::ZERO),
            modifiers: Modifiers::default(),
        },
    );
    press_key(&mut input, "w");
    press_key(&mut input, "d");
    input.handle_event(&InputEvent::Key {
        key: LogicalKey::Named(dirk_input::NamedKey::Space),
        state: ButtonState::Pressed,
        repeat: false,
        modifiers: Modifiers::default(),
    });

    assert_eq!(input.movement_input(), glam::vec3(1.0, 1.0, 1.0));
}

#[test]
fn look_uses_normalized_pointer_delta_while_gate_is_held() {
    let mut input = InputContext::new();
    press(
        &mut input,
        &InputEvent::PointerButton {
            button: PointerButton::Secondary,
            state: ButtonState::Pressed,
            position: NormalizedPosition::new(glam::Vec2::ZERO),
            modifiers: Modifiers::default(),
        },
    );
    input.handle_event(&InputEvent::PointerMoved {
        position: NormalizedPosition::new(glam::Vec2::ONE),
        delta: NormalizedDelta(glam::vec2(2.0, -1.0)),
    });

    assert_eq!(input.look_input(), glam::dvec2(2.0, -1.0));
}

#[test]
fn focus_loss_and_pointer_exit_clear_movement_and_look() {
    for release in [InputEvent::PointerLeft, InputEvent::FocusChanged(false)] {
        let mut input = InputContext::new();
        press(
            &mut input,
            &InputEvent::PointerButton {
                button: PointerButton::Secondary,
                state: ButtonState::Pressed,
                position: NormalizedPosition::new(glam::Vec2::ZERO),
                modifiers: Modifiers::default(),
            },
        );
        press_key(&mut input, "w");

        input.handle_event(&InputEvent::PointerMoved {
            position: NormalizedPosition::new(glam::Vec2::ONE),
            delta: NormalizedDelta(glam::vec2(2.0, -1.0)),
        });
        assert_ne!(input.look_input(), glam::DVec2::ZERO);
        input.handle_event(&release);

        assert_eq!(input.movement_input(), glam::Vec3::ZERO);
        assert_eq!(input.look_input(), glam::DVec2::ZERO);
    }
}

fn press_key(input: &mut InputContext, key: &str) {
    press(
        input,
        &InputEvent::Key {
            key: LogicalKey::character(key),
            state: ButtonState::Pressed,
            repeat: false,
            modifiers: Modifiers::default(),
        },
    );
}

fn press(input: &mut InputContext, event: &InputEvent) {
    input.handle_event(event);
}
