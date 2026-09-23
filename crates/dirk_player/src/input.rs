//! Per-player input handling.

use dirk_input::{InputEvent, InputMap, InputState};

/// Per-player input state and bindings.
#[derive(Default)]
pub struct InputContext {
    snapshot: InputState,
    movement: InputMap,
    pointer_delta: glam::Vec2,
}

impl InputContext {
    /// Creates an input context with the default player bindings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn handle_event(&mut self, event: &InputEvent) {
        self.snapshot.handle_event(event);
        // only store pointer delta for look_input. everything else is handled
        // by the input state
        match event {
            InputEvent::PointerMoved { delta, .. } => {
                if self.snapshot.action_active(&self.movement.movement_gate) {
                    self.pointer_delta += delta.0;
                }
            }
            InputEvent::PointerLeft => {
                self.pointer_delta = glam::Vec2::ZERO;
            }
            InputEvent::Key { .. }
            | InputEvent::Text(_)
            | InputEvent::Ime(_)
            | InputEvent::ModifiersChanged(_)
            | InputEvent::PointerEntered
            | InputEvent::PointerButton { .. }
            | InputEvent::Scroll { .. } => {}
        }
    }

    #[must_use]
    pub(crate) fn movement_input(&self) -> glam::Vec3 {
        self.movement.movement(&self.snapshot)
    }

    #[must_use]
    pub(crate) fn look_input(&self) -> glam::DVec2 {
        self.pointer_delta.as_dvec2()
    }

    pub(crate) fn clear_frame_state(&mut self) {
        self.pointer_delta = glam::Vec2::ZERO;
    }
}

#[cfg(test)]
mod tests {
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
            InputEvent::PointerButton {
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
            InputEvent::PointerButton {
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
    fn pointer_left_releases_movement_gate() {
        let mut input = InputContext::new();
        press(
            &mut input,
            InputEvent::PointerButton {
                button: PointerButton::Secondary,
                state: ButtonState::Pressed,
                position: NormalizedPosition::new(glam::Vec2::ZERO),
                modifiers: Modifiers::default(),
            },
        );
        press_key(&mut input, "w");

        input.handle_event(&InputEvent::PointerLeft);

        assert_eq!(input.movement_input(), glam::Vec3::ZERO);
    }

    fn press_key(input: &mut InputContext, key: &str) {
        press(
            input,
            InputEvent::Key {
                key: LogicalKey::character(key),
                state: ButtonState::Pressed,
                repeat: false,
                modifiers: Modifiers::default(),
            },
        );
    }

    fn press(input: &mut InputContext, event: InputEvent) {
        input.handle_event(&event);
    }
}
