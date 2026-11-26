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
            InputEvent::PointerLeft | InputEvent::FocusChanged(false) => {
                self.pointer_delta = glam::Vec2::ZERO;
            }
            InputEvent::Key { .. }
            | InputEvent::PointerEntered
            | InputEvent::PointerButton { .. }
            | InputEvent::Scroll { .. }
            | InputEvent::FocusChanged(true)
            | InputEvent::Text(_)
            | InputEvent::Ime(_)
            | InputEvent::ModifiersChanged(_) => {}
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
mod tests;
