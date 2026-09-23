#![doc = include_str!("../README.md")]

#[cfg(feature = "egui")]
pub mod egui;

use std::collections::HashSet;

/// Whether a button-like input is pressed or released.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ButtonState {
    /// The input is pressed.
    Pressed,
    /// The input is released.
    Released,
}

impl From<bool> for ButtonState {
    fn from(value: bool) -> Self {
        if value {
            ButtonState::Pressed
        } else {
            ButtonState::Released
        }
    }
}

/// Keyboard input after layout mapping.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum LogicalKey {
    /// A text-producing key. Stored lower-case for binding comparison.
    Character(String),
    /// A named non-text key.
    Named(NamedKey),
}

impl LogicalKey {
    /// Creates a normalized character key.
    #[must_use]
    pub fn character(text: impl AsRef<str>) -> Self {
        Self::Character(text.as_ref().to_lowercase())
    }
    /// Returns the text stored by the logical key
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Character(text) if !text.is_empty() => Some(text),
            Self::Named(NamedKey::Space) => Some(" "),
            _ => None,
        }
    }
}

/// Named logical keyboard keys used by engine bindings.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum NamedKey {
    /// Escape.
    Escape,
    /// Tab.
    Tab,
    /// Enter/Return.
    Enter,
    /// Backspace.
    Backspace,
    /// Space.
    Space,
    /// Up arrow.
    ArrowUp,
    /// Down arrow.
    ArrowDown,
    /// Left arrow.
    ArrowLeft,
    /// Right arrow.
    ArrowRight,
    /// Home.
    Home,
    /// End.
    End,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Insert.
    Insert,
    /// Delete.
    Delete,
    /// Function key, numbered from 1.
    Function(u8),
}

/// Pointer or mouse button.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum PointerButton {
    /// Primary pointer button, usually left mouse.
    Primary,
    /// Secondary pointer button, usually right mouse.
    Secondary,
    /// Middle pointer button.
    Middle,
    /// Back/extra button.
    Back,
    /// Forward/extra button.
    Forward,
    /// Any other button code.
    Other(u16),
}

/// Unit used by scroll wheel or trackpad input.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ScrollUnit {
    /// Logical pixels/points.
    Point,
    /// Text lines.
    Line,
    /// Pages.
    Page,
}

/// Keyboard modifiers active during an input event.
#[derive(Debug, Default, Clone, Copy, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Modifiers {
    /// Alt/Option.
    pub alt: bool,
    /// Control.
    pub ctrl: bool,
    /// Shift.
    pub shift: bool,
    /// Super/Command/Windows.
    pub super_key: bool,
}

/// Viewport-local normalized pointer position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPosition(pub glam::Vec2);

impl NormalizedPosition {
    /// Creates a clamped normalized position.
    #[must_use]
    pub fn new(position: glam::Vec2) -> Self {
        Self(position.clamp(glam::Vec2::ZERO, glam::Vec2::ONE))
    }
}

/// Viewport-local normalized delta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedDelta(pub glam::Vec2);

/// Engine input event. Contains no platform window identity.
#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    /// A logical key changed state.
    Key {
        /// Logical key after layout mapping.
        key: LogicalKey,
        /// Button state.
        state: ButtonState,
        /// Whether this is an OS-generated repeat.
        repeat: bool,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Text committed by the platform input method, preserving case and layout.
    Text(String),
    /// Composition updates from the platform input method.
    Ime(ImeEvent),
    /// Keyboard modifiers changed without a key event.
    ModifiersChanged(Modifiers),
    /// The pointer moved.
    PointerMoved {
        /// Current clamped normalized position.
        position: NormalizedPosition,
        /// Normalized delta since the previous pointer position.
        delta: NormalizedDelta,
    },
    /// The pointer entered the input region.
    PointerEntered,
    /// The pointer left the input region.
    PointerLeft,
    /// A pointer button changed state.
    PointerButton {
        /// Pointer button.
        button: PointerButton,
        /// Button state.
        state: ButtonState,
        /// Current clamped normalized position.
        position: NormalizedPosition,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Scroll wheel or trackpad input, normalized by viewport/window size.
    Scroll {
        /// Normalized scroll delta.
        delta: NormalizedDelta,
        /// Unit used by the scroll delta.
        unit: ScrollUnit,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
}

/// Platform-independent input method event.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ImeEvent {
    /// Composition started.
    Enabled,
    /// Current composition candidate.
    Preedit(String),
    /// Final composition text.
    Commit(String),
    /// Composition ended.
    Disabled,
}

/// A raw input that can activate an [`InputAction`].
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum InputBinding {
    /// A logical keyboard key.
    Key(LogicalKey),
    /// A pointer button.
    PointerButton(PointerButton),
}

/// A named digital input that is active while any of its bindings are held.
#[derive(Debug, Clone)]
pub struct InputAction {
    /// Stable action name.
    pub name: String,
    /// Bindings that activate this action.
    pub bindings: Vec<InputBinding>,
}

impl InputAction {
    /// Creates an action with the supplied bindings.
    #[must_use]
    pub fn new(name: impl Into<String>, bindings: impl Into<Vec<InputBinding>>) -> Self {
        Self {
            name: name.into(),
            bindings: bindings.into(),
        }
    }
}

/// A named scalar axis composed from two digital actions.
#[derive(Debug, Clone)]
pub struct InputAxis {
    /// Stable axis name.
    pub name: String,
    /// Negative direction action.
    pub negative: InputAction,
    /// Positive direction action.
    pub positive: InputAction,
}

impl InputAxis {
    /// Creates a digital axis.
    #[must_use]
    pub fn digital(name: impl Into<String>, negative: InputAction, positive: InputAction) -> Self {
        Self {
            name: name.into(),
            negative,
            positive,
        }
    }
}

/// Default player movement action map.
/// TODO: this should be customizable
#[derive(Debug, Clone)]
pub struct InputMap {
    /// Gate action for movement/look.
    pub movement_gate: InputAction,
    /// Right/left movement axis.
    pub right: InputAxis,
    /// Up/down movement axis.
    pub up: InputAxis,
    /// Forward/back movement axis.
    pub forward: InputAxis,
}

impl InputMap {
    /// Creates the default player map.
    #[must_use]
    pub fn default_player() -> Self {
        Self::default()
    }

    /// Reads movement from the supplied input state.
    #[must_use]
    pub fn movement(&self, input: &InputState) -> glam::Vec3 {
        if !input.action_active(&self.movement_gate) {
            return glam::Vec3::ZERO;
        }

        glam::vec3(
            input.axis_value(&self.right),
            input.axis_value(&self.up),
            input.axis_value(&self.forward),
        )
    }
}

impl Default for InputMap {
    fn default() -> Self {
        Self {
            movement_gate: InputAction::new(
                "move_gate",
                [InputBinding::PointerButton(PointerButton::Secondary)],
            ),
            right: InputAxis::digital(
                "move_right",
                InputAction::new("move_left", [InputBinding::Key(LogicalKey::character("a"))]),
                InputAction::new(
                    "move_right",
                    [InputBinding::Key(LogicalKey::character("d"))],
                ),
            ),
            up: InputAxis::digital(
                "move_up",
                InputAction::new("move_down", [InputBinding::Key(LogicalKey::character("c"))]),
                InputAction::new(
                    "move_up",
                    [InputBinding::Key(LogicalKey::Named(NamedKey::Space))],
                ),
            ),
            forward: InputAxis::digital(
                "move_forward",
                InputAction::new(
                    "move_backward",
                    [InputBinding::Key(LogicalKey::character("s"))],
                ),
                InputAction::new(
                    "move_forward",
                    [InputBinding::Key(LogicalKey::character("w"))],
                ),
            ),
        }
    }
}

/// Current held input state.
#[derive(Debug, Default)]
pub struct InputState {
    held: HashSet<InputBinding>,
}

impl InputState {
    /// Applies an input event to held input state.
    pub fn handle_event(&mut self, event: &InputEvent) {
        match event {
            InputEvent::Key { key, state, .. } => {
                self.set(InputBinding::Key(key.clone()), *state);
            }
            InputEvent::PointerButton { button, state, .. } => {
                self.set(InputBinding::PointerButton(*button), *state);
            }
            InputEvent::PointerMoved { .. }
            | InputEvent::PointerEntered
            | InputEvent::Scroll { .. }
            | InputEvent::Text(_)
            | InputEvent::Ime(_)
            | InputEvent::ModifiersChanged(_) => {}
            // release held keys
            InputEvent::PointerLeft => {
                self.held.clear();
            }
        }
    }

    /// Returns whether a binding is held.
    #[must_use]
    pub fn is_held(&self, binding: &InputBinding) -> bool {
        self.held.contains(binding)
    }

    /// Returns whether an action is active.
    #[must_use]
    pub fn action_active(&self, action: &InputAction) -> bool {
        action.bindings.iter().any(|binding| self.is_held(binding))
    }

    /// Returns a digital axis value in `[-1, 1]`.
    #[must_use]
    pub fn axis_value(&self, axis: &InputAxis) -> f32 {
        let positive = f32::from(u8::from(self.action_active(&axis.positive)));
        let negative = f32::from(u8::from(self.action_active(&axis.negative)));
        positive - negative
    }

    fn set(&mut self, binding: InputBinding, state: ButtonState) {
        match state {
            ButtonState::Pressed => {
                self.held.insert(binding);
            }
            ButtonState::Released => {
                self.held.remove(&binding);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_keys_are_normalized_to_lowercase() {
        assert_eq!(
            LogicalKey::character("W"),
            LogicalKey::Character("w".to_owned())
        );
    }

    #[test]
    fn default_player_map_uses_wasd_space_and_c() {
        let map = InputMap::default_player();

        assert!(
            map.right
                .negative
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Character("a".to_owned())))
        );
        assert!(
            map.right
                .positive
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Character("d".to_owned())))
        );
        assert!(
            map.forward
                .negative
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Character("s".to_owned())))
        );
        assert!(
            map.forward
                .positive
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Character("w".to_owned())))
        );
        assert!(
            map.up
                .negative
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Character("c".to_owned())))
        );
        assert!(
            map.up
                .positive
                .bindings
                .contains(&InputBinding::Key(LogicalKey::Named(NamedKey::Space)))
        );
    }

    #[test]
    fn opposing_axes_cancel() {
        let map = InputMap::default_player();
        let mut input = InputState::default();
        input.handle_event(&InputEvent::PointerButton {
            button: PointerButton::Secondary,
            state: ButtonState::Pressed,
            position: NormalizedPosition::new(glam::Vec2::ZERO),
            modifiers: Modifiers::default(),
        });
        input.handle_event(&InputEvent::Key {
            key: LogicalKey::character("a"),
            state: ButtonState::Pressed,
            repeat: false,
            modifiers: Modifiers::default(),
        });
        input.handle_event(&InputEvent::Key {
            key: LogicalKey::character("d"),
            state: ButtonState::Pressed,
            repeat: false,
            modifiers: Modifiers::default(),
        });

        assert_eq!(map.movement(&input).x, 0.0);
    }

    #[test]
    fn pointer_left_clears_all_held_bindings() {
        let mut input = InputState::default();
        let pointer_binding = InputBinding::PointerButton(PointerButton::Secondary);
        let key_binding = InputBinding::Key(LogicalKey::character("w"));
        input.handle_event(&InputEvent::PointerButton {
            button: PointerButton::Secondary,
            state: ButtonState::Pressed,
            position: NormalizedPosition::new(glam::Vec2::ZERO),
            modifiers: Modifiers::default(),
        });
        input.handle_event(&InputEvent::Key {
            key: LogicalKey::character("w"),
            state: ButtonState::Pressed,
            repeat: false,
            modifiers: Modifiers::default(),
        });

        input.handle_event(&InputEvent::PointerLeft);

        assert!(!input.is_held(&pointer_binding));
        assert!(!input.is_held(&key_binding));
    }
}
