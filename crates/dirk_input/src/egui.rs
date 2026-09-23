//! egui input helpers.

use crate::{
    ButtonState, InputEvent, LogicalKey, Modifiers, NamedKey, NormalizedDelta, NormalizedPosition,
    PointerButton, ScrollUnit,
};

impl From<PointerButton> for egui::PointerButton {
    fn from(value: PointerButton) -> Self {
        match value {
            PointerButton::Primary => Self::Primary,
            PointerButton::Secondary => Self::Secondary,
            PointerButton::Middle => Self::Middle,
            PointerButton::Back | PointerButton::Other(_) => Self::Extra1,
            PointerButton::Forward => Self::Extra2,
        }
    }
}

impl From<::egui::MouseWheelUnit> for ScrollUnit {
    fn from(value: ::egui::MouseWheelUnit) -> Self {
        match value {
            ::egui::MouseWheelUnit::Point => Self::Point,
            ::egui::MouseWheelUnit::Line => Self::Line,
            ::egui::MouseWheelUnit::Page => Self::Page,
        }
    }
}

impl From<ScrollUnit> for ::egui::MouseWheelUnit {
    fn from(value: ScrollUnit) -> Self {
        match value {
            ScrollUnit::Point => Self::Point,
            ScrollUnit::Line => Self::Line,
            ScrollUnit::Page => Self::Page,
        }
    }
}

impl From<::egui::Modifiers> for Modifiers {
    fn from(value: ::egui::Modifiers) -> Self {
        Self {
            alt: value.alt,
            ctrl: value.ctrl || ((!cfg!(target_os = "macos")) && value.command),
            shift: value.shift,
            super_key: value.mac_cmd || (cfg!(target_os = "macos") && value.command),
        }
    }
}

impl From<Modifiers> for ::egui::Modifiers {
    fn from(value: Modifiers) -> Self {
        Self {
            alt: value.alt,
            ctrl: value.ctrl,
            shift: value.shift,
            mac_cmd: cfg!(target_os = "macos") && value.super_key,
            command: if cfg!(target_os = "macos") {
                value.super_key
            } else {
                value.ctrl
            },
        }
    }
}

impl NormalizedPosition {
    /// Creates a [`NormalizedPosition`] from a position & egui rect
    #[must_use]
    pub fn from_egui(rect: ::egui::Rect, pos: ::egui::Pos2) -> Self {
        let local = pos - rect.min;
        Self::new(glam::vec2(
            normalize_component(local.x, rect.width()),
            normalize_component(local.y, rect.height()),
        ))
    }
    /// Turns the [`NormalizedPosition`] in to an [`egui::Pos2`] from
    /// an extent & a scale.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_egui(self, extent: glam::UVec2, native_pixels_per_point: f32) -> egui::Pos2 {
        let scale = native_pixels_per_point.max(f32::EPSILON);
        egui::Pos2::new(
            self.0.x * extent.x as f32 / scale,
            self.0.y * extent.y as f32 / scale,
        )
    }
}

impl NormalizedDelta {
    /// Converts egui delta & size to a [`NormalizedDelta`].
    #[must_use]
    pub fn from_egui(delta: ::egui::Vec2, size: ::egui::Vec2) -> Self {
        Self(glam::vec2(
            normalize_component(delta.x, size.x),
            normalize_component(delta.y, size.y),
        ))
    }
    /// Turns the [`NormalizedDelta`] in to an [`egui::Pos2`] from
    /// an extent & a scale.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_egui(self, extent: glam::UVec2, native_pixels_per_point: f32) -> egui::Vec2 {
        let scale = native_pixels_per_point.max(f32::EPSILON);
        egui::Vec2::new(
            self.0.x * extent.x as f32 / scale,
            self.0.y * extent.y as f32 / scale,
        )
    }
}

impl LogicalKey {
    /// Converts to an [`egui::Key`].
    #[must_use]
    pub fn to_egui(&self) -> Option<egui::Key> {
        match self {
            Self::Character(text) => text.chars().next().and_then(char_to_egui_key),
            Self::Named(named) => named.to_egui(),
        }
    }
}

impl NamedKey {
    /// Converts to an [`egui::Key`].
    #[must_use]
    pub fn to_egui(&self) -> Option<egui::Key> {
        Some(match self {
            Self::ArrowDown => egui::Key::ArrowDown,
            Self::ArrowLeft => egui::Key::ArrowLeft,
            Self::ArrowRight => egui::Key::ArrowRight,
            Self::ArrowUp => egui::Key::ArrowUp,
            Self::Escape => egui::Key::Escape,
            Self::Tab => egui::Key::Tab,
            Self::Backspace => egui::Key::Backspace,
            Self::Enter => egui::Key::Enter,
            Self::Space => egui::Key::Space,
            Self::Insert => egui::Key::Insert,
            Self::Delete => egui::Key::Delete,
            Self::Home => egui::Key::Home,
            Self::End => egui::Key::End,
            Self::PageUp => egui::Key::PageUp,
            Self::PageDown => egui::Key::PageDown,
            Self::Function(1) => egui::Key::F1,
            Self::Function(2) => egui::Key::F2,
            Self::Function(3) => egui::Key::F3,
            Self::Function(4) => egui::Key::F4,
            Self::Function(5) => egui::Key::F5,
            Self::Function(6) => egui::Key::F6,
            Self::Function(7) => egui::Key::F7,
            Self::Function(8) => egui::Key::F8,
            Self::Function(9) => egui::Key::F9,
            Self::Function(10) => egui::Key::F10,
            Self::Function(11) => egui::Key::F11,
            Self::Function(12) => egui::Key::F12,
            Self::Function(13) => egui::Key::F13,
            Self::Function(14) => egui::Key::F14,
            Self::Function(15) => egui::Key::F15,
            Self::Function(16) => egui::Key::F16,
            Self::Function(17) => egui::Key::F17,
            Self::Function(18) => egui::Key::F18,
            Self::Function(19) => egui::Key::F19,
            Self::Function(_) => return None,
        })
    }
}

/// Converts a [`char`] to an [`egui::Key`].
#[must_use]
pub fn char_to_egui_key(character: char) -> Option<egui::Key> {
    Some(match character.to_ascii_lowercase() {
        'a' => egui::Key::A,
        'b' => egui::Key::B,
        'c' => egui::Key::C,
        'd' => egui::Key::D,
        'e' => egui::Key::E,
        'f' => egui::Key::F,
        'g' => egui::Key::G,
        'h' => egui::Key::H,
        'i' => egui::Key::I,
        'j' => egui::Key::J,
        'k' => egui::Key::K,
        'l' => egui::Key::L,
        'm' => egui::Key::M,
        'n' => egui::Key::N,
        'o' => egui::Key::O,
        'p' => egui::Key::P,
        'q' => egui::Key::Q,
        'r' => egui::Key::R,
        's' => egui::Key::S,
        't' => egui::Key::T,
        'u' => egui::Key::U,
        'v' => egui::Key::V,
        'w' => egui::Key::W,
        'x' => egui::Key::X,
        'y' => egui::Key::Y,
        'z' => egui::Key::Z,
        '0' => egui::Key::Num0,
        '1' => egui::Key::Num1,
        '2' => egui::Key::Num2,
        '3' => egui::Key::Num3,
        '4' => egui::Key::Num4,
        '5' => egui::Key::Num5,
        '6' => egui::Key::Num6,
        '7' => egui::Key::Num7,
        '8' => egui::Key::Num8,
        '9' => egui::Key::Num9,
        _ => return None,
    })
}

/// Translates input owned by an egui viewport response into engine input events.
#[must_use]
pub fn input_events_from_egui_response(
    ui: &egui::Ui,
    response: &egui::Response,
    previous_pointer: Option<egui::Pos2>,
) -> Vec<InputEvent> {
    // if the pointer is currently routing input
    let pointer_routes = response.hovered()
        || response.dragged()
        || response.is_pointer_button_down_on()
        || previous_pointer.is_some();
    // if the keyboard is currently routing input
    let keyboard_routes = response.has_focus() || previous_pointer.is_some();
    let rect = response.rect;
    let size = rect.size();
    let mut out = Vec::new();
    let mut request_focus = response.clicked();

    ui.input(|input| {
        for event in &input.events {
            match event {
                egui::Event::PointerMoved(pos) if pointer_routes => {
                    out.push(InputEvent::PointerMoved {
                        position: NormalizedPosition::from_egui(rect, *pos),
                        delta: NormalizedDelta(normalized_delta(size, previous_pointer, *pos)),
                    });
                }
                egui::Event::PointerButton {
                    pos,
                    button,
                    modifiers,
                    pressed,
                    ..
                } if pointer_routes || rect.contains(*pos) => {
                    if *pressed {
                        request_focus = true;
                    }
                    out.push(InputEvent::PointerButton {
                        button: pointer_button(*button),
                        state: ButtonState::from(*pressed),
                        position: NormalizedPosition::from_egui(rect, *pos),
                        modifiers: Modifiers::from(*modifiers),
                    });
                }
                egui::Event::MouseWheel {
                    unit,
                    delta,
                    modifiers,
                    ..
                } if pointer_routes => {
                    out.push(InputEvent::Scroll {
                        delta: NormalizedDelta::from_egui(*delta, size),
                        unit: ScrollUnit::from(*unit),
                        modifiers: Modifiers::from(*modifiers),
                    });
                }
                egui::Event::Key {
                    key,
                    pressed,
                    repeat,
                    modifiers,
                    ..
                } if keyboard_routes => {
                    if let Some(key) = logical_key(*key) {
                        out.push(InputEvent::Key {
                            key,
                            state: ButtonState::from(*pressed),
                            repeat: *repeat,
                            modifiers: Modifiers::from(*modifiers),
                        });
                    }
                }
                egui::Event::PointerGone if pointer_routes => {
                    out.push(InputEvent::PointerLeft);
                }
                egui::Event::WindowFocused(false) if keyboard_routes => {
                    out.push(InputEvent::PointerLeft);
                }
                _ => {}
            }
        }
    });

    if request_focus {
        response.request_focus();
    }

    out
}

fn normalized_delta(
    size: egui::Vec2,
    previous: Option<egui::Pos2>,
    current: egui::Pos2,
) -> glam::Vec2 {
    previous.map_or(glam::Vec2::ZERO, |previous| {
        let delta = current - previous;
        glam::vec2(
            normalize_component(delta.x, size.x),
            normalize_component(delta.y, size.y),
        )
    })
}

fn normalize_component(value: f32, extent: f32) -> f32 {
    value / extent.max(f32::EPSILON)
}

fn pointer_button(button: egui::PointerButton) -> PointerButton {
    match button {
        egui::PointerButton::Primary => PointerButton::Primary,
        egui::PointerButton::Secondary => PointerButton::Secondary,
        egui::PointerButton::Middle => PointerButton::Middle,
        egui::PointerButton::Extra1 => PointerButton::Back,
        egui::PointerButton::Extra2 => PointerButton::Forward,
    }
}

fn logical_key(key: egui::Key) -> Option<LogicalKey> {
    Some(match key {
        egui::Key::ArrowDown => LogicalKey::Named(NamedKey::ArrowDown),
        egui::Key::ArrowLeft => LogicalKey::Named(NamedKey::ArrowLeft),
        egui::Key::ArrowRight => LogicalKey::Named(NamedKey::ArrowRight),
        egui::Key::ArrowUp => LogicalKey::Named(NamedKey::ArrowUp),
        egui::Key::Escape => LogicalKey::Named(NamedKey::Escape),
        egui::Key::Tab => LogicalKey::Named(NamedKey::Tab),
        egui::Key::Backspace => LogicalKey::Named(NamedKey::Backspace),
        egui::Key::Enter => LogicalKey::Named(NamedKey::Enter),
        egui::Key::Space => LogicalKey::Named(NamedKey::Space),
        egui::Key::Insert => LogicalKey::Named(NamedKey::Insert),
        egui::Key::Delete => LogicalKey::Named(NamedKey::Delete),
        egui::Key::Home => LogicalKey::Named(NamedKey::Home),
        egui::Key::End => LogicalKey::Named(NamedKey::End),
        egui::Key::PageUp => LogicalKey::Named(NamedKey::PageUp),
        egui::Key::PageDown => LogicalKey::Named(NamedKey::PageDown),
        egui::Key::A => LogicalKey::character("a"),
        egui::Key::B => LogicalKey::character("b"),
        egui::Key::C => LogicalKey::character("c"),
        egui::Key::D => LogicalKey::character("d"),
        egui::Key::E => LogicalKey::character("e"),
        egui::Key::F => LogicalKey::character("f"),
        egui::Key::G => LogicalKey::character("g"),
        egui::Key::H => LogicalKey::character("h"),
        egui::Key::I => LogicalKey::character("i"),
        egui::Key::J => LogicalKey::character("j"),
        egui::Key::K => LogicalKey::character("k"),
        egui::Key::L => LogicalKey::character("l"),
        egui::Key::M => LogicalKey::character("m"),
        egui::Key::N => LogicalKey::character("n"),
        egui::Key::O => LogicalKey::character("o"),
        egui::Key::P => LogicalKey::character("p"),
        egui::Key::Q => LogicalKey::character("q"),
        egui::Key::R => LogicalKey::character("r"),
        egui::Key::S => LogicalKey::character("s"),
        egui::Key::T => LogicalKey::character("t"),
        egui::Key::U => LogicalKey::character("u"),
        egui::Key::V => LogicalKey::character("v"),
        egui::Key::W => LogicalKey::character("w"),
        egui::Key::X => LogicalKey::character("x"),
        egui::Key::Y => LogicalKey::character("y"),
        egui::Key::Z => LogicalKey::character("z"),
        egui::Key::Num0 => LogicalKey::character("0"),
        egui::Key::Num1 => LogicalKey::character("1"),
        egui::Key::Num2 => LogicalKey::character("2"),
        egui::Key::Num3 => LogicalKey::character("3"),
        egui::Key::Num4 => LogicalKey::character("4"),
        egui::Key::Num5 => LogicalKey::character("5"),
        egui::Key::Num6 => LogicalKey::character("6"),
        egui::Key::Num7 => LogicalKey::character("7"),
        egui::Key::Num8 => LogicalKey::character("8"),
        egui::Key::Num9 => LogicalKey::character("9"),
        egui::Key::F1 => LogicalKey::Named(NamedKey::Function(1)),
        egui::Key::F2 => LogicalKey::Named(NamedKey::Function(2)),
        egui::Key::F3 => LogicalKey::Named(NamedKey::Function(3)),
        egui::Key::F4 => LogicalKey::Named(NamedKey::Function(4)),
        egui::Key::F5 => LogicalKey::Named(NamedKey::Function(5)),
        egui::Key::F6 => LogicalKey::Named(NamedKey::Function(6)),
        egui::Key::F7 => LogicalKey::Named(NamedKey::Function(7)),
        egui::Key::F8 => LogicalKey::Named(NamedKey::Function(8)),
        egui::Key::F9 => LogicalKey::Named(NamedKey::Function(9)),
        egui::Key::F10 => LogicalKey::Named(NamedKey::Function(10)),
        egui::Key::F11 => LogicalKey::Named(NamedKey::Function(11)),
        egui::Key::F12 => LogicalKey::Named(NamedKey::Function(12)),
        _ => return None, // TODO: other keys
    })
}
