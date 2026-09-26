use std::collections::HashMap;

use dirk_input::{
    ButtonState, ImeEvent, InputEvent, LogicalKey, Modifiers, NamedKey, NormalizedDelta,
    NormalizedPosition, PointerButton, ScrollUnit,
};
use tracing::{debug, trace};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, Ime, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{Key, ModifiersState, NamedKey as WinitNamedKey},
    window::{WindowAttributes, WindowId},
};

use crate::{
    PlatformWindows, Window, WindowInputEvent,
    event::{PlatformEvent, WindowEvent as PlatformWindowEvent},
};

pub struct PlatformHandler {
    can_create_surfaces: bool,
    #[cfg(target_os = "macos")]
    shutdown_requested: bool,
    windows: PlatformWindows,

    /// Current keyboard modifier state, updated on every `ModifiersChanged` event.
    /// TODO: should only be tracked by input manager
    modifiers: ModifiersState,
    /// The positions of the pointer on each window in pixels.
    pointer_positions: HashMap<WindowId, glam::DVec2>,

    /// Dispatch [`PlatformEvent`]
    platform_dispatcher: dirk_events::Dispatcher<PlatformEvent>,
    /// Dispatch [`PlatformWindowEvent`]
    window_dispatcher: dirk_events::Dispatcher<PlatformWindowEvent>,
    /// Dispatch [`WindowInputEvent`]
    input_dispatch: dirk_events::Dispatcher<WindowInputEvent>,
}

impl PlatformHandler {
    pub fn new(events: &dirk_events::EventManager, windows: PlatformWindows) -> Self {
        Self {
            can_create_surfaces: false,
            #[cfg(target_os = "macos")]
            shutdown_requested: false,
            windows,
            modifiers: ModifiersState::default(),
            pointer_positions: HashMap::new(),
            platform_dispatcher: events.register(),
            window_dispatcher: events.register(),
            input_dispatch: events.register(),
        }
    }
    fn create_window(&mut self, event_loop: &dyn ActiveEventLoop) -> anyhow::Result<WindowId> {
        let window_attributes = WindowAttributes::default()
            .with_title("DirkEngine")
            .with_transparent(true);

        let window = event_loop.create_window(window_attributes)?;

        let window = Window::new(window);
        let window_id = window.id();
        self.windows.insert(window);
        self.platform_dispatcher
            .dispatch(PlatformEvent::WindowCreated { id: window_id });
        Ok(window_id)
    }
    pub fn is_initialized(&self) -> bool {
        self.can_create_surfaces
    }
    pub fn shutdown(&mut self) {
        let count = self.windows.clear();
        debug!("Closed {count} window(s) during platform shutdown");
    }

    #[cfg(target_os = "macos")]
    pub fn request_shutdown(&mut self) {
        self.shutdown_requested = true;
    }

    fn dispatch_platform_event(&mut self, id: WindowId, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CloseRequested => {
                debug!("Close requested for Window={id:?}");
                self.windows.remove(id);
                self.pointer_positions.remove(&id);
                self.platform_dispatcher
                    .dispatch(PlatformEvent::WindowCloseRequested { id });
            }
            WindowEvent::Destroyed => {
                debug!("Window {id:?} destroyed");
                self.pointer_positions.remove(&id);
                self.platform_dispatcher
                    .dispatch(PlatformEvent::WindowDestroyed { id });
            }
            WindowEvent::SurfaceResized(size) => {
                self.window_dispatcher
                    .dispatch(PlatformWindowEvent::Resized {
                        id,
                        width: size.width,
                        height: size.height,
                    });
            }
            WindowEvent::ThemeChanged(theme) => {
                self.window_dispatcher
                    .dispatch(PlatformWindowEvent::ThemeChanged { id, theme: *theme });
            }
            WindowEvent::Focused(focused) => {
                self.window_dispatcher
                    .dispatch(PlatformWindowEvent::FocusChanged {
                        id,
                        focused: *focused,
                    });
            }
            WindowEvent::Occluded(occluded) => {
                self.window_dispatcher
                    .dispatch(PlatformWindowEvent::Occluded {
                        id,
                        occluded: *occluded,
                    });
            }
            _ => return false,
        }

        true
    }

    fn dispatch_input_event(&mut self, id: WindowId, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
                trace!("Modifiers changed to {:?}", self.modifiers);
                self.dispatch_input(
                    id,
                    InputEvent::ModifiersChanged(modifiers_from_winit(self.modifiers)),
                );
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } => self.dispatch_keyboard_input(id, event),
            WindowEvent::PointerMoved { position, .. } => self.dispatch_pointer_moved(id, position),
            WindowEvent::PointerEntered { position, .. } => {
                self.dispatch_pointer_entered(id, position);
            }
            WindowEvent::PointerLeft { .. } => self.dispatch_pointer_left(id),
            WindowEvent::PointerButton {
                button,
                state,
                position,
                ..
            } => self.dispatch_pointer_button(id, button, *state, position),
            WindowEvent::MouseWheel { delta, .. } => self.dispatch_mouse_wheel(id, delta),
            WindowEvent::Ime(event) => {
                let event = match event {
                    Ime::Enabled => Some(ImeEvent::Enabled),
                    Ime::Preedit(text, _) => Some(ImeEvent::Preedit(text.clone())),
                    Ime::Commit(text) => Some(ImeEvent::Commit(text.clone())),
                    Ime::Disabled => Some(ImeEvent::Disabled),
                    Ime::DeleteSurrounding { .. } => None,
                };
                if let Some(event) = event {
                    self.dispatch_input(id, InputEvent::Ime(event));
                }
            }
            _ => return false,
        }

        true
    }

    fn dispatch_pointer_moved(
        &mut self,
        id: WindowId,
        position: &winit::dpi::PhysicalPosition<f64>,
    ) {
        trace!("Pointer moved to {position:?}");
        let raw_position = glam::dvec2(position.x, position.y);
        let delta = self
            .pointer_positions
            .insert(id, raw_position)
            .map_or(glam::DVec2::ZERO, |previous| raw_position - previous);
        let event = InputEvent::PointerMoved {
            position: self.normalized_position(id, raw_position),
            delta: self.normalized_delta(id, delta),
        };
        self.dispatch_input(id, event);
    }

    fn dispatch_pointer_entered(
        &mut self,
        id: WindowId,
        position: &winit::dpi::PhysicalPosition<f64>,
    ) {
        trace!("Pointer entered Window={id:?}");
        self.pointer_positions
            .insert(id, glam::dvec2(position.x, position.y));
        self.dispatch_input(id, InputEvent::PointerEntered);
    }

    fn dispatch_pointer_left(&mut self, id: WindowId) {
        trace!("Pointer left Window={id:?}");
        self.pointer_positions.remove(&id);
        self.dispatch_input(id, InputEvent::PointerLeft);
    }

    fn dispatch_pointer_button(
        &mut self,
        id: WindowId,
        button: &winit::event::ButtonSource,
        state: ElementState,
        position: &winit::dpi::PhysicalPosition<f64>,
    ) {
        trace!("Pointer button {button:?} {state:?} at {position:?}");
        let Some(button) = pointer_button(button) else {
            return;
        };
        let position = glam::dvec2(position.x, position.y);
        self.pointer_positions.insert(id, position);
        self.dispatch_input(
            id,
            InputEvent::PointerButton {
                button,
                state: match state {
                    ElementState::Pressed => ButtonState::Pressed,
                    ElementState::Released => ButtonState::Released,
                },
                position: self.normalized_position(id, position),
                modifiers: modifiers_from_winit(self.modifiers),
            },
        );
    }

    fn dispatch_mouse_wheel(&self, id: WindowId, delta: &MouseScrollDelta) {
        trace!("Mouse wheel {delta:?}");
        let (delta, unit) = match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                (glam::dvec2(f64::from(*x), f64::from(*y)), ScrollUnit::Line)
            }
            MouseScrollDelta::PixelDelta(px) => (glam::dvec2(px.x, px.y), ScrollUnit::Point),
        };
        self.dispatch_input(
            id,
            InputEvent::Scroll {
                delta: self.normalized_delta(id, delta),
                unit,
                modifiers: modifiers_from_winit(self.modifiers),
            },
        );
    }

    fn dispatch_keyboard_input(&self, id: WindowId, event: &winit::event::KeyEvent) {
        let state = match event.state {
            ElementState::Pressed => ButtonState::Pressed,
            ElementState::Released => ButtonState::Released,
        };
        trace!(
            "Key {:?} {state:?} (repeat={})",
            event.logical_key, event.repeat
        );
        if let Some(key) = logical_key_from_winit(&event.logical_key) {
            self.dispatch_input(
                id,
                InputEvent::Key {
                    key,
                    state,
                    repeat: event.repeat,
                    modifiers: modifiers_from_winit(self.modifiers),
                },
            );
        }
        if state == ButtonState::Pressed
            && (!self.modifiers.control_key() || self.modifiers.alt_key())
            && !self.modifiers.meta_key()
            && let Some(text) = event.text.as_ref()
            && !text.chars().any(char::is_control)
        {
            self.dispatch_input(id, InputEvent::Text(text.to_string()));
        }
    }

    fn dispatch_input(&self, window: WindowId, event: InputEvent) {
        self.input_dispatch
            .dispatch(WindowInputEvent { window, event });
    }

    fn normalized_position(&self, window: WindowId, position: glam::DVec2) -> NormalizedPosition {
        let extent = self.window_extent(window);
        NormalizedPosition::new(glam::vec2(
            normalized_component(position.x, extent.x),
            normalized_component(position.y, extent.y),
        ))
    }

    fn normalized_delta(&self, window: WindowId, delta: glam::DVec2) -> NormalizedDelta {
        let extent = self.window_extent(window);
        NormalizedDelta(glam::vec2(
            normalized_component(delta.x, extent.x),
            normalized_component(delta.y, extent.y),
        ))
    }

    fn window_extent(&self, window: WindowId) -> glam::DVec2 {
        self.windows
            .windows()
            .get(&window)
            .map_or(glam::DVec2::ONE, |window| {
                let size = window.size();
                glam::dvec2(f64::from(size.width.max(1)), f64::from(size.height.max(1)))
            })
    }
}

impl ApplicationHandler for PlatformHandler {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let id = self
            .create_window(event_loop)
            .expect("failed to create main window");
        self.windows.set_main_window(id);
        self.can_create_surfaces = true;
    }

    fn window_event(&mut self, _loop: &dyn ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.dispatch_platform_event(id, &event) {
            return;
        }
        self.dispatch_input_event(id, &event);
    }

    #[cfg(target_os = "macos")]
    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.shutdown_requested {
            event_loop.exit();
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn normalized_component(value: f64, extent: f64) -> f32 {
    (value / extent.max(f64::EPSILON)) as f32
}

fn modifiers_from_winit(modifiers: ModifiersState) -> Modifiers {
    Modifiers {
        alt: modifiers.alt_key(),
        ctrl: modifiers.control_key(),
        shift: modifiers.shift_key(),
        super_key: modifiers.meta_key(),
    }
}

pub(crate) fn logical_key_from_winit(key: &Key) -> Option<LogicalKey> {
    Some(match key {
        Key::Character(text) => {
            let text: &str = text.as_ref();
            if text == " " {
                LogicalKey::Named(NamedKey::Space)
            } else {
                LogicalKey::character(text)
            }
        }
        Key::Named(named) => LogicalKey::Named(named_key_from_winit(*named)?),
        Key::Unidentified(_) | Key::Dead(_) => return None,
    })
}

fn named_key_from_winit(key: WinitNamedKey) -> Option<NamedKey> {
    Some(match key {
        WinitNamedKey::Escape => NamedKey::Escape,
        WinitNamedKey::Tab => NamedKey::Tab,
        WinitNamedKey::Enter => NamedKey::Enter,
        WinitNamedKey::Backspace => NamedKey::Backspace,
        WinitNamedKey::ArrowUp => NamedKey::ArrowUp,
        WinitNamedKey::ArrowDown => NamedKey::ArrowDown,
        WinitNamedKey::ArrowLeft => NamedKey::ArrowLeft,
        WinitNamedKey::ArrowRight => NamedKey::ArrowRight,
        WinitNamedKey::Home => NamedKey::Home,
        WinitNamedKey::End => NamedKey::End,
        WinitNamedKey::PageUp => NamedKey::PageUp,
        WinitNamedKey::PageDown => NamedKey::PageDown,
        WinitNamedKey::Insert => NamedKey::Insert,
        WinitNamedKey::Delete => NamedKey::Delete,
        WinitNamedKey::F1 => NamedKey::Function(1),
        WinitNamedKey::F2 => NamedKey::Function(2),
        WinitNamedKey::F3 => NamedKey::Function(3),
        WinitNamedKey::F4 => NamedKey::Function(4),
        WinitNamedKey::F5 => NamedKey::Function(5),
        WinitNamedKey::F6 => NamedKey::Function(6),
        WinitNamedKey::F7 => NamedKey::Function(7),
        WinitNamedKey::F8 => NamedKey::Function(8),
        WinitNamedKey::F9 => NamedKey::Function(9),
        WinitNamedKey::F10 => NamedKey::Function(10),
        WinitNamedKey::F11 => NamedKey::Function(11),
        WinitNamedKey::F12 => NamedKey::Function(12),
        WinitNamedKey::F13 => NamedKey::Function(13),
        WinitNamedKey::F14 => NamedKey::Function(14),
        WinitNamedKey::F15 => NamedKey::Function(15),
        WinitNamedKey::F16 => NamedKey::Function(16),
        WinitNamedKey::F17 => NamedKey::Function(17),
        WinitNamedKey::F18 => NamedKey::Function(18),
        WinitNamedKey::F19 => NamedKey::Function(19),
        WinitNamedKey::F20 => NamedKey::Function(20),
        WinitNamedKey::F21 => NamedKey::Function(21),
        WinitNamedKey::F22 => NamedKey::Function(22),
        WinitNamedKey::F23 => NamedKey::Function(23),
        WinitNamedKey::F24 => NamedKey::Function(24),
        _ => return None,
    })
}

fn pointer_button(button: &winit::event::ButtonSource) -> Option<PointerButton> {
    Some(match button.clone().mouse_button()? {
        winit::event::MouseButton::Left => PointerButton::Primary,
        winit::event::MouseButton::Right => PointerButton::Secondary,
        winit::event::MouseButton::Middle => PointerButton::Middle,
        winit::event::MouseButton::Back => PointerButton::Back,
        winit::event::MouseButton::Forward => PointerButton::Forward,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_key_conversion_ignores_physical_key_data() {
        assert_eq!(
            logical_key_from_winit(&Key::Character("W".into())),
            Some(LogicalKey::Character("w".to_owned()))
        );
    }

    #[test]
    fn pointer_position_normalization_clamps_to_unit_range() {
        let position = NormalizedPosition::new(glam::vec2(1.5, -1.0));

        assert_eq!(position.0, glam::vec2(1.0, 0.0));
    }

    #[test]
    fn normalized_pointer_delta_can_exceed_unit_range() {
        assert_eq!(
            NormalizedDelta(glam::vec2(2.0, -3.0)).0,
            glam::vec2(2.0, -3.0)
        );
    }
}
