use dirk_events::Event;
use winit::window::WindowId;

/// All platform events.
/// These are specific to global platform stuff. No input or window
/// specific events are listed here.
/// The only exeptions are window closing and creating events.
#[derive(Debug, Clone, Event)]
#[expect(
    missing_docs,
    reason = "Each variant's field is the id of the window it is about."
)]
pub enum PlatformEvent {
    /// Window created event.
    WindowCreated { id: WindowId },
    /// The OS is asking us to close this window.
    /// This should be when the renderer window is destroyed.
    WindowCloseRequested { id: WindowId },
    /// The window has been finally destroyed. This event should not be
    /// used as all window related objects should have been destroyed on
    /// [`Self::WindowCloseRequested`].
    WindowDestroyed { id: WindowId },
}

/// All window specific events.
#[derive(Debug, Clone, Event)]
#[expect(
    missing_docs,
    reason = "Variant fields are the window's id and its new state, named after the variant."
)]
pub enum WindowEvent {
    Resized {
        id: WindowId,
        width: u32,
        height: u32,
    },
    FocusChanged {
        id: WindowId,
        focused: bool,
    },
    Occluded {
        id: WindowId,
        occluded: bool,
    },
    ThemeChanged {
        id: WindowId,
        theme: winit::window::Theme,
    },
}

impl WindowEvent {
    /// Returns the ID of the window referenced in the event. This is
    /// just a match that extracts the ID out of every variant.
    #[must_use]
    pub fn id(&self) -> &WindowId {
        match self {
            Self::Resized { id, .. }
            | Self::Occluded { id, .. }
            | Self::FocusChanged { id, .. }
            | Self::ThemeChanged { id, .. } => id,
        }
    }
}

/// Input produced by a platform window.
#[derive(Debug, Clone, Event)]
#[event("window input")]
pub struct WindowInputEvent {
    /// Window that produced this input.
    pub window: WindowId,
    /// Window-local normalized input event.
    pub event: dirk_input::InputEvent,
}
