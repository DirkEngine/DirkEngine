use std::{collections::HashMap, ops::Deref, sync::Arc};

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use winit::{
    cursor::CursorIcon,
    dpi::{PhysicalPosition, PhysicalSize},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{ImeCapabilities, ImeEnableRequest, ImeRequest, ImeRequestData, Theme, WindowId},
};

use crate::event::WindowEvent;

#[derive(Default)]
struct PlatformWindowState {
    windows: HashMap<WindowId, Window>,
    main_window: Option<WindowId>,
}

/// Shared access to platform windows owned by the platform subsystem.
#[derive(Clone, Default)]
pub struct PlatformWindows {
    inner: Arc<RwLock<PlatformWindowState>>,
}

impl PlatformWindows {
    /// Applies the cursor and input method state requested by the UI drawn
    /// in a window.
    ///
    /// The UI requests its state every frame, so unchanged state is detected
    /// under the read lock and not reapplied to the native window.
    pub fn apply_ui_state(&self, id: WindowId, state: UiState) {
        let unchanged = self
            .inner
            .read()
            .windows
            .get(&id)
            .is_none_or(|window| window.ui_state == state);
        if unchanged {
            return;
        }
        if let Some(window) = self.inner.write().windows.get_mut(&id) {
            window.apply_ui_state(state);
        }
    }

    pub(crate) fn insert(&self, window: Window) -> WindowId {
        let id = window.id();
        self.inner.write().windows.insert(id, window);
        id
    }

    pub(crate) fn remove(&self, id: WindowId) -> Option<Window> {
        let mut state = self.inner.write();
        if state.main_window == Some(id) {
            state.main_window = None;
        }
        state.windows.remove(&id)
    }

    pub(crate) fn set_main_window(&self, id: WindowId) {
        self.inner.write().main_window = Some(id);
    }

    pub(crate) fn clear(&self) -> usize {
        let mut state = self.inner.write();
        let count = state.windows.len();
        state.windows.clear();
        state.main_window = None;
        count
    }

    pub(crate) fn handle_window_event(&self, event: &WindowEvent) {
        if let Some(window) = self.inner.write().windows.get_mut(event.id()) {
            window.handle_event(event);
        }
    }

    /// Returns a guard for the main platform window.
    ///
    /// # Panics
    ///
    /// Panics if the platform has not created its main window yet.
    #[must_use]
    pub fn main_window(&self) -> MainWindow<'_> {
        let guard = self.inner.read();
        let main_window = guard
            .main_window
            .expect("there should always be a main window");
        MainWindow {
            guard: RwLockReadGuard::map(guard, |state| {
                state
                    .windows
                    .get(&main_window)
                    .expect("there should always be a main window")
            }),
        }
    }

    /// Returns a guard for all live platform windows.
    #[must_use]
    pub fn windows(&self) -> Windows<'_> {
        Windows {
            guard: self.inner.read(),
        }
    }

    /// Returns whether there are no live platform windows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.read().windows.is_empty()
    }
}

/// Cursor and input method state requested by the UI drawn in a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiState {
    /// Cursor icon, or `None` to hide the cursor.
    pub cursor: Option<CursorIcon>,
    /// Text input area, or `None` when the UI is not editing text.
    pub ime: Option<ImeArea>,
}

impl Default for UiState {
    /// The state of a newly created native window.
    fn default() -> Self {
        Self {
            cursor: Some(CursorIcon::Default),
            ime: None,
        }
    }
}

/// Area of a window, in physical pixels, where text is being edited. The
/// input method places its candidate window next to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImeArea {
    /// Left edge.
    pub x: f64,
    /// Top edge.
    pub y: f64,
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl ImeArea {
    fn request_data(self) -> ImeRequestData {
        ImeRequestData::default().with_cursor_area(
            PhysicalPosition::new(self.x, self.y).into(),
            PhysicalSize::new(self.width, self.height).into(),
        )
    }
}

/// Read guard for the main platform window.
pub struct MainWindow<'a> {
    guard: MappedRwLockReadGuard<'a, Window>,
}

impl Deref for MainWindow<'_> {
    type Target = Window;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

/// Read guard for all live platform windows.
pub struct Windows<'a> {
    guard: RwLockReadGuard<'a, PlatformWindowState>,
}

impl Deref for Windows<'_> {
    type Target = HashMap<WindowId, Window>;

    fn deref(&self) -> &Self::Target {
        &self.guard.windows
    }
}

/// Internal platform representation of a window. Holds the platform's
/// mutable window state and shares the native window through a
/// [`WindowSurfaceTarget`].
pub struct Window {
    raw: Arc<WindowSurfaceTarget>,
    focused: bool,
    theme: Theme,
    /// If the window is completely hidden (minized or covered by another
    /// window)
    occluded: bool,
    /// UI state last applied to the native window.
    ui_state: UiState,
}

impl Window {
    /// Creates a new default window object using the [`winit::window::Window`].
    #[must_use]
    pub fn new(window: Box<dyn winit::window::Window>) -> Self {
        Self {
            focused: false,
            theme: window.theme().unwrap_or(Theme::Dark),
            occluded: false,
            ui_state: UiState::default(),
            raw: Arc::new(WindowSurfaceTarget { raw: window }),
        }
    }
    /// Returns the unique ID of the window
    #[must_use]
    pub fn id(&self) -> WindowId {
        self.raw.raw.id()
    }
    /// Returns the size of the window's renderable surface. Used by
    /// renderer to create correct surface sizes
    #[must_use]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.raw.raw.surface_size()
    }

    /// Returns a shared owner of the native window, for graphics surfaces
    /// that must keep it alive after the platform removes this window.
    ///
    /// Callers that only borrow the window use its [`HasWindowHandle`] and
    /// [`HasDisplayHandle`] implementations instead. See
    /// [`WindowSurfaceTarget`] for the thread the last clone must be
    /// dropped on.
    #[must_use]
    pub fn surface_target(&self) -> Arc<WindowSurfaceTarget> {
        self.raw.clone()
    }

    /// Returns the native scale factor for this window.
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.raw.raw.scale_factor()
    }

    /// Returns whether this window is focused.
    #[must_use]
    pub fn focused(&self) -> bool {
        self.focused
    }

    /// Returns the current window theme.
    #[must_use]
    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// Applies the parts of `state` that differ from the last applied state.
    fn apply_ui_state(&mut self, state: UiState) {
        let window = &self.raw.raw;
        if self.ui_state.cursor.is_some() != state.cursor.is_some() {
            window.set_cursor_visible(state.cursor.is_some());
        }
        if let Some(icon) = state.cursor
            && self.ui_state.cursor != Some(icon)
        {
            window.set_cursor(icon.into());
        }

        let request = match (self.ui_state.ime, state.ime) {
            (None, Some(area)) => ImeEnableRequest::new(
                ImeCapabilities::new().with_cursor_area(),
                area.request_data(),
            )
            .map(ImeRequest::Enable),
            (Some(old), Some(area)) if old != area => Some(ImeRequest::Update(area.request_data())),
            (Some(_), None) => Some(ImeRequest::Disable),
            _ => None,
        };
        if let Some(request) = request
            && let Err(error) = window.request_ime_update(request)
        {
            tracing::warn!(?error, "could not update the window input method");
        }

        self.ui_state = state;
    }

    /// Handles [`WindowEvent`]. These should first be proccessed
    /// and accepted by the window.
    pub fn handle_event(&mut self, event: &WindowEvent) {
        if *event.id() != self.id() {
            return;
        }

        match *event {
            // Resizing is handled by the renderer.
            WindowEvent::Resized { .. } => {}
            WindowEvent::Occluded { id: _, occluded } => self.occluded = occluded,
            WindowEvent::FocusChanged { id: _, focused } => self.focused = focused,
            WindowEvent::ThemeChanged { id: _, theme } => self.theme = theme,
        }
    }
}

impl HasWindowHandle for Window {
    fn window_handle(
        &self,
    ) -> Result<winit::raw_window_handle::WindowHandle<'_>, winit::raw_window_handle::HandleError>
    {
        self.raw.window_handle()
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(
        &self,
    ) -> Result<winit::raw_window_handle::DisplayHandle<'_>, winit::raw_window_handle::HandleError>
    {
        self.raw.display_handle()
    }
}

/// Native window owned jointly by the platform's [`Window`] and the graphics
/// surfaces created for it.
///
/// A graphics surface must not outlive its native window, but the platform
/// removes a [`Window`] as soon as it is closed, while the renderer destroys
/// the surface only once the GPU has finished with it. Surfaces therefore
/// hold a clone, which keeps the native window alive until they are gone.
///
/// The last clone must be dropped on the main thread, which runs the platform
/// event loop. Dropping it closes the native window, and on macOS winit
/// dispatches that close synchronously to the main thread, so dropping it on
/// another thread while the main thread waits for that thread deadlocks.
pub struct WindowSurfaceTarget {
    raw: Box<dyn winit::window::Window>,
}

impl HasWindowHandle for WindowSurfaceTarget {
    fn window_handle(
        &self,
    ) -> Result<winit::raw_window_handle::WindowHandle<'_>, winit::raw_window_handle::HandleError>
    {
        self.raw.window_handle()
    }
}

impl HasDisplayHandle for WindowSurfaceTarget {
    fn display_handle(
        &self,
    ) -> Result<winit::raw_window_handle::DisplayHandle<'_>, winit::raw_window_handle::HandleError>
    {
        self.raw.display_handle()
    }
}
