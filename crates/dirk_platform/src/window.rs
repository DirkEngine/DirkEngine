use std::{collections::HashMap, ops::Deref, sync::Arc};

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use winit::{
    dpi::PhysicalSize,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Theme, WindowId},
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

/// Internal platform representation of a window. Holds the
/// [`winit::window::Window`] and other state.
pub struct Window {
    raw: Arc<dyn winit::window::Window>,
    focused: bool,
    theme: Theme,
    /// If the window is completely hidden (minized or covered by another
    /// window)
    occluded: bool,
}

impl Window {
    /// Creates a new default window object using the [`winit::window::Window`].
    #[must_use]
    pub fn new(window: Box<dyn winit::window::Window>) -> Self {
        Self {
            focused: false,
            theme: window.theme().unwrap_or(Theme::Dark),
            occluded: false,
            raw: Arc::from(window),
        }
    }
    /// Returns the unique ID of the window
    #[must_use]
    pub fn id(&self) -> WindowId {
        self.raw.id()
    }
    /// Returns the size of the window's renderable surface. Used by
    /// renderer to create correct surface sizes
    #[must_use]
    pub fn size(&self) -> PhysicalSize<u32> {
        self.raw.surface_size()
    }

    /// Returns an owned native-handle provider suitable for a graphics surface.
    #[must_use]
    pub fn surface_target(&self) -> Arc<WindowSurfaceTarget> {
        Arc::new(WindowSurfaceTarget {
            raw: self.raw.clone(),
        })
    }

    /// Returns the native scale factor for this window.
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.raw.scale_factor()
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

/// Owned native window handles retained by presentation backends.
pub struct WindowSurfaceTarget {
    raw: Arc<dyn winit::window::Window>,
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
