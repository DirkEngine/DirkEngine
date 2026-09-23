//! Engine-owned editor subsystem and capability API.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

use anyhow::Context as _;
use dirk_universe::Universe;
use egui_dock::{
    DockArea, DockState, NodeIndex, Split, SurfaceIndex, TabViewer, tab_viewer::OnCloseResponse,
};
use parking_lot::Mutex;

use crate::{EngineBuildContext, EngineHandle, Error, Result};

pub mod commands;
use commands::{EditorCommand, EditorCommandSender};

#[cfg(test)]
mod tests;

/// The category that all universe-related windows have.
pub const UNIVERSE_CATEGORY: &str = "Universe";
/// Category reserved for viewports.
pub const VIEWPORT_CATEGORY: &str = "Viewport";
/// Category that all engine specific windows have.
/// This includes settings, editor diagnostics, ...
pub const EDITOR_CATEGORY: &str = "Editor";

/// Editor lifecycle subsystem owned by the engine.
pub trait EditorSubsystem: Send + 'static {
    /// The subsystem name used for diagnostics.
    fn name(&self) -> &'static str;

    /// Starts the editor subsystem after regular subsystems have started.
    ///
    /// # Errors
    ///
    /// Returns an error if startup fails.
    fn start(&mut self, _engine: &EngineHandle, _editor: &EditorServices) -> anyhow::Result<()> {
        Ok(())
    }

    /// Advances the editor subsystem by one engine tick.
    ///
    /// # Errors
    ///
    /// Returns an error if ticking fails.
    fn tick(
        &mut self,
        _engine: &EngineHandle,
        _editor: &EditorServices,
        _delta_time: f64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Shuts the editor subsystem down.
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails.
    fn shutdown(&mut self, _engine: &EngineHandle, _editor: &EditorServices) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Build-time context for editor subsystem factories.
pub struct EditorBuildContext<'a> {
    /// Mutable engine build context for core services and typed resources.
    pub engine: &'a mut EngineBuildContext,
    /// Shared editor services available to editor subsystems.
    pub editor: &'a EditorServices,
}

/// Per-frame context passed to editor rendering.
pub struct EditorRenderContext<'a> {
    /// Seconds elapsed since the previous engine tick.
    delta_time: f64,
    /// Shared engine handle.
    pub handle: &'a EngineHandle,
}

impl<'a> EditorRenderContext<'a> {
    /// Creates a context for rendering editor UI.
    #[must_use]
    pub fn new(delta_time: f64, handle: &'a EngineHandle) -> Self {
        Self { delta_time, handle }
    }

    /// Returns the seconds elapsed since the previous engine tick.
    #[must_use]
    pub fn delta_time(&self) -> f64 {
        self.delta_time
    }
}

/// Per-frame context passed to editor UI capabilities.
pub struct EditorUiContext<'a> {
    /// Seconds elapsed since the previous engine tick.
    delta_time: f64,
    commands: EditorCommandSender,
    /// Shared engine handle.
    pub handle: &'a EngineHandle,
    /// The Universe
    universe: &'a Universe,
}

impl EditorUiContext<'_> {
    /// Returns the seconds elapsed since the previous engine tick.
    #[must_use]
    pub fn delta_time(&self) -> f64 {
        self.delta_time
    }

    /// Requests that an editor window be opened.
    pub fn open_window(&self, id: EditorWindowId) {
        self.commands.open_window(id);
    }

    /// Returns an immutable reference to the [`Universe`].
    #[must_use]
    pub fn universe(&self) -> &Universe {
        self.universe
    }

    /// Returns the editor command sender for this UI pass.
    #[must_use]
    pub fn commands(&self) -> &EditorCommandSender {
        &self.commands
    }
}

/// Static metadata for an editor-native window capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditorWindowDescriptor {
    /// Window title.
    pub title: String,
    /// Window category used by the built-in Windows menu.
    pub category: String,
    /// Whether the window starts open after registration.
    pub default_open: bool,
    /// Whether the window is shown in the built-in Windows menu.
    pub show_in_list: bool,
}

impl Default for EditorWindowDescriptor {
    fn default() -> Self {
        Self {
            title: String::new(),
            category: String::new(),
            default_open: false,
            show_in_list: true,
        }
    }
}

/// Static metadata for an editor menu capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditorMenuDescriptor {
    /// Menu title shown in the editor menu bar.
    pub title: String,
}

/// Snapshot of a registered editor window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditorWindowInfo {
    /// Stable window identifier.
    pub id: EditorWindowId,
    /// Window title.
    pub title: String,
    /// Window category.
    pub category: String,
    /// Whether the window is currently open.
    pub open: bool,
    /// Whether the window is shown in the built-in Windows menu.
    pub show_in_list: bool,
}

/// Global editor state snapshot and controls available to menu capabilities.
pub struct EditorMenuContext<'a> {
    windows: &'a [EditorWindowInfo],
    commands: EditorCommandSender,
}

impl<'a> EditorMenuContext<'a> {
    fn new(windows: &'a [EditorWindowInfo], commands: EditorCommandSender) -> Self {
        Self { windows, commands }
    }

    /// Returns all registered windows in registration order.
    #[must_use]
    pub fn windows(&self) -> &[EditorWindowInfo] {
        self.windows
    }

    /// Returns a registered window by id.
    #[must_use]
    pub fn window(&self, id: EditorWindowId) -> Option<&EditorWindowInfo> {
        self.windows.iter().find(|window| window.id == id)
    }

    /// Returns whether a registered window is currently open.
    #[must_use]
    pub fn is_open(&self, id: EditorWindowId) -> Option<bool> {
        self.window(id).map(|window| window.open)
    }

    /// Requests that an editor window be opened.
    pub fn open_window(&mut self, id: EditorWindowId) {
        self.commands.open_window(id);
    }

    /// Returns the editor command sender for this UI pass.
    #[must_use]
    pub fn commands(&self) -> &EditorCommandSender {
        &self.commands
    }
}

/// Stable editor window identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EditorWindowId(u64);

impl EditorWindowId {
    /// Returns the raw numeric identifier.
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Stable editor menu identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EditorMenuId(u64);

impl EditorMenuId {
    /// Returns the raw numeric identifier.
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Editor window.
pub trait EditorWindow: Send + 'static {
    /// Returns this window's descriptor.
    fn descriptor(&self) -> EditorWindowDescriptor;

    /// Draws this window's contents.
    ///
    /// # Errors
    ///
    /// Returns an error when the window cannot complete drawing.
    fn ui(&mut self, ui: &mut egui::Ui, context: &mut EditorUiContext<'_>) -> anyhow::Result<()>;
}

/// Editor menu.
pub trait EditorMenu: Send + 'static {
    /// Returns this menu's descriptor.
    fn descriptor(&self) -> EditorMenuDescriptor;

    /// Draws this menu's contents.
    ///
    /// # Errors
    ///
    /// Returns an error when the menu cannot complete drawing.
    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut EditorUiContext<'_>,
        editor: &mut EditorMenuContext<'_>,
    ) -> anyhow::Result<()>;
}

/// Generic editor style hook.
#[derive(Clone)]
pub struct EditorStyle {
    apply: Arc<dyn Fn(&egui::Context) + Send + Sync + 'static>,
}

impl EditorStyle {
    /// Creates a style hook from a callback.
    pub fn new<F>(apply: F) -> Self
    where
        F: Fn(&egui::Context) + Send + Sync + 'static,
    {
        Self {
            apply: Arc::new(apply),
        }
    }

    /// Applies this style hook to an egui context.
    pub fn apply(&self, ctx: &egui::Context) {
        (self.apply)(ctx);
    }
}

/// Shared state with editor services.
///
/// Services are extensions of the editor. Notably, windows & menus.
#[derive(Clone)]
pub struct EditorServices {
    state: Arc<Mutex<EditorServicesState>>,
    next_window_id: Arc<AtomicU64>,
    next_menu_id: Arc<AtomicU64>,
}

impl EditorServices {
    /// Creates empty editor services.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(EditorServicesState::new())),
            next_window_id: Arc::new(AtomicU64::new(0)),
            next_menu_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Registers an editor-native window.
    pub fn add_window<W>(&self, window: W) -> EditorWindowId
    where
        W: EditorWindow,
    {
        let descriptor = window.descriptor();
        let default_open = descriptor.default_open;
        let id = EditorWindowId(self.next_window_id.fetch_add(1, Ordering::Relaxed));
        let mut state = self.state.lock();
        state.windows.push(RegisteredWindow {
            id,
            descriptor,
            window: Arc::new(Mutex::new(Box::new(window))),
        });
        state
            .window_states
            .insert(id, WindowState { open: default_open });
        if default_open {
            if state.dock_layout_bootstrapped {
                if state.is_window_category(id, VIEWPORT_CATEGORY)
                    && state.find_dock_tab_by_category(VIEWPORT_CATEGORY).is_none()
                {
                    state.rebuild_default_dock_layout();
                } else {
                    state.insert_window_tab(id);
                }
            } else {
                state.rebuild_default_dock_layout();
            }
        }

        id
    }

    /// Registers an editor-native window from a callback.
    pub fn add_window_fn<F>(&self, descriptor: EditorWindowDescriptor, ui: F) -> EditorWindowId
    where
        F: FnMut(&mut egui::Ui, &mut EditorUiContext<'_>) -> anyhow::Result<()> + Send + 'static,
    {
        self.add_window(FnEditorWindow { descriptor, ui })
    }

    /// Removes a registered editor-native window.
    ///
    /// Returns `true` when the window existed and was removed. This removes the
    /// window metadata, open state, and any matching dock tab without affecting
    /// menus or other windows.
    #[allow(clippy::must_use_candidate)]
    // this function actually mutates state. clippy just can't tell as its a mutex
    pub fn remove_window(&self, id: EditorWindowId) -> bool {
        self.state.lock().remove_window(id)
    }

    /// Registers an editor menu capability.
    pub fn add_menu<M>(&self, menu: M) -> EditorMenuId
    where
        M: EditorMenu,
    {
        let descriptor = menu.descriptor();
        let id = EditorMenuId(self.next_menu_id.fetch_add(1, Ordering::Relaxed));
        self.state.lock().menus.push(RegisteredMenu {
            id,
            title: descriptor.title,
            menu: Arc::new(Mutex::new(Box::new(menu))),
        });
        id
    }

    /// Registers an editor menu capability from a callback.
    pub fn add_menu_fn<F>(&self, descriptor: EditorMenuDescriptor, ui: F) -> EditorMenuId
    where
        F: FnMut(
                &mut egui::Ui,
                &mut EditorUiContext<'_>,
                &mut EditorMenuContext<'_>,
            ) -> anyhow::Result<()>
            + Send
            + 'static,
    {
        self.add_menu(FnEditorMenu { descriptor, ui })
    }

    /// Replaces the style stack with one style hook.
    pub fn set_style(&self, style: impl Into<EditorStyle>) {
        self.state.lock().styles = vec![style.into()];
    }

    /// Adds a style hook to the style stack.
    pub fn add_style(&self, style: impl Into<EditorStyle>) {
        self.state.lock().styles.push(style.into());
    }

    /// Clears the configured editor style hooks.
    pub fn clear_style(&self) {
        self.state.lock().styles.clear();
    }

    /// Returns whether a registered window is currently open.
    #[must_use]
    pub fn is_open(&self, id: EditorWindowId) -> Option<bool> {
        self.state
            .lock()
            .window_states
            .get(&id)
            .map(|state| state.open)
    }

    /// Returns registered windows in registration order.
    #[must_use]
    pub fn windows(&self) -> Vec<EditorWindowInfo> {
        self.state.lock().windows()
    }

    /// Returns a registered window by id.
    #[must_use]
    pub fn window(&self, id: EditorWindowId) -> Option<EditorWindowInfo> {
        self.state
            .lock()
            .windows()
            .into_iter()
            .find(|window| window.id == id)
    }

    /// Returns the registered window count.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.state.lock().windows.len()
    }

    /// Returns the registered menu count.
    #[must_use]
    pub fn menu_count(&self) -> usize {
        self.state.lock().menus.len()
    }

    /// Returns registered window titles in registration order.
    #[must_use]
    pub fn window_titles(&self) -> Vec<String> {
        self.state
            .lock()
            .windows
            .iter()
            .map(|window| window.descriptor.title.clone())
            .collect()
    }

    /// Returns registered menu titles in registration order.
    #[must_use]
    pub fn menu_titles(&self) -> Vec<String> {
        self.state
            .lock()
            .menus
            .iter()
            .map(|menu| menu.title.clone())
            .collect()
    }

    /// Renders editor menus and windows.
    ///
    /// # Errors
    ///
    /// Returns the first error emitted by a registered capability callback.
    pub fn render_ui(
        &self,
        ctx: &egui::Context,
        context: &EditorRenderContext<'_>,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        let styles = self.state.lock().styles.clone();
        for style in styles {
            style.apply(ctx);
        }

        let (editor_commands, command_receiver) = mpsc::channel();
        let editor_commands = EditorCommandSender::new(editor_commands);

        let (menus, window_infos) = {
            let state = self.state.lock();
            (state.menus.clone(), state.windows())
        };
        Self::render_menus(
            &menus,
            &window_infos,
            ctx,
            context,
            &editor_commands,
            universe,
        )?;

        let (windows, mut dock_state) = {
            let mut state = self.state.lock();
            state.apply_commands(command_receiver.try_iter());
            state.bootstrap_default_dock_layout();
            state.sync_dock_tabs_with_open_windows();
            (state.windows.clone(), state.dock_state.clone())
        };

        let mut closed_windows = Vec::new();
        let result = Self::render_windows(
            &windows,
            &mut dock_state,
            &mut closed_windows,
            ctx,
            context,
            &editor_commands,
            universe,
        );

        let mut state = self.state.lock();
        for id in closed_windows {
            if let Some(window) = state.window_states.get_mut(&id) {
                window.open = false;
            }
        }
        state.dock_state = dock_state;
        // Callbacks may have added or removed windows while the snapshot was
        // rendered. Reconcile those changes before applying open commands.
        let new_viewport = state.windows.iter().any(|window| {
            window.descriptor.category == VIEWPORT_CATEGORY
                && !windows.iter().any(|rendered| rendered.id == window.id)
                && state
                    .window_states
                    .get(&window.id)
                    .is_some_and(|window| window.open)
        });
        if new_viewport && state.find_dock_tab_by_category(VIEWPORT_CATEGORY).is_none() {
            state.rebuild_default_dock_layout();
        } else {
            state.sync_dock_tabs_with_open_windows();
        }
        state.apply_commands(command_receiver.try_iter());
        result
    }
    fn render_menus(
        menus: &[RegisteredMenu],
        windows: &[EditorWindowInfo],
        ctx: &egui::Context,
        context: &EditorRenderContext<'_>,
        editor_commands: &EditorCommandSender,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        if menus.is_empty() {
            return Ok(());
        }

        let mut menu_context = EditorMenuContext::new(windows, (*editor_commands).clone());
        let mut context = EditorUiContext {
            delta_time: context.delta_time(),
            commands: (*editor_commands).clone(),
            handle: context.handle,
            universe,
        };

        let mut result = Ok(());
        egui::TopBottomPanel::top("dirk_editor_menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                for menu in menus {
                    ui.push_id(menu.id.raw(), |ui| {
                        ui.menu_button(menu.title.clone(), |ui| {
                            if result.is_ok() {
                                result = menu
                                    .menu
                                    .lock()
                                    .ui(ui, &mut context, &mut menu_context)
                                    .with_context(|| {
                                        format!("menu `{}` failed to render", menu.title)
                                    });
                            }
                        });
                    });
                }
            });
        });

        result
    }

    fn render_windows(
        windows: &[RegisteredWindow],
        dock_state: &mut DockState<EditorWindowId>,
        closed_windows: &mut Vec<EditorWindowId>,
        ctx: &egui::Context,
        context: &EditorRenderContext<'_>,
        editor_commands: &EditorCommandSender,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        let mut result = Ok(());
        let mut ui_context = EditorUiContext {
            delta_time: context.delta_time(),
            commands: (*editor_commands).clone(),
            handle: context.handle,
            universe,
        };
        let mut tab_viewer = EditorDockTabViewer {
            windows,
            closed_windows,
            context: &mut ui_context,
            result: &mut result,
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            DockArea::new(dock_state)
                .style(egui_dock::Style::from_egui(ui.style()))
                .show_leaf_collapse_buttons(false)
                .show_inside(ui, &mut tab_viewer);
        });

        result
    }
}

impl Default for EditorServices {
    fn default() -> Self {
        Self::new()
    }
}

struct EditorServicesState {
    windows: Vec<RegisteredWindow>,
    window_states: HashMap<EditorWindowId, WindowState>,
    dock_state: DockState<EditorWindowId>,
    dock_layout_bootstrapped: bool,
    menus: Vec<RegisteredMenu>,
    styles: Vec<EditorStyle>,
}

impl EditorServicesState {
    fn new() -> Self {
        Self {
            windows: Vec::new(),
            window_states: HashMap::new(),
            dock_state: DockState::new(Vec::new()),
            dock_layout_bootstrapped: false,
            menus: Vec::new(),
            styles: Vec::new(),
        }
    }

    fn windows(&self) -> Vec<EditorWindowInfo> {
        self.windows
            .iter()
            .map(|window| EditorWindowInfo {
                id: window.id,
                title: window.descriptor.title.clone(),
                category: window.descriptor.category.clone(),
                open: self
                    .window_states
                    .get(&window.id)
                    .is_some_and(|state| state.open),
                show_in_list: window.descriptor.show_in_list,
            })
            .collect()
    }

    fn apply_commands(&mut self, commands: impl IntoIterator<Item = EditorCommand>) {
        for command in commands {
            match command {
                EditorCommand::OpenWindow(id) => {
                    if let Some(state) = self.window_states.get_mut(&id) {
                        state.open = true;
                        self.insert_window_tab(id);
                        self.focus_window_tab(id);
                    }
                }
            }
        }
    }

    fn remove_window(&mut self, id: EditorWindowId) -> bool {
        let Some(index) = self.windows.iter().position(|window| window.id == id) else {
            return false;
        };

        self.windows.remove(index);
        self.window_states.remove(&id);
        self.dock_state.retain_tabs(|tab| *tab != id);
        true
    }

    fn sync_dock_tabs_with_open_windows(&mut self) {
        let open_windows = self.open_window_ids();

        self.dock_state
            .retain_tabs(|tab| open_windows.contains(tab));

        for id in open_windows {
            if !self.dock_contains_window(id) {
                self.insert_window_tab(id);
            }
        }
    }

    fn bootstrap_default_dock_layout(&mut self) {
        if self.dock_layout_bootstrapped {
            return;
        }

        self.rebuild_default_dock_layout();
        self.dock_layout_bootstrapped = true;
    }

    fn rebuild_default_dock_layout(&mut self) {
        let open_windows = self.open_window_ids();
        self.dock_state = DockState::new(Vec::new());

        let Some(center) = self.default_center_window(&open_windows) else {
            return;
        };

        self.dock_state = DockState::new(vec![center]);

        let right_windows: Vec<_> = open_windows
            .iter()
            .copied()
            .filter(|id| *id != center && self.is_window_category(*id, UNIVERSE_CATEGORY))
            .collect();
        if !right_windows.is_empty() {
            self.dock_state
                .main_surface_mut()
                .split_right(NodeIndex::root(), 0.75, right_windows);
        }

        let bottom_windows: Vec<_> = open_windows
            .iter()
            .copied()
            .filter(|id| *id != center && self.is_window_category(*id, EDITOR_CATEGORY))
            .collect();
        if !bottom_windows.is_empty() {
            if let Some((node, _tab)) = self.dock_state.main_surface().find_tab(&center) {
                self.dock_state
                    .main_surface_mut()
                    .split_below(node, 0.70, bottom_windows);
            } else {
                for tab in bottom_windows {
                    self.dock_state.push_to_first_leaf(tab);
                }
            }
        }

        for id in open_windows {
            if !self.dock_contains_window(id) {
                self.insert_window_tab(id);
            }
        }
    }

    fn open_window_ids(&self) -> Vec<EditorWindowId> {
        self.windows
            .iter()
            .filter(|window| {
                self.window_states
                    .get(&window.id)
                    .is_some_and(|state| state.open)
            })
            .map(|window| window.id)
            .collect()
    }

    fn default_center_window(&self, open_windows: &[EditorWindowId]) -> Option<EditorWindowId> {
        open_windows
            .iter()
            .copied()
            .find(|id| self.is_window_category(*id, VIEWPORT_CATEGORY))
            .or_else(|| open_windows.first().copied())
    }

    fn insert_window_tab(&mut self, id: EditorWindowId) {
        if self.dock_contains_window(id) || !self.window_exists(id) {
            return;
        }

        if self.dock_state.main_surface().is_empty() {
            self.dock_state.push_to_first_leaf(id);
            return;
        }

        if let Some(descriptor) = self.window_descriptor(id) {
            self.insert_near_category(id, &descriptor.category.clone());
        } else {
            self.dock_state.push_to_focused_leaf(id);
        }
    }

    fn insert_near_category(&mut self, id: EditorWindowId, category: &str) {
        if let Some((surface, node, _tab)) = self.find_dock_tab_by_category(category) {
            self.dock_state
                .set_focused_node_and_surface((surface, node));
            self.dock_state.push_to_focused_leaf(id);
            return;
        }

        let split = if category == EDITOR_CATEGORY {
            Split::Below
        } else {
            Split::Right
        };
        self.dock_state.split(
            (SurfaceIndex::main(), NodeIndex::root()),
            split,
            0.75,
            egui_dock::Node::leaf(id),
        );
    }

    fn focus_window_tab(&mut self, id: EditorWindowId) {
        if let Some((surface, node, tab)) = self.dock_state.find_tab(&id) {
            self.dock_state
                .set_focused_node_and_surface((surface, node));
            self.dock_state.set_active_tab((surface, node, tab));
        }
    }

    fn dock_contains_window(&self, id: EditorWindowId) -> bool {
        self.dock_state.find_tab(&id).is_some()
    }

    #[cfg(test)]
    fn dock_tab_count(&self) -> usize {
        self.dock_state
            .iter_all_tabs()
            .filter(|&(_index, tab)| self.window_exists(*tab))
            .count()
    }

    fn find_dock_tab_by_category(
        &self,
        category: &str,
    ) -> Option<(SurfaceIndex, NodeIndex, egui_dock::TabIndex)> {
        self.dock_state.find_tab_from(|tab| {
            self.window_descriptor(*tab)
                .is_some_and(|descriptor| descriptor.category == category)
        })
    }

    fn is_window_category(&self, id: EditorWindowId, category: &str) -> bool {
        self.window_descriptor(id)
            .is_some_and(|descriptor| descriptor.category == category)
    }

    fn window_exists(&self, id: EditorWindowId) -> bool {
        self.windows.iter().any(|window| window.id == id)
    }

    fn window_descriptor(&self, id: EditorWindowId) -> Option<&EditorWindowDescriptor> {
        self.windows
            .iter()
            .find(|window| window.id == id)
            .map(|window| &window.descriptor)
    }
}

#[derive(Clone)]
struct RegisteredWindow {
    id: EditorWindowId,
    descriptor: EditorWindowDescriptor,
    window: Arc<Mutex<Box<dyn EditorWindow>>>,
}

struct WindowState {
    open: bool,
}

struct EditorDockTabViewer<'a, 'b> {
    windows: &'a [RegisteredWindow],
    closed_windows: &'a mut Vec<EditorWindowId>,
    context: &'a mut EditorUiContext<'b>,
    result: &'a mut anyhow::Result<()>,
}

impl TabViewer for EditorDockTabViewer<'_, '_> {
    type Tab = EditorWindowId;

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        egui::Id::new(("dirk_editor_window", tab.raw()))
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        self.windows
            .iter()
            .find(|window| window.id == *tab)
            .map_or_else(
                || "<missing window>".into(),
                |window| window.descriptor.title.clone().into(),
            )
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if self.result.is_err() {
            return;
        }

        let Some(window) = self.windows.iter().find(|window| window.id == *tab) else {
            ui.label("Window is no longer registered");
            return;
        };

        let title = window.descriptor.title.clone();
        *self.result = window
            .window
            .lock()
            .ui(ui, self.context)
            .with_context(|| format!("window `{title}` failed to render"));
    }

    fn is_closeable(&self, _tab: &Self::Tab) -> bool {
        true
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
        self.closed_windows.push(*tab);
        OnCloseResponse::Close
    }
}

pub(crate) struct EditorRuntime {
    services: EditorServices,
    subsystems: Vec<Box<dyn EditorSubsystem>>,
}

impl EditorRuntime {
    pub(crate) fn new(services: EditorServices, subsystems: Vec<Box<dyn EditorSubsystem>>) -> Self {
        Self {
            services,
            subsystems,
        }
    }

    pub(crate) fn start(&mut self, engine: &EngineHandle) -> Result<()> {
        for subsystem in &mut self.subsystems {
            let name = subsystem.name();
            subsystem
                .start(engine, &self.services)
                .map_err(|source| Error::EditorSubsystemFailedStart { name, source })?;
        }
        self.services.state.lock().bootstrap_default_dock_layout();
        Ok(())
    }

    pub(crate) fn tick(&mut self, delta_time: f64, engine: &EngineHandle) -> crate::Result<()> {
        for subsystem in &mut self.subsystems {
            let name = subsystem.name();
            subsystem
                .tick(engine, &self.services, delta_time)
                .map_err(|source| Error::EditorSubsystemFailedTick { name, source })?;
        }
        Ok(())
    }

    pub(crate) fn shutdown(&mut self, engine: &EngineHandle) -> crate::Result<()> {
        let mut first_error = None;
        while let Some(mut subsystem) = self.subsystems.pop() {
            let name = subsystem.name();
            if let Err(source) = subsystem.shutdown(engine, &self.services) {
                let err = Error::EditorSubsystemFailedShutdown { name, source };
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    tracing::error!("additional editor shutdown failure: {err:#}");
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
struct RegisteredMenu {
    id: EditorMenuId,
    title: String,
    menu: Arc<Mutex<Box<dyn EditorMenu>>>,
}

struct FnEditorWindow<F> {
    descriptor: EditorWindowDescriptor,
    ui: F,
}

impl<F> EditorWindow for FnEditorWindow<F>
where
    F: FnMut(&mut egui::Ui, &mut EditorUiContext<'_>) -> anyhow::Result<()> + Send + 'static,
{
    fn descriptor(&self) -> EditorWindowDescriptor {
        self.descriptor.clone()
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut EditorUiContext<'_>) -> anyhow::Result<()> {
        (self.ui)(ui, context)
    }
}

struct FnEditorMenu<F> {
    descriptor: EditorMenuDescriptor,
    ui: F,
}

impl<F> EditorMenu for FnEditorMenu<F>
where
    F: FnMut(
            &mut egui::Ui,
            &mut EditorUiContext<'_>,
            &mut EditorMenuContext<'_>,
        ) -> anyhow::Result<()>
        + Send
        + 'static,
{
    fn descriptor(&self) -> EditorMenuDescriptor {
        self.descriptor.clone()
    }

    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut EditorUiContext<'_>,
        editor: &mut EditorMenuContext<'_>,
    ) -> anyhow::Result<()> {
        (self.ui)(ui, context, editor)
    }
}

pub(crate) type EditorSubsystemFactory = Box<
    dyn FnOnce(&mut EditorBuildContext<'_>) -> crate::Result<Box<dyn EditorSubsystem>>
        + Send
        + 'static,
>;
