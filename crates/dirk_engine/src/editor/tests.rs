use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use dirk_universe::Universe;
use parking_lot::Mutex;

use crate::{
    EngineBuilder,
    editor::{
        EDITOR_CATEGORY, EditorMenuDescriptor, EditorRenderContext, EditorRuntime, EditorServices,
        EditorServicesState, EditorStyle, EditorSubsystem, EditorWindowDescriptor, EditorWindowId,
        EditorWindowInfo, UNIVERSE_CATEGORY, commands::EditorCommand,
    },
    errors::{Error, Result},
    tests::build_context,
};

impl EditorServices {
    pub(crate) fn open_window_for_tests(&self, id: EditorWindowId) {
        self.state
            .lock()
            .apply_commands(std::iter::once(EditorCommand::OpenWindow(id)));
    }

    pub(crate) fn bootstrap_default_dock_layout_for_tests(&self) {
        self.state.lock().bootstrap_default_dock_layout();
    }

    pub(crate) fn close_window_tab_for_tests(&self, id: EditorWindowId) {
        self.state.lock().close_window_tab(id);
    }

    pub(crate) fn dock_contains_window_for_tests(&self, id: EditorWindowId) -> bool {
        self.state.lock().dock_contains_window(id)
    }

    pub(crate) fn dock_tab_count_for_tests(&self) -> usize {
        self.state.lock().dock_tab_count()
    }

    pub(crate) fn dock_windows_share_leaf_for_tests(
        &self,
        first: EditorWindowId,
        second: EditorWindowId,
    ) -> bool {
        self.state.lock().dock_windows_share_leaf(first, second)
    }

    pub(crate) fn move_window_to_first_dock_leaf_for_tests(&self, id: EditorWindowId) {
        self.state.lock().move_window_to_first_dock_leaf(id);
    }

    pub(crate) fn window_is_closeable_for_tests(&self, id: EditorWindowId) -> bool {
        self.state.lock().window_is_closeable(id)
    }

    pub(crate) fn render_menu_for_tests(
        &self,
        title: &str,
        ui: &mut egui::Ui,
        context: &EditorRenderContext<'_>,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        use std::sync::mpsc;

        use anyhow::Context;

        use crate::editor::{EditorMenuContext, EditorUiContext, commands::EditorCommandSender};

        let (editor_commands, command_receiver) = mpsc::channel();
        let editor_commands = EditorCommandSender::new(editor_commands);
        let (windows, menu) = {
            let state = self.state.lock();
            (
                state.windows(),
                state.menus.iter().find(|menu| menu.title == title).cloned(),
            )
        };
        let mut menu_context = EditorMenuContext::new(&windows, editor_commands.clone());
        let mut ui_context = EditorUiContext {
            delta_time: context.delta_time(),
            commands: editor_commands,
            handle: context.handle,
            universe,
        };

        let Some(menu) = menu else {
            return Err(anyhow::anyhow!("menu `{title}` is not registered"));
        };

        menu.menu
            .lock()
            .ui(ui, &mut ui_context, &mut menu_context)
            .with_context(|| format!("menu `{title}` failed to render"))?;
        self.state
            .lock()
            .apply_commands(command_receiver.try_iter());
        Ok(())
    }
}

impl EditorServicesState {
    #[cfg(test)]
    fn close_window_tab(&mut self, id: EditorWindowId) {
        if let Some(state) = self.window_states.get_mut(&id) {
            state.open = false;
        }
        if let Some(index) = self.dock_state.find_tab(&id) {
            self.dock_state.remove_tab(index);
        }
    }

    #[cfg(test)]
    fn dock_windows_share_leaf(&self, first: EditorWindowId, second: EditorWindowId) -> bool {
        let first = self
            .dock_state
            .find_tab(&first)
            .map(|(surface, node, _tab)| (surface, node));
        let second = self
            .dock_state
            .find_tab(&second)
            .map(|(surface, node, _tab)| (surface, node));

        first.is_some() && first == second
    }

    #[cfg(test)]
    fn move_window_to_first_dock_leaf(&mut self, id: EditorWindowId) {
        if let Some(index) = self.dock_state.find_tab(&id) {
            self.dock_state.remove_tab(index);
            self.dock_state.push_to_first_leaf(id);
        }
    }

    #[cfg(test)]
    fn window_is_closeable(&self, id: EditorWindowId) -> bool {
        self.window_exists(id)
    }
}

impl EditorRuntime {
    #[cfg(test)]
    pub(crate) fn empty_for_tests() -> Self {
        Self::new(EditorServices::new(), Vec::new())
    }
}

struct FirstEditorSubsystem {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl EditorSubsystem for FirstEditorSubsystem {
    fn name(&self) -> &'static str {
        "first-editor"
    }

    fn start(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
    ) -> anyhow::Result<()> {
        self.events.lock().push("first-start");
        Ok(())
    }

    fn tick(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
        _delta_time: f64,
    ) -> anyhow::Result<()> {
        self.events.lock().push("first-tick");
        Ok(())
    }

    fn shutdown(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
    ) -> anyhow::Result<()> {
        self.events.lock().push("first-shutdown");
        Ok(())
    }
}

struct SecondEditorSubsystem {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl EditorSubsystem for SecondEditorSubsystem {
    fn name(&self) -> &'static str {
        "second-editor"
    }

    fn start(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
    ) -> anyhow::Result<()> {
        self.events.lock().push("second-start");
        Ok(())
    }

    fn tick(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
        _delta_time: f64,
    ) -> anyhow::Result<()> {
        self.events.lock().push("second-tick");
        Ok(())
    }

    fn shutdown(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
    ) -> anyhow::Result<()> {
        self.events.lock().push("second-shutdown");
        Ok(())
    }
}

struct DuplicateEditorSubsystem;

impl EditorSubsystem for DuplicateEditorSubsystem {
    fn name(&self) -> &'static str {
        "duplicate-editor"
    }
}

struct StartFailingEditorSubsystem;

impl EditorSubsystem for StartFailingEditorSubsystem {
    fn name(&self) -> &'static str {
        "start-failing-editor"
    }

    fn start(
        &mut self,
        _engine: &crate::EngineHandle,
        _editor: &EditorServices,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("editor start failed"))
    }
}

fn render_services_with_input(
    services: &EditorServices,
    universe: &Universe,
    raw_input: egui::RawInput,
) -> anyhow::Result<()> {
    let ctx = egui::Context::default();
    ctx.begin_pass(raw_input);

    let handle = build_context().handle().clone();
    let frame = EditorRenderContext::new(0.016, &handle);
    let result = services.render_ui(&ctx, &frame, universe);
    let _ = ctx.end_pass();
    result
}

fn render_services(services: &EditorServices, universe: &Universe) -> anyhow::Result<()> {
    render_services_with_input(
        services,
        universe,
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..egui::RawInput::default()
        },
    )
}

fn descriptor(title: &str, default_open: bool) -> EditorWindowDescriptor {
    EditorWindowDescriptor {
        title: title.to_owned(),
        category: "Tests".to_owned(),
        default_open,
        show_in_list: true,
    }
}

#[test]
fn editor_subsystem_registration_order_drives_lifecycle_order() -> Result<()> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut builder = EngineBuilder::new();

    {
        let events = Arc::clone(&events);
        builder.add_editor_subsystem(move |_ctx| Ok(FirstEditorSubsystem { events }));
    }
    {
        let events = Arc::clone(&events);
        builder.add_editor_subsystem(move |_ctx| Ok(SecondEditorSubsystem { events }));
    }

    let mut engine = builder.build()?;
    engine.start()?;
    engine.tick()?;
    engine.shutdown()?;

    assert_eq!(
        &*events.lock(),
        &[
            "first-start",
            "second-start",
            "first-tick",
            "second-tick",
            "second-shutdown",
            "first-shutdown",
        ]
    );
    Ok(())
}

#[test]
fn duplicate_editor_subsystem_registration_is_skipped() -> Result<()> {
    let builds = Arc::new(AtomicUsize::new(0));
    let mut builder = EngineBuilder::new();

    {
        let builds = Arc::clone(&builds);
        builder.add_editor_subsystem(move |_ctx| {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok(DuplicateEditorSubsystem)
        });
    }
    {
        let builds = Arc::clone(&builds);
        builder.add_editor_subsystem(move |_ctx| {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok(DuplicateEditorSubsystem)
        });
    }

    let _engine = builder.build()?;

    assert_eq!(builds.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn editor_lifecycle_errors_include_subsystem_names() -> Result<()> {
    let mut builder = EngineBuilder::new();
    builder.add_editor_subsystem(|_ctx| Ok(StartFailingEditorSubsystem));
    let mut engine = builder.build()?;

    let result = engine.start();

    assert!(matches!(
        result,
        Err(Error::EditorSubsystemFailedStart {
            name: "start-failing-editor",
            ..
        })
    ));
    Ok(())
}

#[test]
fn editor_services_are_published_as_an_engine_resource() -> Result<()> {
    let observed = Arc::new(Mutex::new(None));
    let observed_services = Arc::clone(&observed);
    let mut builder = EngineBuilder::new();

    builder.add_editor_subsystem(move |ctx| {
        *observed_services.lock() = Some(ctx.engine.resource::<EditorServices>()?);
        Ok(DuplicateEditorSubsystem)
    });

    let _engine = builder.build()?;

    assert!(observed.lock().is_some());
    Ok(())
}

#[test]
fn window_ids_are_stable_and_increasing() {
    let services = EditorServices::new();

    let first = services.add_window_fn(descriptor("first", false), |_ui, _context| Ok(()));
    let second = services.add_window_fn(descriptor("second", false), |_ui, _context| Ok(()));

    assert_eq!(first.raw(), 0);
    assert_eq!(second.raw(), 1);
}

#[test]
fn window_default_open_state_is_honored() {
    let services = EditorServices::new();

    let open = services.add_window_fn(descriptor("open", true), |_ui, _context| Ok(()));
    let closed = services.add_window_fn(descriptor("closed", false), |_ui, _context| Ok(()));

    assert_eq!(services.is_open(open), Some(true));
    assert_eq!(services.is_open(closed), Some(false));
}

#[test]
fn default_open_windows_are_inserted_into_dock_state() {
    let services = EditorServices::new();

    let first = services.add_window_fn(descriptor("first", true), |_ui, _context| Ok(()));
    let second = services.add_window_fn(descriptor("second", true), |_ui, _context| Ok(()));
    let closed = services.add_window_fn(descriptor("closed", false), |_ui, _context| Ok(()));

    assert!(services.dock_contains_window_for_tests(first));
    assert!(services.dock_contains_window_for_tests(second));
    assert!(!services.dock_contains_window_for_tests(closed));
    assert_eq!(services.dock_tab_count_for_tests(), 2);
}

#[test]
fn default_open_windows_registered_after_bootstrap_preserve_existing_dock_layout() {
    let services = EditorServices::new();

    let viewport = services.add_window_fn(
        EditorWindowDescriptor {
            title: "viewport".to_owned(),
            category: crate::editor::VIEWPORT_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    let universe = services.add_window_fn(
        EditorWindowDescriptor {
            title: "universe".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    services.bootstrap_default_dock_layout_for_tests();
    services.move_window_to_first_dock_leaf_for_tests(universe);

    assert!(services.dock_windows_share_leaf_for_tests(viewport, universe));

    let late = services.add_window_fn(
        EditorWindowDescriptor {
            title: "late universe".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );

    assert!(services.dock_contains_window_for_tests(late));
    assert!(services.dock_windows_share_leaf_for_tests(viewport, universe));
    assert!(services.dock_windows_share_leaf_for_tests(universe, late));
}

#[test]
fn open_windows_render_through_dock_tabs() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let calls = Arc::new(Mutex::new(Vec::new()));

    let calls_for_window = Arc::clone(&calls);
    services.add_window_fn(descriptor("window", true), move |_ui, _context| {
        calls_for_window.lock().push("window");
        Ok(())
    });

    let universe = Universe::builder().build();
    render_services(&services, &universe)?;

    assert_eq!(*calls.lock(), vec!["window"]);
    Ok(())
}

#[test]
fn window_callback_can_query_and_mutate_services() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let callback_services = services.clone();
    let observed = Arc::new(Mutex::new(None));
    let callback_observed = Arc::clone(&observed);
    services.add_window_fn(descriptor("source", true), move |_ui, _context| {
        *callback_observed.lock() = Some(callback_services.window_count());
        callback_services.add_window_fn(descriptor("added", false), |_ui, _context| Ok(()));
        Ok(())
    });

    render_services(&services, &Universe::builder().build())?;

    assert_eq!(*observed.lock(), Some(1));
    assert_eq!(services.window_count(), 2);
    Ok(())
}

#[test]
fn menu_callback_can_query_services() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let callback_services = services.clone();
    let observed = Arc::new(Mutex::new(None));
    let callback_observed = Arc::clone(&observed);
    services.add_menu_fn(
        EditorMenuDescriptor {
            title: "query".to_owned(),
        },
        move |_ui, _context, _editor| {
            *callback_observed.lock() = Some(callback_services.menu_count());
            Ok(())
        },
    );

    let ctx = egui::Context::default();
    ctx.begin_pass(egui::RawInput::default());
    let handle = build_context().handle().clone();
    let frame = EditorRenderContext::new(0.016, &handle);
    let mut result = Ok(());
    egui::CentralPanel::default().show(&ctx, |ui| {
        result = services.render_menu_for_tests("query", ui, &frame, &Universe::builder().build());
    });
    let _ = ctx.end_pass();
    result?;
    assert_eq!(*observed.lock(), Some(1));
    Ok(())
}

#[test]
fn editor_shutdown_attempts_all_callbacks_after_failure() {
    struct ShutdownRecorder {
        name: &'static str,
        fail: bool,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl EditorSubsystem for ShutdownRecorder {
        fn name(&self) -> &'static str {
            self.name
        }

        fn shutdown(
            &mut self,
            _engine: &crate::EngineHandle,
            _editor: &EditorServices,
        ) -> anyhow::Result<()> {
            self.events.lock().push(self.name);
            anyhow::ensure!(!self.fail, "shutdown failed");
            Ok(())
        }
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let subsystems: Vec<Box<dyn EditorSubsystem>> = [("first", false), ("second", true)]
        .into_iter()
        .map(|(name, fail)| {
            Box::new(ShutdownRecorder {
                name,
                fail,
                events: Arc::clone(&events),
            }) as Box<dyn EditorSubsystem>
        })
        .collect();
    let mut editor = EditorRuntime::new(EditorServices::new(), subsystems);
    let handle = build_context().handle().clone();

    assert!(matches!(
        editor.shutdown(&handle),
        Err(Error::EditorSubsystemFailedShutdown { name: "second", .. })
    ));
    assert_eq!(*events.lock(), vec!["second", "first"]);
}

#[test]
fn same_title_windows_have_distinct_widget_ids() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let ids = Arc::new(Mutex::new(Vec::new()));
    for category in ["left", "right"] {
        let ids = Arc::clone(&ids);
        services.add_window_fn(
            EditorWindowDescriptor {
                title: "same title".to_owned(),
                category: category.to_owned(),
                default_open: true,
                show_in_list: true,
            },
            move |ui, _context| {
                ids.lock().push(ui.id());
                Ok(())
            },
        );
    }

    render_services(&services, &Universe::builder().build())?;

    let ids = ids.lock();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    Ok(())
}

#[test]
fn adding_new_category_with_only_floating_tabs_keeps_main_surface_valid() {
    let services = EditorServices::new();
    let first = services.add_window_fn(
        EditorWindowDescriptor {
            category: "first".to_owned(),
            ..descriptor("floating", true)
        },
        |_ui, _context| Ok(()),
    );
    {
        let mut state = services.state.lock();
        let tab = state.dock_state.find_tab(&first).expect("first tab exists");
        state.dock_state.remove_tab(tab);
        state.dock_state.add_window(vec![first]);
        assert!(state.dock_state.main_surface().is_empty());
    }

    let second = services.add_window_fn(
        EditorWindowDescriptor {
            category: "second".to_owned(),
            ..descriptor("main", true)
        },
        |_ui, _context| Ok(()),
    );

    assert!(services.dock_contains_window_for_tests(first));
    assert!(services.dock_contains_window_for_tests(second));
    assert!(!services.state.lock().dock_state.main_surface().is_empty());
}

#[test]
fn editor_style_is_applied_before_registered_capabilities_render() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_window_fill = Arc::new(Mutex::new(None));
    let styled_window_fill = egui::Color32::from_rgb(0x4a, 0x12, 0x7f);

    {
        let calls = Arc::clone(&calls);
        services.set_style(EditorStyle::new(move |ctx| {
            calls.fetch_add(1, Ordering::Relaxed);
            ctx.style_mut(|style| {
                style.visuals.window_fill = styled_window_fill;
            });
        }));
    }
    {
        let observed_window_fill = Arc::clone(&observed_window_fill);
        services.add_window_fn(descriptor("styled", true), move |ui, _context| {
            *observed_window_fill.lock() = Some(ui.visuals().window_fill);
            Ok(())
        });
    }

    let universe = Universe::builder().build();
    render_services(&services, &universe)?;

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(*observed_window_fill.lock(), Some(styled_window_fill));
    Ok(())
}

#[test]
fn editor_styles_stack_in_registration_order() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let observed_window_fill = Arc::new(Mutex::new(None));
    let first_window_fill = egui::Color32::from_rgb(0x10, 0x20, 0x30);
    let second_window_fill = egui::Color32::from_rgb(0x40, 0x50, 0x60);

    {
        let calls = Arc::clone(&calls);
        services.add_style(EditorStyle::new(move |ctx| {
            calls.lock().push("first");
            ctx.style_mut(|style| {
                style.visuals.window_fill = first_window_fill;
            });
        }));
    }
    {
        let calls = Arc::clone(&calls);
        services.add_style(EditorStyle::new(move |ctx| {
            calls.lock().push("second");
            ctx.style_mut(|style| {
                style.visuals.window_fill = second_window_fill;
            });
        }));
    }
    {
        let observed_window_fill = Arc::clone(&observed_window_fill);
        services.add_window_fn(descriptor("stacked", true), move |ui, _context| {
            *observed_window_fill.lock() = Some(ui.visuals().window_fill);
            Ok(())
        });
    }

    let universe = Universe::builder().build();
    render_services(&services, &universe)?;

    assert_eq!(*calls.lock(), vec!["first", "second"]);
    assert_eq!(*observed_window_fill.lock(), Some(second_window_fill));
    Ok(())
}

#[test]
fn editor_style_can_be_cleared() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let calls = Arc::new(AtomicUsize::new(0));

    {
        let calls = Arc::clone(&calls);
        services.set_style(EditorStyle::new(move |_ctx| {
            calls.fetch_add(1, Ordering::Relaxed);
        }));
    }
    services.clear_style();

    let universe = Universe::builder().build();
    render_services(&services, &universe)?;

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn closing_a_tab_sets_window_open_state_false() {
    let services = EditorServices::new();
    let id = services.add_window_fn(descriptor("window", true), |_ui, _context| Ok(()));

    services.close_window_tab_for_tests(id);

    assert_eq!(services.is_open(id), Some(false));
    assert!(!services.dock_contains_window_for_tests(id));
}

#[test]
fn opening_a_closed_window_readds_its_tab() {
    let services = EditorServices::new();
    let id = services.add_window_fn(descriptor("window", true), |_ui, _context| Ok(()));
    services.close_window_tab_for_tests(id);

    services.open_window_for_tests(id);

    assert_eq!(services.is_open(id), Some(true));
    assert!(services.dock_contains_window_for_tests(id));
}

#[test]
fn all_windows_are_closeable() {
    let services = EditorServices::new();
    let id = services.add_window_fn(descriptor("window", false), |_ui, _context| Ok(()));

    assert!(services.window_is_closeable_for_tests(id));
}

#[test]
fn window_render_errors_include_window_title() {
    let services = EditorServices::new();
    services.add_window_fn(descriptor("error", true), |_ui, _context| {
        Err(anyhow::anyhow!("window failed"))
    });

    let universe = Universe::builder().build();
    let err = render_services(&services, &universe).expect_err("render should fail");

    assert!(err.to_string().contains("window `error`"));
}

#[test]
fn windows_expose_registered_window_metadata_and_open_state() {
    let services = EditorServices::new();

    let first = services.add_window_fn(
        EditorWindowDescriptor {
            title: "zeta".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    let second = services.add_window_fn(
        EditorWindowDescriptor {
            title: "alpha".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            default_open: false,
            show_in_list: false,
        },
        |_ui, _context| Ok(()),
    );

    assert_eq!(
        services.windows(),
        vec![
            EditorWindowInfo {
                id: first,
                title: "zeta".to_owned(),
                category: UNIVERSE_CATEGORY.to_owned(),
                open: true,
                show_in_list: true,
            },
            EditorWindowInfo {
                id: second,
                title: "alpha".to_owned(),
                category: EDITOR_CATEGORY.to_owned(),
                open: false,
                show_in_list: false,
            },
        ]
    );
    assert_eq!(
        services.window(second),
        Some(EditorWindowInfo {
            id: second,
            title: "alpha".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            open: false,
            show_in_list: false,
        })
    );
}

#[test]
fn editor_open_window_command_reopens_closed_windows() {
    let services = EditorServices::new();
    let id = services.add_window_fn(descriptor("window", false), |_ui, _context| Ok(()));

    services.open_window_for_tests(id);

    assert_eq!(services.is_open(id), Some(true));
    assert!(services.dock_contains_window_for_tests(id));
}

#[test]
fn windows_can_request_other_windows_to_open_during_render() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let target = services.add_window_fn(descriptor("target", false), |_ui, _context| Ok(()));
    services.add_window_fn(descriptor("source", true), move |_ui, context| {
        context.open_window(target);
        Ok(())
    });

    let universe = Universe::builder().build();
    render_services(&services, &universe)?;

    assert_eq!(services.is_open(target), Some(true));
    Ok(())
}

#[test]
fn menu_capabilities_are_registered_in_registration_order() {
    let services = EditorServices::new();

    for label in ["first", "second", "third"] {
        services.add_menu_fn(
            EditorMenuDescriptor {
                title: label.to_owned(),
            },
            move |_ui, _context, editor| {
                let _ = editor.windows();
                Ok(())
            },
        );
    }

    assert_eq!(services.menu_titles(), vec!["first", "second", "third"]);
}

#[test]
fn menu_commands_still_open_windows() -> anyhow::Result<()> {
    let services = EditorServices::new();
    let target = services.add_window_fn(descriptor("target", false), |_ui, _context| Ok(()));
    services.add_menu_fn(
        EditorMenuDescriptor {
            title: "Open".to_owned(),
        },
        move |_ui, _context, editor| {
            editor.open_window(target);
            Ok(())
        },
    );

    let universe = Universe::builder().build();
    let ctx = egui::Context::default();
    ctx.begin_pass(egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        ..egui::RawInput::default()
    });
    let handle = build_context().handle().clone();
    let frame = EditorRenderContext::new(0.016, &handle);
    let mut result = Ok(());
    egui::CentralPanel::default().show(&ctx, |ui| {
        result = services.render_menu_for_tests("Open", ui, &frame, &universe);
    });
    let _ = ctx.end_pass();
    result?;

    assert_eq!(services.is_open(target), Some(true));
    assert!(services.dock_contains_window_for_tests(target));
    Ok(())
}
