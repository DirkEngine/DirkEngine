use std::sync::{Arc, OnceLock};

use dirk_engine::editor::{EDITOR_CATEGORY, EditorStyle, EditorSubsystem as _, UNIVERSE_CATEGORY};

use crate::style::{EditorPalette, default_editor_style};

fn egui_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        ..egui::RawInput::default()
    }
}

fn test_handle() -> dirk_engine::EngineHandle {
    static HANDLE: OnceLock<dirk_engine::EngineHandle> = OnceLock::new();

    HANDLE
        .get_or_init(|| {
            struct HandleCapture;

            impl dirk_engine::Subsystem for HandleCapture {
                fn name(&self) -> &'static str {
                    "handle-capture"
                }
            }

            let captured = Arc::new(parking_lot::Mutex::new(None));
            let observed = Arc::clone(&captured);
            let mut builder = dirk_engine::Engine::builder();
            builder.add_subsystem(move |ctx| {
                *observed.lock() = Some(ctx.handle().clone());
                Ok(HandleCapture)
            });
            let _engine = builder.build().expect("engine should build");
            captured
                .lock()
                .clone()
                .expect("handle should be captured while building")
        })
        .clone()
}

#[test]
fn default_editor_style_applies_dark_editor_visuals() {
    let ctx = egui::Context::default();
    ctx.begin_pass(egui_input());

    default_editor_style().apply(&ctx);

    let palette = EditorPalette::default();
    let style = ctx.global_style();
    assert_eq!(ctx.theme(), egui::Theme::Dark);
    assert_eq!(style.visuals.window_fill, palette.surface);
    assert_eq!(style.visuals.panel_fill, palette.background);
    assert_eq!(style.visuals.selection.bg_fill, palette.selection);
    assert_eq!(style.visuals.widgets.active.bg_fill, palette.control_active);
    assert_eq!(style.spacing.interact_size, egui::vec2(22.0, 18.0));
    assert_eq!(
        style.visuals.window_corner_radius,
        egui::CornerRadius::same(2)
    );

    ctx.end_pass().drop_without_applying_deltas();
}

#[test]
fn editor_palette_converts_to_editor_style() {
    let ctx = egui::Context::default();
    ctx.begin_pass(egui_input());

    let palette = EditorPalette {
        surface: egui::Color32::from_rgb(0x32, 0x10, 0x44),
        ..EditorPalette::default()
    };
    let style: EditorStyle = palette.into();
    style.apply(&ctx);

    assert_eq!(ctx.global_style().visuals.window_fill, palette.surface);

    ctx.end_pass().drop_without_applying_deltas();
}

#[test]
fn builtin_editor_subsystem_registers_expected_default_capabilities() -> anyhow::Result<()> {
    let services = crate::EditorServices::new();
    let handle = test_handle();
    let universe = dirk_universe::Universe::builder().build();
    let mut subsystem = crate::BuiltinEditorSubsystem;

    subsystem.start(&handle, &services)?;

    assert_eq!(services.menu_titles(), vec!["Main", "Settings", "Windows"]);
    assert_eq!(
        services.window_titles(),
        vec!["Settings", "Engine", "Worlds", "Entities", "Entity Details",]
    );
    let open_windows = services
        .windows()
        .into_iter()
        .filter(|window| window.open)
        .map(|window| window.title)
        .collect::<Vec<_>>();
    assert_eq!(open_windows, vec!["Engine", "Entities", "Entity Details"]);

    let ctx = egui::Context::default();
    let frame = dirk_engine::editor::EditorRenderContext::new(0.016, &handle);
    services
        .render_ui(&ctx, egui_input(), &frame, &universe)?
        .drop_without_applying_deltas();

    let palette = EditorPalette::default();
    let style = ctx.global_style();
    assert_eq!(style.visuals.window_fill, palette.surface);
    assert_eq!(style.visuals.panel_fill, palette.background);
    assert_eq!(style.visuals.selection.bg_fill, palette.selection);
    assert_eq!(style.visuals.widgets.active.bg_fill, palette.control_active);
    assert_eq!(style.spacing.window_margin, egui::Margin::symmetric(6, 5));

    Ok(())
}

#[test]
fn window_list_menu_groups_categories_and_windows_alphabetically() {
    let services = crate::EditorServices::new();

    services.add_window_fn(
        crate::EditorWindowDescriptor {
            title: "zeta".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    services.add_window_fn(
        crate::EditorWindowDescriptor {
            title: "alpha".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    services.add_window_fn(
        crate::EditorWindowDescriptor {
            title: "beta".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    services.add_window_fn(
        crate::EditorWindowDescriptor {
            title: "alpha".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        |_ui, _context| Ok(()),
    );
    services.add_window_fn(
        crate::EditorWindowDescriptor {
            title: "hidden".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            default_open: false,
            show_in_list: false,
        },
        |_ui, _context| Ok(()),
    );

    let grouped_titles = crate::grouped_window_menu_entries(&services.windows())
        .into_iter()
        .map(|(category, windows)| {
            (
                category,
                windows
                    .into_iter()
                    .map(|window| window.title)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        grouped_titles,
        vec![
            (
                EDITOR_CATEGORY.to_owned(),
                vec!["alpha".to_owned(), "beta".to_owned()],
            ),
            (
                UNIVERSE_CATEGORY.to_owned(),
                vec!["alpha".to_owned(), "zeta".to_owned()],
            ),
        ]
    );
}

#[test]
fn editor_plugin_registers_builtin_editor_subsystem() -> anyhow::Result<()> {
    struct Observer;

    impl dirk_engine::editor::EditorSubsystem for Observer {
        fn name(&self) -> &'static str {
            "observer"
        }
    }

    let observed = Arc::new(parking_lot::Mutex::new(None));
    let observed_services = Arc::clone(&observed);
    let mut builder = dirk_engine::Engine::builder();

    builder.with_plugin(crate::EditorPlugin)?;
    builder.add_editor_subsystem(move |ctx| {
        *observed_services.lock() = Some(ctx.engine.resource::<crate::EditorServices>()?);
        Ok(Observer)
    });

    let mut engine = builder.build()?;
    engine.start()?;

    let services = observed
        .lock()
        .clone()
        .expect("observer should capture editor services");

    assert!(services.window_titles().contains(&"Engine".to_owned()));
    Ok(())
}
