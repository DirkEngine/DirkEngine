//! Default editor feature package for `DirkEngine`.
//!
//! This also contains loads of utility functions for working with the Editor.

use std::collections::BTreeMap;

use dirk_universe::Universe;

use dirk_engine::{
    EngineHandle,
    editor::{
        EDITOR_CATEGORY, EditorMenu, EditorMenuContext, EditorMenuDescriptor, EditorServices,
        EditorUiContext, EditorWindow, EditorWindowDescriptor, EditorWindowId, EditorWindowInfo,
        UNIVERSE_CATEGORY,
    },
};

mod settings;
pub mod style;
mod universe;

#[cfg(test)]
mod tests;

/// Registers the built-in editor subsystem package with the engine.
pub struct EditorPlugin;

impl dirk_engine::EnginePlugin for EditorPlugin {
    fn name(&self) -> &'static str {
        "editor"
    }

    fn build(&self, builder: &mut dirk_engine::EngineBuilder) -> anyhow::Result<()> {
        builder.add_editor_subsystem(|ctx| {
            ctx.engine
                .extend_universe(Universe::builder().with_journal(universe::HISTORY_CAPACITY));
            Ok(BuiltinEditorSubsystem)
        });
        Ok(())
    }
}

/// Built-in editor subsystem package.
struct BuiltinEditorSubsystem;

impl dirk_engine::editor::EditorSubsystem for BuiltinEditorSubsystem {
    fn name(&self) -> &'static str {
        "builtin-editor"
    }

    fn start(&mut self, _engine: &EngineHandle, services: &EditorServices) -> anyhow::Result<()> {
        services.add_style(style::default_editor_style());
        services.add_menu(MainMenu);

        settings::register_capabilities(services);
        services.add_menu(WindowListMenu);
        services.add_window(EngineDiagnosticsWindow);
        universe::register_capabilities(services);
        Ok(())
    }
}

struct MainMenu;

impl EditorMenu for MainMenu {
    fn descriptor(&self) -> EditorMenuDescriptor {
        EditorMenuDescriptor {
            title: "Main".to_owned(),
        }
    }

    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        context: &mut EditorUiContext<'_>,
        _editor: &mut EditorMenuContext<'_>,
    ) -> anyhow::Result<()> {
        if ui.button("Exit").clicked() {
            context.handle.exit();
            ui.close();
        }
        Ok(())
    }
}

/// Window list menu capability for the default editor package.
struct WindowListMenu;

impl EditorMenu for WindowListMenu {
    fn descriptor(&self) -> EditorMenuDescriptor {
        EditorMenuDescriptor {
            title: "Windows".to_owned(),
        }
    }

    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        _context: &mut EditorUiContext<'_>,
        editor: &mut EditorMenuContext<'_>,
    ) -> anyhow::Result<()> {
        for (category, windows) in grouped_window_menu_entries(editor.windows()) {
            ui.menu_button(category, |ui| {
                for window in windows {
                    if ui.button(&window.title).clicked() {
                        editor.open_window(window.id);
                        ui.close();
                    }
                }
            });
        }
        Ok(())
    }
}

fn grouped_window_menu_entries(
    entries: &[EditorWindowInfo],
) -> Vec<(String, Vec<EditorWindowInfo>)> {
    let mut grouped: BTreeMap<String, Vec<EditorWindowInfo>> = BTreeMap::new();
    for entry in entries {
        if !entry.show_in_list {
            continue;
        }
        grouped
            .entry(entry.category.clone())
            .or_default()
            .push(entry.clone());
    }
    for windows in grouped.values_mut() {
        windows.sort_by(|left, right| left.title.cmp(&right.title));
    }
    grouped.into_iter().collect()
}

/// Engine diagnostics window capability for the default editor package.
struct EngineDiagnosticsWindow;

impl EditorWindow for EngineDiagnosticsWindow {
    fn descriptor(&self) -> EditorWindowDescriptor {
        EditorWindowDescriptor {
            title: "Engine".to_owned(),
            category: EDITOR_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, context: &mut EditorUiContext<'_>) -> anyhow::Result<()> {
        let metadata = context.handle.metadata();

        ui.label(format!(
            "app: {} {}",
            metadata.app_name(),
            metadata.app_version()
        ));
        ui.label(format!(
            "engine: {} {}",
            metadata.engine_name(),
            metadata.engine_version()
        ));
        ui.label(format!("frame: {}", context.handle.frame()));
        ui.label(format!("status: {:?}", context.handle.status()));
        ui.label(format!("delta: {:.2} ms", context.delta_time() * 1_000.0));
        ui.label(format!("fps: {:.0}", 1.0 / context.delta_time()));
        Ok(())
    }
}
