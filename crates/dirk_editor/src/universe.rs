//! Editor windows for inspecting a [`dirk_universe::Universe`].

use std::sync::Arc;

use parking_lot::Mutex;

use crate::{EditorServices, EditorWindowDescriptor, UNIVERSE_CATEGORY};

use dirk_universe::{Entity, Universe, WorldId};

pub fn register_capabilities(services: &EditorServices) {
    let universe_windows = Arc::new(Mutex::new(UniverseWindows::default()));
    let entity_details_window = Arc::new(Mutex::new(None));

    {
        let universe_windows = Arc::clone(&universe_windows);
        services.add_window_fn(
            EditorWindowDescriptor {
                title: "Worlds".to_owned(),
                category: UNIVERSE_CATEGORY.to_owned(),
                default_open: false,
                show_in_list: true,
            },
            move |ui, context| {
                universe_windows
                    .lock()
                    .world_list_ui(ui, context.universe());
                Ok(())
            },
        );
    }

    {
        let entity_details_window = Arc::clone(&entity_details_window);
        let universe_windows = Arc::clone(&universe_windows);
        services.add_window_fn(
            EditorWindowDescriptor {
                title: "Entities".to_owned(),
                category: UNIVERSE_CATEGORY.to_owned(),
                default_open: true,
                show_in_list: true,
            },
            move |ui, context| {
                let entity_clicked = universe_windows
                    .lock()
                    .entity_list_ui(ui, context.universe());
                if entity_clicked && let Some(entity_details) = *entity_details_window.lock() {
                    context.open_window(entity_details);
                }
                Ok(())
            },
        );
    }

    let entity_details = services.add_window_fn(
        EditorWindowDescriptor {
            title: "Entity Details".to_owned(),
            category: UNIVERSE_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        },
        move |ui, context| {
            universe_windows
                .lock()
                .entity_details_ui(ui, context.universe());
            Ok(())
        },
    );
    *entity_details_window.lock() = Some(entity_details);
}

/// Shared state for Universe editor windows.
#[derive(Default)]
struct UniverseWindows {
    selected_world: Option<WorldId>,
    selected_entity: Option<Entity>,
}

impl UniverseWindows {
    /// Draws a deterministic list of worlds.
    fn world_list_ui(&mut self, ui: &mut egui::Ui, universe: &Universe) {
        let mut worlds: Vec<_> = universe.worlds().collect();
        worlds.sort_by_key(|world| world.id().raw());

        for world in worlds {
            let label = format!(
                "{}: {} ({} entities)",
                world.id().raw(),
                world.name(),
                world.entity_count()
            );
            if ui
                .selectable_label(self.selected_world == Some(world.id()), label)
                .clicked()
            {
                self.selected_world = Some(world.id());
            }
        }
    }

    /// Draws a deterministic list of entities.
    fn entity_list_ui(&mut self, ui: &mut egui::Ui, universe: &Universe) -> bool {
        let mut entity_clicked = false;

        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.selected_world.is_none(), "All Worlds")
                .clicked()
            {
                self.selected_world = None;
            }

            let mut worlds: Vec<_> = universe.worlds().collect();
            worlds.sort_by_key(|world| world.id().raw());
            for world in worlds {
                if ui
                    .selectable_label(
                        self.selected_world == Some(world.id()),
                        format!("World {}", world.id().raw()),
                    )
                    .clicked()
                {
                    self.selected_world = Some(world.id());
                }
            }
        });

        ui.separator();

        let mut entities: Vec<_> = match self.selected_world {
            Some(world) => universe
                .entities_in_world(world)
                .map(|entity| (entity, world))
                .collect(),
            None => universe.entities().collect(),
        };
        entities.sort_by_key(|(entity, _)| entity.raw());

        let row_height = ui.spacing().interact_size.y;
        egui::ScrollArea::vertical().show_rows(ui, row_height, entities.len(), |ui, visible| {
            for &(entity, world) in &entities[visible] {
                let label = format!("{} (world {})", entity.raw(), world.raw());
                if ui
                    .selectable_label(self.selected_entity == Some(entity), label)
                    .clicked()
                {
                    self.selected_entity = Some(entity);
                    self.selected_world = Some(world);
                    entity_clicked = true;
                }
            }
        });

        entity_clicked
    }

    /// Draws details for the selected entity.
    fn entity_details_ui(&mut self, ui: &mut egui::Ui, universe: &Universe) {
        let Some(entity) = self.selected_entity else {
            ui.label("No entity selected");
            return;
        };

        ui.label(format!("entity: {}", entity.raw()));

        let Some(world) = universe.get_world(entity) else {
            ui.label("entity is no longer live");
            return;
        };

        ui.label(format!("world: {}", world.raw()));
        ui.separator();

        let mut components: Vec<_> = universe.component_infos(entity).collect();
        components.sort_by_key(|component| component.type_name);

        for component in components {
            ui.collapsing(component.type_name, |ui| {
                ui.label(format!("type id: {:?}", component.type_id));
                ui.monospace(format!("{:#?}", component.debug));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirk_universe::World;

    #[test]
    fn entity_list_only_builds_visible_rows() {
        let mut world = World::builder("many entities");
        for _ in 0..5_000 {
            world = world.with_entity(Entity::builder());
        }
        let mut universe = Universe::builder().with_world(world).build();
        universe.tick(0.0);

        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..egui::RawInput::default()
        });
        egui::CentralPanel::default().show(&ctx, |ui| {
            UniverseWindows::default().entity_list_ui(ui, &universe);
        });
        let output = ctx.end_pass();

        assert!(output.shapes.len() < 500, "too many entity rows rendered");
    }
}
