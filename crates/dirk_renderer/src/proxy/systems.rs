//! Replays ordered ECS changes into the renderer through one command channel.

use crate::{Error, render_commands::RenderCommandSender};
use dirk_player::PlayerId;
use dirk_universe::{
    changes::{Change, ComponentChange},
    systems::{Changes, System},
};
use dirk_world::components::{Renderable, Transform};

pub struct RendererSystem {
    sender: RenderCommandSender,
}

impl RendererSystem {
    pub fn new(sender: RenderCommandSender) -> Self {
        Self { sender }
    }

    fn mesh(&self, change: &ComponentChange<'_, Renderable>) {
        let (entity, model) = match *change {
            ComponentChange::Added { entity, component }
            | ComponentChange::Updated {
                entity,
                new: component,
                ..
            } => (entity, Some(component.model.clone())),
            ComponentChange::Removed { entity, .. } => (entity, None),
        };
        self.sender.enqueue_command(move |renderer| {
            let proxy = renderer
                .scene_manager
                .get_proxy_mut(entity)
                .ok_or(Error::EntityDoesNotExist(entity))?;
            proxy.set_model(model);
            Ok(())
        });
    }

    fn transform(&self, change: &ComponentChange<'_, Transform>) {
        let (entity, model, view) = match *change {
            ComponentChange::Added { entity, component }
            | ComponentChange::Updated {
                entity,
                new: component,
                ..
            } => (entity, Some(component.matrix()), Some(component.view())),
            ComponentChange::Removed { entity, .. } => (entity, None, None),
        };
        self.sender.enqueue_command(move |renderer| {
            let proxy = renderer
                .scene_manager
                .get_proxy_mut(entity)
                .ok_or(Error::EntityDoesNotExist(entity))?;
            proxy.set_model_matrix(model);
            proxy.set_view(view);
            Ok(())
        });
    }

    fn player(&self, change: &ComponentChange<'_, PlayerId>) {
        let (entity, old, new) = match *change {
            ComponentChange::Added { entity, component } => (entity, None, Some(*component)),
            ComponentChange::Updated { entity, old, new } => (entity, Some(*old), Some(*new)),
            ComponentChange::Removed { entity, component } => (entity, Some(*component), None),
        };
        self.sender.enqueue_command(move |renderer| {
            if let Some(old) = old
                && let Some(viewport) = renderer.viewports.get_mut(&old)
                && viewport.camera == Some(entity)
            {
                viewport.camera = None;
                viewport.world = None;
            }
            if let Some(new) = new {
                renderer.bind_viewport_to_entity(new, entity);
            }
            Ok(())
        });
    }
}

impl System<Changes<'_>> for RendererSystem {
    fn run(&mut self, changes: Changes<'_>) {
        // One ordered stream ensures that scenes/proxies exist before updates,
        // and component cleanup precedes proxy/scene destruction.
        for change in changes.iter() {
            match *change {
                Change::WorldCreated { world } => self.sender.enqueue_command(move |renderer| {
                    renderer.scene_manager.create_scene(world)?;
                    Ok(())
                }),
                Change::WorldDestroyed { world } => self.sender.enqueue_command(move |renderer| {
                    renderer.scene_manager.destroy_scene(world);
                    Ok(())
                }),
                Change::EntitySpawned { entity, world } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.create_proxy(entity, world)?;
                        Ok(())
                    });
                }
                Change::EntityMoved { entity, to, .. } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.send_proxy(entity, to)?;
                        renderer.update_viewport_world_for_camera(entity, to);
                        Ok(())
                    });
                }
                Change::EntityDespawned { entity, .. } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.clear_viewports_for_camera(entity);
                        renderer.scene_manager.destroy_proxy(entity)?;
                        Ok(())
                    });
                }
                _ => {
                    if let Some(change) = change.component::<Renderable>() {
                        self.mesh(&change);
                    }
                    if let Some(change) = change.component::<Transform>() {
                        self.transform(&change);
                    }
                    if let Some(change) = change.component::<PlayerId>() {
                        self.player(&change);
                    }
                }
            }
        }
    }
}
