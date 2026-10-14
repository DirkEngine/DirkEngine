//! Synchronizes renderer state from current components and structural IDs.

use std::any::TypeId;

use crate::{Error, render_commands::RenderCommandSender};
use dirk_player::PlayerId;
use dirk_universe::{
    Entity,
    lifecycle::LifecycleEvent,
    query::{Query, filter::Changed},
    systems::{Lifecycle, System},
};
use dirk_world::components::{Renderable, Transform};

pub struct RendererSystem {
    sender: RenderCommandSender,
}

impl RendererSystem {
    pub fn new(sender: RenderCommandSender) -> Self {
        Self { sender }
    }

    fn mesh(&self, entity: Entity, component: Option<&Renderable>) {
        let model = component.map(|component| component.model.clone());
        self.sender.enqueue_command(move |renderer| {
            let proxy = renderer
                .scene_manager
                .get_proxy_mut(entity)
                .ok_or(Error::EntityDoesNotExist(entity))?;
            proxy.set_model(model);
            Ok(())
        });
    }

    fn transform(&self, entity: Entity, component: Option<&Transform>) {
        let model = component.map(Transform::matrix);
        let view = component.map(Transform::view);
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

    fn player(&self, entity: Entity, player: Option<PlayerId>) {
        self.sender.enqueue_command(move |renderer| {
            // The renderer already owns the previous binding; ECS snapshots
            // are unnecessary when a PlayerId changes or is removed.
            renderer.clear_viewports_for_camera(entity);
            if let Some(player) = player {
                renderer.bind_viewport_to_entity(player, entity);
            }
            Ok(())
        });
    }
}

type RendererParams<'u> = (
    Lifecycle<'u>,
    Query<'u, (Entity, &'u Renderable), Changed<Renderable>>,
    Query<'u, (Entity, &'u Transform), Changed<Transform>>,
    Query<'u, (Entity, &'u PlayerId), Changed<PlayerId>>,
);

impl System<RendererParams<'_>> for RendererSystem {
    fn run(&mut self, (lifecycle, meshes, transforms, players): RendererParams<'_>) {
        // Replay structure before uploading current values. This handles
        // transient entities and component removal/reinsertion in one tick.
        for event in lifecycle.iter() {
            match *event {
                LifecycleEvent::WorldCreated { world } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.create_scene(world)?;
                        Ok(())
                    });
                }
                LifecycleEvent::WorldDestroyed { world } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.destroy_scene(world);
                        Ok(())
                    });
                }
                LifecycleEvent::EntitySpawned { entity, world } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.create_proxy(entity, world)?;
                        Ok(())
                    });
                }
                LifecycleEvent::EntityMoved { entity, to, .. } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.send_proxy(entity, to)?;
                        renderer.update_viewport_world_for_camera(entity, to);
                        Ok(())
                    });
                }
                LifecycleEvent::EntityDespawned { entity, .. } => {
                    self.sender.enqueue_command(move |renderer| {
                        renderer.clear_viewports_for_camera(entity);
                        renderer.scene_manager.destroy_proxy(entity)?;
                        Ok(())
                    });
                }
                LifecycleEvent::ComponentRemoved { entity, type_id } => {
                    if type_id == TypeId::of::<Renderable>() {
                        self.mesh(entity, None);
                    } else if type_id == TypeId::of::<Transform>() {
                        self.transform(entity, None);
                    } else if type_id == TypeId::of::<PlayerId>() {
                        self.player(entity, None);
                    }
                }
            }
        }
        for (entity, component) in &meshes {
            self.mesh(entity, Some(&component));
        }
        for (entity, component) in &transforms {
            self.transform(entity, Some(&component));
        }
        for (entity, player) in &players {
            self.player(entity, Some(*player));
        }
    }
}
