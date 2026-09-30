//! Synchronizes renderer state from the deltas of worlds, entity locations
//! and rendered components.

use std::collections::HashSet;

use crate::{Error, render_commands::RenderCommandSender};
use dirk_player::PlayerId;
use dirk_universe::{
    Entity, InWorld, World, WorldId,
    query::{Delta, Query},
    systems::{PerWorld, System},
};
use dirk_world::components::{Renderable, Transform};

pub struct RendererSystem {
    sender: RenderCommandSender,
    /// Entities with a proxy, so placing one distinguishes spawns from moves.
    proxies: HashSet<Entity>,
}

impl RendererSystem {
    pub fn new(sender: RenderCommandSender) -> Self {
        Self {
            sender,
            proxies: HashSet::new(),
        }
    }

    fn place(&mut self, entity: Entity, world: WorldId) {
        if self.proxies.insert(entity) {
            self.sender.enqueue_command(move |renderer| {
                renderer.scene_manager.create_proxy(entity, world)?;
                Ok(())
            });
        } else {
            self.sender.enqueue_command(move |renderer| {
                renderer.scene_manager.send_proxy(entity, world)?;
                renderer.update_viewport_world_for_camera(entity, world);
                Ok(())
            });
        }
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

impl System for RendererSystem {
    // Every world renders, isolated ones included, so entities are read per world.
    type Params<'u> = (
        Query<'u, (Entity, Delta<'u, World>)>,
        PerWorld<
            'u,
            (
                Query<'u, (Entity, Delta<'u, InWorld>)>,
                Query<'u, (Entity, Delta<'u, Renderable>)>,
                Query<'u, (Entity, Delta<'u, Transform>)>,
                Query<'u, (Entity, Delta<'u, PlayerId>)>,
            ),
        >,
    );

    fn run(&mut self, (worlds, entities): Self::Params<'_>) {
        // Scenes exist before their proxies, and proxies are destroyed before
        // their scenes. Component updates apply to proxies that remain.
        let mut destroyed_worlds = Vec::new();
        for (entity, delta) in &worlds {
            match delta {
                Delta::Set(world) => {
                    let world = world.id();
                    self.sender.enqueue_command(move |renderer| {
                        renderer.scene_manager.create_scene(world)?;
                        Ok(())
                    });
                }
                Delta::Removed => destroyed_worlds.push(WorldId::from_entity(entity)),
            }
        }

        let mut despawned = HashSet::new();
        for (_, (locations, meshes, transforms, players)) in &entities {
            for (entity, delta) in locations {
                match delta {
                    Delta::Set(in_world) => self.place(entity, in_world.0),
                    Delta::Removed => {
                        despawned.insert(entity);
                    }
                }
            }
            for (entity, delta) in meshes.iter().filter(|(e, _)| !despawned.contains(e)) {
                self.mesh(entity, delta.value());
            }
            for (entity, delta) in transforms.iter().filter(|(e, _)| !despawned.contains(e)) {
                self.transform(entity, delta.value());
            }
            for (entity, delta) in players.iter().filter(|(e, _)| !despawned.contains(e)) {
                self.player(entity, delta.value().copied());
            }
        }

        for entity in despawned {
            self.proxies.remove(&entity);
            self.sender.enqueue_command(move |renderer| {
                renderer.clear_viewports_for_camera(entity);
                renderer.scene_manager.destroy_proxy(entity)?;
                Ok(())
            });
        }
        for world in destroyed_worlds {
            self.sender.enqueue_command(move |renderer| {
                renderer.scene_manager.destroy_scene(world);
                Ok(())
            });
        }
    }
}
