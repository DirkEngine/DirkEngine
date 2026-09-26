//! Marks renderer-relevant IDs from lifecycle events and changed components.
//! The renderer extracts final state once after the universe tick.

use std::any::TypeId;

use crate::render_commands::RenderChanges;
use dirk_player::PlayerId;
use dirk_universe::{
    Entity,
    lifecycle::LifecycleEvent,
    query::{Query, filter::Changed},
    systems::{Lifecycle, System},
};
use dirk_world::components::{Renderable, Transform};

pub struct RendererSystem {
    changes: RenderChanges,
}

impl RendererSystem {
    pub fn new(changes: RenderChanges) -> Self {
        Self { changes }
    }
}

type RendererParams<'u> = (
    Lifecycle<'u>,
    Query<'u, Entity, Changed<Renderable>>,
    Query<'u, Entity, Changed<Transform>>,
    Query<'u, Entity, Changed<PlayerId>>,
);

impl System<RendererParams<'_>> for RendererSystem {
    fn run(&mut self, (lifecycle, meshes, transforms, players): RendererParams<'_>) {
        for event in lifecycle.iter() {
            match *event {
                LifecycleEvent::WorldCreated { world }
                | LifecycleEvent::WorldDestroyed { world } => {
                    self.changes.world(world);
                }
                LifecycleEvent::EntitySpawned { entity, .. }
                | LifecycleEvent::EntityMoved { entity, .. }
                | LifecycleEvent::EntityDespawned { entity, .. } => {
                    self.changes.entity(entity);
                }
                LifecycleEvent::ComponentRemoved { entity, type_id } => {
                    if type_id == TypeId::of::<Renderable>()
                        || type_id == TypeId::of::<Transform>()
                        || type_id == TypeId::of::<PlayerId>()
                    {
                        self.changes.entity(entity);
                    }
                }
            }
        }
        for entity in meshes.iter().chain(transforms.iter()).chain(players.iter()) {
            self.changes.entity(entity);
        }
    }
}
