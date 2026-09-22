//! Marks renderer-relevant IDs from the ordered universe change log.
//! The renderer extracts final state once after the universe tick.

use crate::render_commands::RenderChanges;
use dirk_player::PlayerId;
use dirk_universe::{
    changes::Change,
    systems::{Changes, System},
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

impl System<Changes<'_>> for RendererSystem {
    fn run(&mut self, changes: Changes<'_>) {
        for change in changes.iter() {
            match change {
                Change::WorldCreated { world } | Change::WorldDestroyed { world } => {
                    self.changes.world(*world);
                }
                Change::EntitySpawned { entity, .. }
                | Change::EntityMoved { entity, .. }
                | Change::EntityDespawned { entity, .. } => {
                    self.changes.entity(*entity);
                }
                Change::ComponentAdded { entity, .. }
                | Change::ComponentUpdated { entity, .. }
                | Change::ComponentRemoved { entity, .. } => {
                    if change.component::<Renderable>().is_some()
                        || change.component::<Transform>().is_some()
                        || change.component::<PlayerId>().is_some()
                    {
                        self.changes.entity(*entity);
                    }
                }
            }
        }
    }
}
