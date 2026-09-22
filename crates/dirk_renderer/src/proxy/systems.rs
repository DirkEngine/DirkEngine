//! Marks renderer-relevant IDs from the deltas of worlds, entity locations and
//! rendered components. The renderer extracts final state once after the
//! universe tick.

use crate::render_commands::RenderChanges;
use dirk_player::PlayerId;
use dirk_universe::{
    Entity, InWorld, World, WorldId,
    query::{Delta, Query},
    systems::{PerWorld, System},
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
        for (world, _) in &worlds {
            self.changes.world(WorldId::from_entity(world));
        }
        for (_, (locations, meshes, transforms, players)) in &entities {
            let changed = locations
                .into_iter()
                .map(|(entity, _)| entity)
                .chain(meshes.into_iter().map(|(entity, _)| entity))
                .chain(transforms.into_iter().map(|(entity, _)| entity))
                .chain(players.into_iter().map(|(entity, _)| entity));
            for entity in changed {
                self.changes.entity(entity);
            }
        }
    }
}
