//! Dirty notifications collapse into final-state deltas after the universe tick.
use dirk_universe::{Entity, Universe, WorldId};
use dirk_world::components::{Renderable, Transform};
use parking_lot::Mutex;
use std::{collections::HashSet, sync::Arc};

#[derive(Default)]
struct Dirty {
    entities: HashSet<Entity>,
    worlds: HashSet<WorldId>,
}
#[derive(Clone, Default)]
pub struct RenderChanges(Arc<Mutex<Dirty>>);
pub struct EntityData {
    pub world: WorldId,
    pub model: Option<dirk_assets::AssetHandle>,
    pub transform: Option<Transform>,
    pub player: Option<dirk_player::PlayerId>,
}
pub struct RenderDelta {
    pub entity: Entity,
    pub state: Option<EntityData>,
}
impl RenderChanges {
    pub fn entity(&self, entity: Entity) {
        self.0.lock().entities.insert(entity);
    }
    pub fn world(&self, world: WorldId) {
        self.0.lock().worlds.insert(world);
    }
    pub fn extract(
        &self,
        universe: &Universe,
        previous: impl Iterator<Item = (Entity, WorldId)>,
    ) -> Vec<RenderDelta> {
        let mut dirty = std::mem::take(&mut *self.0.lock());
        if !dirty.worlds.is_empty() {
            for (entity, world) in previous.chain(universe.entities()) {
                if dirty.worlds.contains(&world) {
                    dirty.entities.insert(entity);
                }
            }
        }
        dirty
            .entities
            .into_iter()
            .map(|entity| RenderDelta {
                entity,
                state: universe.get_world(entity).map(|world| EntityData {
                    world,
                    model: universe
                        .component::<Renderable>(entity)
                        .map(|r| r.model.clone()),
                    transform: universe.component::<Transform>(entity).cloned(),
                    player: universe.component::<dirk_player::PlayerId>(entity).copied(),
                }),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::systems::RendererSystem;
    use dirk_universe::World;

    #[test]
    fn extraction_coalesces_component_changes_moves_and_despawn() {
        let changes = RenderChanges::default();
        let mut universe = Universe::builder()
            .with_system(RendererSystem::new(changes.clone()))
            .build();
        let mut commands = universe.handle().command_buffer();
        let first = commands.create_world(World::builder("first"));
        let second = commands.create_world(World::builder("second"));
        let entity = commands.spawn(
            first,
            Entity::builder().with_component(Transform::default()),
        );
        commands.set_component(
            entity,
            Transform {
                location: glam::Vec3::X,
                ..Transform::default()
            },
        );
        commands.set_component(
            entity,
            Transform {
                location: glam::Vec3::Y,
                ..Transform::default()
            },
        );
        commands.send(entity, second);
        commands.submit();
        universe.tick(0.0);
        let deltas = changes.extract(&universe, std::iter::empty());
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].entity, entity);
        let final_state = deltas[0].state.as_ref().expect("live entity");
        assert_eq!(final_state.world, second);
        assert_eq!(
            final_state
                .transform
                .as_ref()
                .expect("final transform")
                .location,
            glam::Vec3::Y
        );
        assert!(changes.extract(&universe, std::iter::empty()).is_empty());
        let mut commands = universe.handle().command_buffer();
        commands.remove_component::<Transform>(entity);
        commands.submit();
        universe.tick(0.0);
        let deltas = changes.extract(&universe, [(entity, second)].into_iter());
        assert!(
            deltas[0]
                .state
                .as_ref()
                .expect("still alive")
                .transform
                .is_none()
        );
        let mut commands = universe.handle().command_buffer();
        commands.despawn(entity);
        commands.submit();
        universe.tick(0.0);
        let deltas = changes.extract(&universe, [(entity, second)].into_iter());
        assert_eq!(deltas.len(), 1);
        assert!(deltas[0].state.is_none());
    }
}
