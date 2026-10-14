//! Player movement systems.

use dirk_universe::{
    query::Query,
    systems::{DeltaTime, System},
};

use crate::{PlayerId, PlayerInputState};
use dirk_world::components::Transform;

/// Default movement speed in world units per second.
pub const DEFAULT_PLAYER_MOVE_SPEED: f64 = 350.0;
/// Default pointer-look sensitivity, in radians per normalized viewport unit.
pub const DEFAULT_PLAYER_LOOK_SENSITIVITY: f32 = 0.5;

/// Applies player movement input to entities that have a [`PlayerId`] and
/// [`dirk_world::components::Transform`].
pub struct PlayerMovementSystem {
    input_state: PlayerInputState,
    speed: f64,
    look_sensitivity: f32,
}

impl PlayerMovementSystem {
    /// Creates a movement system using the default movement speed.
    #[must_use]
    pub fn new(input_state: PlayerInputState) -> Self {
        Self {
            input_state,
            speed: DEFAULT_PLAYER_MOVE_SPEED,
            look_sensitivity: DEFAULT_PLAYER_LOOK_SENSITIVITY,
        }
    }
}

impl System<(Query<'_, (&PlayerId, &mut Transform)>, DeltaTime)> for PlayerMovementSystem {
    fn run(
        &mut self,
        (mut query, DeltaTime(delta_time)): (Query<'_, (&PlayerId, &mut Transform)>, DeltaTime),
    ) {
        for (player, mut transform) in &mut query {
            let player = *player;
            let input = self.input_state.get(player);
            if input.movement == glam::Vec3::ZERO && input.look == glam::DVec2::ZERO {
                continue;
            }

            if input.look != glam::DVec2::ZERO {
                transform.rotate_by_pointer_delta(input.look, self.look_sensitivity);
            }

            if input.movement != glam::Vec3::ZERO {
                let movement = transform.movement_direction(input.movement);
                if movement != glam::Vec3::ZERO {
                    #[allow(clippy::cast_possible_truncation)]
                    let distance = (self.speed * delta_time) as f32;
                    transform.location += movement * distance;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlayerInputFrame;
    use dirk_universe::{Entity, Universe, World};

    #[test]
    fn registered_movement_updates_components_in_the_same_tick() {
        let input = PlayerInputState::default();
        let player = PlayerId::default();
        input.set(
            player,
            PlayerInputFrame {
                movement: glam::Vec3::X,
                ..PlayerInputFrame::default()
            },
        );
        let mut universe = Universe::builder()
            .with_system(PlayerMovementSystem::new(input.clone()))
            .build();
        let mut cmd = universe.handle().command_buffer();
        let world = cmd.create_world(World::builder("movement"));
        let moving = [0, 1].map(|_| {
            cmd.spawn(
                world,
                Entity::builder()
                    .with_component(player)
                    .with_component(Transform::default()),
            )
        });
        let stationary = cmd.spawn(
            world,
            Entity::builder().with_component(Transform::default()),
        );
        cmd.spawn(world, Entity::builder().with_component(player));
        cmd.submit();
        universe.tick(0.5);
        for entity in moving {
            assert_eq!(
                universe
                    .component::<Transform>(entity)
                    .expect("transform")
                    .location,
                glam::Vec3::new(175.0, 0.0, 0.0)
            );
        }
        input.set(player, PlayerInputFrame::default());
        universe.tick(0.0);
        for entity in moving {
            assert_eq!(
                universe
                    .component::<Transform>(entity)
                    .expect("transform")
                    .location,
                glam::Vec3::new(175.0, 0.0, 0.0)
            );
        }
        assert_eq!(
            universe
                .component::<Transform>(stationary)
                .expect("transform")
                .location,
            glam::Vec3::ZERO
        );
    }
}
