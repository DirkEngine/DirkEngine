//! Player movement systems.

use dirk_universe::{
    CommandBuffer, Entity, Universe,
    query::Query,
    systems::{System, TickingSystem},
};

use crate::{PlayerId, PlayerInputState};

/// Default movement speed in world units per second.
pub const DEFAULT_PLAYER_MOVE_SPEED: f64 = 350.0;
/// Default pointer-look sensitivity, in radians per normalized viewport unit.
pub const DEFAULT_PLAYER_LOOK_SENSITIVITY: f32 = 0.5;

/// Applies player movement input to entities that have a [`PlayerId`] and
/// [`dirk_world::components::Transform`].
#[derive(System)]
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

impl TickingSystem for PlayerMovementSystem {
    fn tick(
        &self,
        cmd: &mut CommandBuffer,
        universe: &Universe,
        delta_time: f64,
        entities: &mut dyn Iterator<Item = Entity>,
    ) {
        for entity in entities {
            let Some(player) = universe.component::<PlayerId>(entity).copied() else {
                continue;
            };
            let input = self.input_state.get(player);
            if input.movement == glam::Vec3::ZERO && input.look == glam::DVec2::ZERO {
                continue;
            }

            let Some(transform) = universe.component::<dirk_world::components::Transform>(entity)
            else {
                continue;
            };

            let mut transform = transform.clone();
            if input.look != glam::DVec2::ZERO {
                transform.rotate_by_pointer_delta(input.look, self.look_sensitivity);
            }

            if input.movement != glam::Vec3::ZERO {
                let movement = transform.movement_direction(input.movement);
                if movement != glam::Vec3::ZERO {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "A frame's movement distance only loses precision as an f32, the transforms' precision."
                    )]
                    let distance = (self.speed * delta_time) as f32;
                    transform.location += movement * distance;
                }
            }

            cmd.set_component(entity, transform);
        }
    }

    fn query(&self) -> Query {
        Query::empty()
            .with_component::<PlayerId>()
            .with_component::<dirk_world::components::Transform>()
    }
}
