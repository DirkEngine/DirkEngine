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

impl System for PlayerMovementSystem {
    type Params<'u> = (Query<'u, (&'u PlayerId, &'u mut Transform)>, DeltaTime);

    fn run(&mut self, (mut query, DeltaTime(delta_time)): Self::Params<'_>) {
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
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "A frame's movement distance only loses precision as an f32, the transforms' precision."
                    )]
                    let distance = (self.speed * delta_time) as f32;
                    transform.location += movement * distance;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
