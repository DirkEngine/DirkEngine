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
        .build()
        .expect("movement should schedule");
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
