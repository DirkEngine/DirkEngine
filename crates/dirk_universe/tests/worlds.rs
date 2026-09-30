//! Tests for per-world queries and isolated worlds.

use std::{any::type_name, cell::RefCell, rc::Rc};

use dirk_universe::{prelude::*, schedule::ScheduleError};

#[derive(Debug, Component)]
struct Position(i32);

#[derive(Debug, Component)]
struct Gravity(i32);

/// Creates worlds from `builders`, returning the universe and their IDs.
fn universe_with<const N: usize>(
    builder: dirk_universe::UniverseBuilder,
    builders: [dirk_universe::WorldBuilder; N],
) -> (Universe, [WorldId; N]) {
    let mut universe = builder.build().expect("systems should schedule");
    let mut cmd = universe.handle().command_buffer();
    let ids = builders.map(|builder| cmd.create_world(builder));
    cmd.submit();
    universe.tick(0.0);
    (universe, ids)
}

fn spawn(universe: &mut Universe, world: WorldId, position: i32) -> Entity {
    let mut cmd = universe.handle().command_buffer();
    let entity = cmd.spawn(world, Entity::builder().with_component(Position(position)));
    cmd.submit();
    universe.tick(0.0);
    entity
}

type Seen<T> = Rc<RefCell<Vec<T>>>;

#[test]
fn per_world_queries_see_only_their_world_including_isolated_ones() {
    let seen: Seen<(WorldId, Vec<i32>)> = Seen::default();
    let shared: Seen<Vec<i32>> = Seen::default();
    let (per_world_seen, shared_seen) = (Rc::clone(&seen), Rc::clone(&shared));
    let builder = Universe::builder().with_system(
        move |worlds: PerWorld<Query<&Position>>, everywhere: Query<&Position>| {
            for (world, positions) in &worlds {
                let mut values: Vec<_> = positions.iter().map(|p| p.0).collect();
                values.sort_unstable();
                per_world_seen.borrow_mut().push((world, values));
            }
            let mut values: Vec<_> = everywhere.iter().map(|p| p.0).collect();
            values.sort_unstable();
            shared_seen.borrow_mut().push(values);
        },
    );
    let (mut universe, [open, other, isolated]) = universe_with(
        builder,
        [
            World::builder("open"),
            World::builder("other"),
            World::builder("isolated").isolated(),
        ],
    );
    spawn(&mut universe, open, 1);
    spawn(&mut universe, other, 2);
    spawn(&mut universe, isolated, 3);
    seen.borrow_mut().clear();
    shared.borrow_mut().clear();

    universe.tick(0.0);

    assert_eq!(
        *seen.borrow(),
        vec![(open, vec![1]), (other, vec![2]), (isolated, vec![3])]
    );
    assert_eq!(*shared.borrow(), vec![vec![1, 2]]);
    // Queries from outside systems see every world.
    assert_eq!(universe.query::<&Position>().iter().count(), 3);
}

#[test]
fn entities_cannot_cross_into_or_out_of_isolated_worlds() {
    let (mut universe, [open, isolated]) = universe_with(
        Universe::builder(),
        [
            World::builder("open"),
            World::builder("isolated").isolated(),
        ],
    );
    let outside = spawn(&mut universe, open, 1);
    let inside = spawn(&mut universe, isolated, 2);

    let mut cmd = universe.handle().command_buffer();
    cmd.send(outside, isolated);
    cmd.send(inside, open);
    cmd.submit();
    universe.tick(0.0);

    assert_eq!(universe.get_world(outside), Some(open));
    assert_eq!(universe.get_world(inside), Some(isolated));
}

#[test]
fn per_world_deltas_report_removals_where_they_happened() {
    let seen: Seen<(WorldId, Vec<(Entity, bool)>)> = Seen::default();
    let shared: Seen<Vec<Entity>> = Seen::default();
    let (per_world_seen, shared_seen) = (Rc::clone(&seen), Rc::clone(&shared));
    let builder = Universe::builder().with_system(
        move |worlds: PerWorld<Query<(Entity, Delta<InWorld>)>>,
              everywhere: Query<(Entity, Delta<InWorld>)>| {
            for (world, locations) in &worlds {
                let mut deltas: Vec<_> = locations
                    .iter()
                    .map(|(entity, delta)| (entity, delta.value().is_some()))
                    .collect();
                deltas.sort_by_key(|(entity, _)| entity.raw());
                if !deltas.is_empty() {
                    per_world_seen.borrow_mut().push((world, deltas));
                }
            }
            let mut entities: Vec<_> = everywhere.iter().map(|(entity, _)| entity).collect();
            entities.sort_by_key(|entity| entity.raw());
            shared_seen.borrow_mut().push(entities);
        },
    );
    let (mut universe, [open, isolated]) = universe_with(
        builder,
        [
            World::builder("open"),
            World::builder("isolated").isolated(),
        ],
    );
    let outside = spawn(&mut universe, open, 1);
    let inside = spawn(&mut universe, isolated, 2);
    seen.borrow_mut().clear();
    shared.borrow_mut().clear();

    let mut cmd = universe.handle().command_buffer();
    cmd.despawn(outside);
    cmd.destroy_world(isolated);
    cmd.submit();
    universe.tick(0.0);

    // The destroyed world is still visited once, to report its removals.
    assert_eq!(
        *seen.borrow(),
        vec![
            (open, vec![(outside, false)]),
            (isolated, vec![(inside, false)])
        ]
    );
    // Removals from isolated worlds stay hidden from queries spanning worlds.
    assert_eq!(*shared.borrow(), vec![vec![outside]]);

    seen.borrow_mut().clear();
    universe.tick(0.0);
    assert!(seen.borrow().is_empty());
}

#[test]
fn worlds_carry_their_own_data() {
    let seen: Seen<(WorldId, i32, Vec<i32>)> = Seen::default();
    let system_seen = Rc::clone(&seen);
    let builder = Universe::builder().with_system(
        move |gravity: Query<&Gravity>, mut worlds: PerWorld<Query<&mut Position>>| {
            for (world, mut positions) in &mut worlds {
                let Some(gravity) = gravity.get(world.entity()) else {
                    continue;
                };
                let mut fallen = Vec::new();
                for mut position in &mut positions {
                    position.0 += gravity.0;
                    fallen.push(position.0);
                }
                system_seen.borrow_mut().push((world, gravity.0, fallen));
            }
        },
    );
    let (mut universe, [earth, moon]) = universe_with(
        builder,
        [
            World::builder("earth").with_component(Gravity(-10)),
            World::builder("moon").with_component(Gravity(-2)),
        ],
    );
    spawn(&mut universe, earth, 100);
    spawn(&mut universe, moon, 100);
    seen.borrow_mut().clear();

    universe.tick(0.0);

    assert_eq!(
        *seen.borrow(),
        vec![(earth, -10, vec![70]), (moon, -2, vec![96])]
    );
}

#[test]
fn per_world_queries_share_access_rules_with_other_queries() {
    fn conflicting(_worlds: PerWorld<Query<&mut Position>>, _all: Query<&Position>) {}
    let error = Universe::builder()
        .with_system(conflicting)
        .build()
        .err()
        .expect("conflicting borrows should not schedule");
    assert_eq!(
        error,
        ScheduleError::ConflictingAccess {
            system: std::any::type_name_of_val(&conflicting),
            component: type_name::<Position>(),
        }
    );
}
