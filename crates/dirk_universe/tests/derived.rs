//! Tests for derived components.

use std::{any::type_name, cell::Cell};

use dirk_universe::{UniverseBuilder, prelude::*, schedule::ScheduleError};

#[derive(Debug, Clone, Component)]
struct Velocity(i32);

#[derive(Debug, Clone, Component)]
struct Mass(i32);

#[derive(Debug, Clone, PartialEq, Component)]
#[component(read_only)]
struct Momentum(i32);

#[derive(Debug, Clone, PartialEq, Component)]
#[component(read_only)]
struct Moving(bool);

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

fn momentum(velocity: &Velocity, mass: Option<&Mass>) -> Momentum {
    CALLS.set(CALLS.get() + 1);
    Momentum(velocity.0 * mass.map_or(1, |mass| mass.0))
}

fn moving(momentum: &Momentum) -> Moving {
    Moving(momentum.0 != 0)
}

/// Builds `builder` with one world and returns it with the world's ID.
fn build(builder: UniverseBuilder) -> (Universe, WorldId) {
    let mut universe = builder
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let world = universe.worlds().next().expect("one world").id();
    (universe, world)
}

/// Applies `edit` through a command buffer, then ticks once.
fn apply(universe: &mut Universe, edit: impl FnOnce(&mut CommandBuffer)) {
    let mut cmd = universe.handle().command_buffer();
    edit(&mut cmd);
    cmd.submit();
    universe.tick(0.0);
}

fn momentum_of(universe: &Universe, entity: Entity) -> Option<i32> {
    universe.component::<Momentum>(entity).map(|m| m.0)
}

#[test]
fn outputs_follow_their_inputs() {
    let (mut universe, world) = build(Universe::builder().with_derived(momentum));
    let mut entity = None;
    let mut still = None;
    apply(&mut universe, |cmd| {
        entity = Some(cmd.spawn(world, Entity::builder().with_component(Velocity(3))));
        still = Some(cmd.spawn(world, Entity::builder().with_component(Mass(5))));
    });
    let (entity, still) = (entity.expect("spawned"), still.expect("spawned"));

    assert_eq!(momentum_of(&universe, entity), Some(3));
    assert_eq!(momentum_of(&universe, still), None);

    apply(&mut universe, |cmd| cmd.set_component(entity, Mass(2)));
    assert_eq!(momentum_of(&universe, entity), Some(6));

    apply(&mut universe, |cmd| cmd.remove_component::<Mass>(entity));
    assert_eq!(momentum_of(&universe, entity), Some(3));

    apply(&mut universe, |cmd| {
        cmd.remove_component::<Velocity>(entity)
    });
    assert_eq!(momentum_of(&universe, entity), None);
}

#[test]
fn outputs_are_recomputed_only_when_inputs_change() {
    let (mut universe, world) = build(Universe::builder().with_derived(momentum));
    let mut entity = None;
    apply(&mut universe, |cmd| {
        entity = Some(cmd.spawn(world, Entity::builder().with_component(Velocity(1))));
        cmd.spawn(world, Entity::builder().with_component(Velocity(2)));
    });
    let entity = entity.expect("spawned");
    CALLS.set(0);

    universe.tick(0.0);
    assert_eq!(CALLS.get(), 0);

    apply(&mut universe, |cmd| cmd.set_component(entity, Velocity(4)));
    assert_eq!(CALLS.get(), 1);
}

#[test]
fn equal_outputs_are_not_changes() {
    thread_local! {
        static CHANGED: Cell<usize> = const { Cell::new(0) };
    }
    fn observe(changed: Query<Entity, Changed<Moving>>) {
        CHANGED.set(CHANGED.get() + changed.iter().count());
    }
    let (mut universe, world) = build(
        Universe::builder()
            .with_derived(momentum)
            .with_derived(moving)
            .with_system(observe),
    );
    let mut entity = None;
    apply(&mut universe, |cmd| {
        entity = Some(cmd.spawn(world, Entity::builder().with_component(Velocity(1))));
    });
    let entity = entity.expect("spawned");
    assert_eq!(CHANGED.get(), 1);

    apply(&mut universe, |cmd| cmd.set_component(entity, Velocity(7)));
    assert_eq!(momentum_of(&universe, entity), Some(7));
    assert_eq!(CHANGED.get(), 1, "Moving stayed true");

    apply(&mut universe, |cmd| cmd.set_component(entity, Velocity(0)));
    assert_eq!(CHANGED.get(), 2);
}

#[test]
fn derivations_run_between_input_writers_and_output_readers_in_one_tick() {
    thread_local! {
        static SEEN: Cell<Option<bool>> = const { Cell::new(None) };
    }
    fn read(moving: Query<&Moving>) {
        SEEN.set(moving.iter().next().map(|moving| moving.0));
    }
    fn stop(mut velocities: Query<&mut Velocity>) {
        for mut velocity in &mut velocities {
            velocity.0 = 0;
        }
    }
    let (mut universe, world) = build(
        Universe::builder()
            .with_system(read)
            .with_derived(moving)
            .with_system(stop)
            .with_derived(momentum),
    );
    apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Velocity(5)));
    });

    // `stop` zeroes the velocity before both derivations and the reader run.
    assert_eq!(SEEN.get(), Some(false));
    let names: Vec<_> = universe
        .schedule()
        .systems()
        .iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(
        names,
        vec![
            std::any::type_name_of_val(&stop),
            std::any::type_name_of_val(&momentum),
            std::any::type_name_of_val(&moving),
            std::any::type_name_of_val(&read),
        ]
    );
}

#[test]
fn removed_outputs_are_reported_as_deltas() {
    thread_local! {
        static REMOVED: Cell<usize> = const { Cell::new(0) };
    }
    fn observe(momenta: Query<(Entity, Delta<Momentum>)>) {
        let removed = momenta.iter().filter(|(_, delta)| delta.value().is_none());
        REMOVED.set(REMOVED.get() + removed.count());
    }
    let (mut universe, world) = build(
        Universe::builder()
            .with_system(observe)
            .with_derived(momentum),
    );
    let mut entity = None;
    apply(&mut universe, |cmd| {
        entity = Some(cmd.spawn(world, Entity::builder().with_component(Velocity(1))));
    });
    apply(&mut universe, |cmd| {
        cmd.remove_component::<Velocity>(entity.expect("spawned"));
    });
    assert_eq!(REMOVED.get(), 1);
}

#[test]
fn closures_derive_components_too() {
    let (mut universe, world) =
        build(Universe::builder().with_derived(|velocity: &Velocity| Momentum(velocity.0 * 10)));
    let mut entity = None;
    apply(&mut universe, |cmd| {
        entity = Some(cmd.spawn(world, Entity::builder().with_component(Velocity(2))));
    });
    assert_eq!(momentum_of(&universe, entity.expect("spawned")), Some(20));
}

#[test]
fn one_component_has_one_derivation() {
    fn doubled(velocity: &Velocity) -> Momentum {
        Momentum(velocity.0 * 2)
    }
    let error = Universe::builder()
        .with_derived(momentum)
        .with_derived(doubled)
        .build()
        .err()
        .expect("two derivations of one component should not schedule");
    assert_eq!(
        error,
        ScheduleError::DuplicateDerivation {
            component: type_name::<Momentum>(),
            first: std::any::type_name_of_val(&momentum),
            second: std::any::type_name_of_val(&doubled),
        }
    );
}
