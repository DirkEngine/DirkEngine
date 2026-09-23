//! Integration tests for the `universe` crate.

use dirk_universe::{
    CommandBuffer, Entity, EntityBuilder, Universe, World, WorldId,
    components::Component,
    query::{QueryItem, Read},
    systems::FuncSystem,
};

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Position(i32, i32);

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Hidden;

fn spawn_entity(universe: &mut Universe, world: WorldId, builder: EntityBuilder) -> Entity {
    let mut cmd = universe.handle().command_buffer();
    let entity = cmd.spawn(world, builder);
    cmd.submit();
    universe.tick(0.0);
    entity
}

#[test]
fn universe_public_api_supports_entity_lifecycle_across_worlds() {
    let mut universe = Universe::builder()
        .with_world(World::builder("overworld"))
        .with_world(World::builder("dungeon"))
        .build();
    universe.tick(0.0);

    let overworld = dirk_universe::WorldId::default();
    let dungeon = overworld + 1;

    let entity = spawn_entity(
        &mut universe,
        overworld,
        Entity::builder().with_component(Position(2, 3)),
    );

    assert!(universe.is_alive(entity));
    assert!(universe.is_in_world(overworld, entity));
    assert_eq!(universe.get_world(entity), Some(overworld));

    let mut cmd = universe.handle().command_buffer();
    cmd.send(entity, dungeon);
    cmd.submit();
    universe.tick(0.016);

    assert!(universe.is_in_world(dungeon, entity));
    assert_eq!(universe.get_world(entity), Some(dungeon));

    let mut cmd = universe.handle().command_buffer();
    cmd.despawn(entity);
    cmd.submit();
    universe.tick(0.016);

    assert!(!universe.is_alive(entity));
}

#[test]
fn buffered_spawns_are_applied_on_tick_and_components_are_readable() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = dirk_universe::WorldId::default();

    let mut cmd = universe.handle().command_buffer();
    let e0 = cmd.spawn(world, Entity::builder().with_component(Position(1, 1)));
    let e1 = cmd.spawn(
        world,
        Entity::builder()
            .with_component(Position(9, 9))
            .with_component(Hidden),
    );
    cmd.submit();

    universe.tick(0.016);

    assert_eq!(universe.alive_count(), 2);

    assert_eq!(
        universe.component::<Position>(e0).map(|p| (p.0, p.1)),
        Some((1, 1))
    );
    assert_eq!(universe.component::<Hidden>(e0).map(|_| true), None);

    assert_eq!(
        universe.component::<Position>(e1).map(|p| (p.0, p.1)),
        Some((9, 9))
    );
    assert_eq!(universe.component::<Hidden>(e1).map(|_| true), Some(true));
}

#[test]
fn public_command_buffers_allocate_unique_handles() {
    let mut universe = Universe::builder().build();

    let mut first_cmd = universe.handle().command_buffer();
    let first_world = first_cmd.create_world(World::builder("first"));
    let first_entity = first_cmd.spawn(
        first_world,
        Entity::builder().with_component(Position(4, 8)),
    );

    let mut second_cmd = universe.handle().command_buffer();
    let second_world = second_cmd.create_world(World::builder("second"));
    let second_entity = second_cmd.spawn(
        second_world,
        Entity::builder().with_component(Position(16, 32)),
    );

    first_cmd.submit();
    second_cmd.submit();
    universe.tick(0.016);

    assert_ne!(first_world, second_world);
    assert_ne!(first_entity, second_entity);
    assert_eq!(universe.world(first_world).map(World::name), Some("first"));
    assert_eq!(
        universe.world(second_world).map(World::name),
        Some("second")
    );
    assert!(universe.is_in_world(first_world, first_entity));
    assert!(universe.is_in_world(second_world, second_entity));
    assert_eq!(
        universe
            .component::<Position>(first_entity)
            .map(|p| (p.0, p.1)),
        Some((4, 8))
    );
    assert_eq!(
        universe
            .component::<Position>(second_entity)
            .map(|p| (p.0, p.1)),
        Some((16, 32))
    );
}

#[test]
fn registered_function_systems_keep_state_and_defer_commands_until_next_tick() {
    use std::{cell::RefCell, rc::Rc};

    let seen = Rc::new(RefCell::new(Vec::new()));
    let system_seen = Rc::clone(&seen);
    let mut calls = 0;
    let system = FuncSystem::new(
        move |cmd: &mut CommandBuffer, universe: &Universe, delta_time| {
            for query in QueryItem::<Read<Position>>::iter(universe) {
                calls += 1;
                let position = query.params();
                system_seen
                    .borrow_mut()
                    .push((calls, query.entity(), position.0, delta_time));
                cmd.set_component(query.entity(), Position(position.0 + calls, position.1));
            }
        },
    );
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .with_system(system)
        .build();
    universe.tick(0.0);
    assert!(seen.borrow().is_empty());

    let mut cmd = universe.handle().command_buffer();
    let entity = cmd.spawn(
        WorldId::default(),
        Entity::builder().with_component(Position(2, 3)),
    );
    cmd.submit();

    universe.tick(0.25);
    assert_eq!(universe.component::<Position>(entity).map(|p| p.0), Some(2));
    universe.tick(0.5);
    assert_eq!(universe.component::<Position>(entity).map(|p| p.0), Some(3));
    assert_eq!(
        *seen.borrow(),
        vec![(1, entity, 2, 0.25), (2, entity, 3, 0.5)]
    );

    let mut cmd = universe.handle().command_buffer();
    cmd.despawn(entity);
    cmd.submit();
    universe.tick(1.0);
    assert!(!universe.is_alive(entity));
    assert_eq!(seen.borrow().len(), 2);
}

#[test]
fn composed_builders_preserve_system_order_and_use_the_final_command_queue() {
    use std::{cell::RefCell, rc::Rc};

    let seen = Rc::new(RefCell::new(Vec::new()));
    let make_system = |index| {
        let seen = Rc::clone(&seen);
        FuncSystem::new(move |cmd: &mut CommandBuffer, universe: &Universe, _| {
            for query in QueryItem::<Read<Position>>::iter(universe) {
                seen.borrow_mut().push((index, query.params().0));
                cmd.set_component(query.entity(), Position(index, 0));
            }
        })
    };
    let other = Universe::builder().with_system(make_system(2));
    let mut universe = Universe::builder()
        .with_world(
            World::builder("w").with_entity(Entity::builder().with_component(Position(0, 0))),
        )
        .with_system(make_system(1))
        .with_other(other)
        .build();

    universe.tick(0.0);
    assert_eq!(*seen.borrow(), vec![(1, 0), (2, 0)]);

    universe.tick(0.0);
    assert_eq!(*seen.borrow(), vec![(1, 0), (2, 0), (1, 2), (2, 2)]);
}

#[test]
fn systems_run_once_per_tick_even_without_entities_and_share_changes() {
    use std::{cell::RefCell, rc::Rc};
    let observed = Rc::new(RefCell::new(Vec::new()));
    let mut builder = Universe::builder();
    for id in [1, 2] {
        let observed = Rc::clone(&observed);
        let mut calls = 0;
        builder = builder.with_system(FuncSystem::new(move |_, universe, delta_time| {
            calls += 1;
            observed
                .borrow_mut()
                .push((id, calls, universe.changes().count(), delta_time));
        }));
    }
    let mut universe = builder.build();
    universe.tick(0.25);
    let mut cmd = universe.handle().command_buffer();
    cmd.create_world(World::builder("empty"));
    cmd.submit();
    universe.tick(0.5);
    universe.tick(1.0);
    assert_eq!(
        *observed.borrow(),
        vec![
            (1, 1, 0, 0.25),
            (2, 1, 0, 0.25),
            (1, 2, 1, 0.5),
            (2, 2, 1, 0.5),
            (1, 3, 0, 1.0),
            (2, 3, 0, 1.0),
        ]
    );
}

#[test]
fn change_log_retains_intermediate_values_and_orders_transient_lifecycles() {
    use dirk_universe::changes::{Change, ComponentChange};
    let mut universe = Universe::builder().build();
    let mut cmd = universe.handle().command_buffer();
    let first = cmd.create_world(World::builder("first"));
    let second = cmd.create_world(World::builder("second"));
    let entity = cmd.spawn(first, Entity::builder().with_component(Position(1, 0)));
    cmd.set_component(entity, Position(2, 0));
    cmd.set_component(entity, Position(3, 0));
    cmd.remove_component::<Position>(entity);
    cmd.remove_component::<Position>(entity); // No duplicate removal event.
    cmd.set_component(entity, Position(4, 0));
    cmd.send(entity, second);
    cmd.destroy_world(second);
    cmd.despawn(entity); // Already removed by world destruction.
    cmd.set_component(entity, Position(5, 0)); // Must not revive it.
    cmd.destroy_world(first);
    cmd.submit();
    universe.tick(0.0);

    let values: Vec<_> = universe
        .component_changes::<Position>()
        .map(|change| match change {
            ComponentChange::Added { component, .. } => ("added", None, Some(component.0)),
            ComponentChange::Updated { old, new, .. } => ("updated", Some(old.0), Some(new.0)),
            ComponentChange::Removed { component, .. } => ("removed", Some(component.0), None),
        })
        .collect();
    assert_eq!(
        values,
        vec![
            ("added", None, Some(1)),
            ("updated", Some(1), Some(2)),
            ("updated", Some(2), Some(3)),
            ("removed", Some(3), None),
            ("added", None, Some(4)),
            ("removed", Some(4), None),
        ]
    );
    let lifecycle: Vec<_> = universe
        .changes()
        .map(|change| match change {
            Change::WorldCreated { .. } => "world-created",
            Change::EntitySpawned {
                entity: changed,
                world,
            } => {
                assert_eq!((*changed, *world), (entity, first));
                "spawned"
            }
            Change::ComponentAdded { .. } => "added",
            Change::ComponentUpdated { .. } => "updated",
            Change::ComponentRemoved { .. } => "removed",
            Change::EntityMoved {
                entity: changed,
                from,
                to,
            } => {
                assert_eq!((*changed, *from, *to), (entity, first, second));
                "moved"
            }
            Change::EntityDespawned {
                entity: changed,
                world,
            } => {
                assert_eq!((*changed, *world), (entity, second));
                "despawned"
            }
            Change::WorldDestroyed { .. } => "world-destroyed",
        })
        .collect();
    assert_eq!(
        lifecycle,
        vec![
            "world-created",
            "world-created",
            "spawned",
            "added",
            "updated",
            "updated",
            "removed",
            "added",
            "moved",
            "removed",
            "despawned",
            "world-destroyed",
            "world-destroyed"
        ]
    );
    assert_eq!(universe.alive_count(), 0);
    assert_eq!(universe.worlds().count(), 0);
    assert_eq!(QueryItem::<Read<Position>>::iter(&universe).count(), 0);
    universe.tick(0.0);
    assert_eq!(universe.changes().count(), 0);
}

#[test]
fn removed_non_clone_components_are_released_when_the_log_expires() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Debug, Component)]
    struct Tracked(Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut universe = Universe::builder().build();
    let mut cmd = universe.handle().command_buffer();
    let world = cmd.create_world(World::builder("w"));
    let entity = cmd.spawn(
        world,
        Entity::builder().with_component(Tracked(Arc::clone(&dropped))),
    );
    cmd.despawn(entity);
    cmd.submit();
    universe.tick(0.0);
    assert_eq!(universe.component_changes::<Tracked>().count(), 2);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    universe.tick(0.0);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
