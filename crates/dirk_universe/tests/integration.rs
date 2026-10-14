//! Integration tests for the `universe` crate.

use dirk_universe::{
    Entity, EntityBuilder, Universe, World, WorldId,
    components::Component,
    query::{
        Query,
        filter::{Added, Changed, Without},
    },
    systems::{Commands, DeltaTime, Lifecycle, RemovedComponents, ToSystem},
};

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Position(i32, i32);

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Hidden;

#[derive(Debug, Component)]
struct Counter(i32);

#[derive(Debug, Component)]
struct Step(i32);

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
    let system = move |mut cmd: Commands<'_>,
                       query: Query<'_, (Entity, &Position)>,
                       DeltaTime(delta_time)| {
        calls += 1;
        for (entity, position) in &query {
            system_seen
                .borrow_mut()
                .push((calls, entity, position.0, delta_time));
            cmd.set_component(entity, Position(position.0 + calls, position.1));
        }
    };
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
    assert_eq!(universe.component::<Position>(entity).map(|p| p.0), Some(4));
    assert_eq!(
        *seen.borrow(),
        vec![(2, entity, 2, 0.25), (3, entity, 4, 0.5)]
    );

    let mut cmd = universe.handle().command_buffer();
    cmd.despawn(entity);
    cmd.submit();
    universe.tick(1.0);
    assert!(!universe.is_alive(entity));
    assert_eq!(seen.borrow().len(), 2);
}

#[test]
fn multiple_queries_are_independent_collections_across_worlds() {
    use std::{cell::RefCell, rc::Rc};

    let observed = Rc::new(RefCell::new(Vec::new()));
    let system_observed = Rc::clone(&observed);
    let mut universe = Universe::builder()
        .with_world(
            World::builder("first")
                .with_entity(Entity::builder().with_component(Position(1, 2)))
                .with_entity(Entity::builder().with_component(Hidden))
                .with_entity(
                    Entity::builder()
                        .with_component(Position(3, 4))
                        .with_component(Hidden),
                ),
        )
        .with_world(
            World::builder("second").with_entity(
                Entity::builder()
                    .with_component(Position(5, 6))
                    .with_component(Hidden),
            ),
        )
        .with_system(
            move |position: Query<'_, &Position>, hidden: Query<'_, &Hidden>| {
                assert_eq!(hidden.iter().count(), 3);
                system_observed
                    .borrow_mut()
                    .extend(position.iter().map(|position| position.0));
            },
        )
        .build();

    universe.tick(0.0);
    observed.borrow_mut().sort_unstable();
    assert_eq!(*observed.borrow(), vec![1, 3, 5]);
}

#[test]
fn independent_observers_see_mutations_according_to_system_order() {
    use std::{cell::RefCell, rc::Rc};
    let before = Rc::new(RefCell::new(Vec::new()));
    let after = Rc::new(RefCell::new(Vec::new()));
    let second_after = Rc::new(RefCell::new(Vec::new()));
    let observer = |seen: Rc<RefCell<Vec<Vec<i32>>>>| {
        move |query: Query<&Counter, Changed<Counter>>| {
            seen.borrow_mut()
                .push(query.iter().map(|counter| counter.0).collect());
        }
    };
    let mut writes = 0;
    let mut universe = Universe::builder()
        .with_world(World::builder("w").with_entity(Entity::builder().with_component(Counter(0))))
        .with_system(observer(before.clone()))
        .with_system(move |mut query: Query<&mut Counter>| {
            if writes < 2 {
                for mut counter in &mut query {
                    counter.0 += 1;
                }
                writes += 1;
            }
        })
        .with_system(observer(after.clone()))
        .with_system(observer(second_after.clone()))
        .build();
    for _ in 0..4 {
        universe.tick(0.0);
    }
    assert_eq!(*before.borrow(), vec![vec![0], vec![1], vec![2], vec![]]);
    assert_eq!(*after.borrow(), vec![vec![1], vec![2], vec![], vec![]]);
    assert_eq!(*second_after.borrow(), *after.borrow());
}

#[test]
#[should_panic(expected = "overlapping mutable queries")]
fn overlapping_mutable_component_access_is_rejected_at_registration() {
    let _ = Universe::builder().with_system(|_: Query<'_, (&Counter, &mut Counter)>| {});
}

#[test]
#[should_panic(expected = "overlapping mutable queries")]
fn overlapping_query_parameters_are_rejected_at_registration() {
    let _ =
        Universe::builder().with_system(|_: Query<'_, &Counter>, _: Query<'_, &mut Counter>| {});
}

#[test]
fn queries_run_once_with_zero_or_multiple_matches_and_support_lookups() {
    use std::{cell::RefCell, rc::Rc};
    let seen = Rc::new(RefCell::new(Vec::new()));
    let system_seen = Rc::clone(&seen);
    let mut universe = Universe::builder()
        .with_system(move |query: Query<(Entity, &Position)>| {
            let items: Vec<_> = query.iter().collect();
            for (entity, position) in &items {
                assert_eq!(query.get(*entity).unwrap().1.0, position.0);
            }
            system_seen.borrow_mut().push(items.len());
        })
        .build();
    universe.tick(0.0);
    let mut cmd = universe.handle().command_buffer();
    let world = cmd.create_world(World::builder("aggregate"));
    for x in [1, 2] {
        cmd.spawn(world, Entity::builder().with_component(Position(x, 0)));
    }
    cmd.submit();
    universe.tick(0.0);
    assert_eq!(*seen.borrow(), vec![0, 2]);
}

#[test]
fn mutable_reads_stay_clean_and_equal_writes_count_as_changes() {
    use std::{cell::RefCell, rc::Rc};
    let seen = Rc::new(RefCell::new(Vec::new()));
    let system_seen = Rc::clone(&seen);
    let mut run = 0;
    let mut universe = Universe::builder()
        .with_world(World::builder("w").with_entity(Entity::builder().with_component(Counter(7))))
        .with_system(move |mut query: Query<&mut Counter>| {
            run += 1;
            for mut counter in &mut query {
                assert_eq!(counter.0, 7);
                if run == 3 {
                    counter.0 = 7;
                }
            }
        })
        .with_system(move |query: Query<&Counter, Changed<Counter>>| {
            system_seen.borrow_mut().push(query.iter().count());
        })
        .build();
    for _ in 0..4 {
        universe.tick(0.0);
    }
    assert_eq!(*seen.borrow(), vec![1, 0, 1, 0]);
}

#[test]
fn composed_builders_preserve_system_order_and_use_the_final_command_queue() {
    use std::{cell::RefCell, rc::Rc};

    let seen = Rc::new(RefCell::new(Vec::new()));
    let make_system = |index| {
        let seen = Rc::clone(&seen);
        move |mut cmd: Commands<'_>, query: Query<'_, (Entity, &Position)>| {
            for (entity, position) in &query {
                seen.borrow_mut().push((index, position.0));
                cmd.set_component(entity, Position(index, 0));
            }
        }
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
#[should_panic(expected = "cannot merge a UniverseBuilder with issued handles or queued commands")]
fn composed_builders_reject_retained_other_handle() {
    let other = Universe::builder();
    let _handle = other.handle();
    let _ = Universe::builder().with_other(other);
}

#[test]
#[should_panic(expected = "cannot merge a UniverseBuilder with issued handles or queued commands")]
fn composed_builders_reject_queued_other_commands() {
    let other = Universe::builder();
    let mut cmd = other.handle().command_buffer();
    cmd.create_world(World::builder("would disappear"));
    cmd.submit();
    let _ = Universe::builder().with_other(other);
}

#[test]
fn destroyed_world_rejects_later_spawns_and_transfers() {
    let mut universe = Universe::builder().build();
    let mut cmd = universe.handle().command_buffer();
    let destroyed = cmd.create_world(World::builder("destroyed"));
    let surviving = cmd.create_world(World::builder("surviving"));
    let existing = cmd.spawn(surviving, Entity::builder());
    cmd.destroy_world(destroyed);
    let rejected = cmd.spawn(destroyed, Entity::builder());
    cmd.send(existing, destroyed);
    cmd.submit();
    universe.tick(0.0);

    assert!(universe.world(destroyed).is_none());
    assert!(!universe.is_alive(rejected));
    assert_eq!(universe.get_world(existing), Some(surviving));
    assert_eq!(universe.alive_count(), 1);
}

#[test]
fn despawn_discards_later_component_writes() {
    let mut universe = Universe::builder().build();
    let mut cmd = universe.handle().command_buffer();
    let world = cmd.create_world(World::builder("world"));
    let entity = cmd.spawn(world, Entity::builder().with_component(Position(1, 2)));
    cmd.despawn(entity);
    cmd.set_component(entity, Hidden);
    cmd.submit();
    universe.tick(0.0);

    assert!(!universe.is_alive(entity));
    assert!(universe.component::<Position>(entity).is_none());
    assert!(universe.component::<Hidden>(entity).is_none());
}

#[test]
fn systems_run_once_per_tick_even_without_entities_and_share_lifecycle() {
    use std::{cell::RefCell, rc::Rc};
    let observed = Rc::new(RefCell::new(Vec::new()));
    let mut builder = Universe::builder();
    for id in [1, 2] {
        let observed = Rc::clone(&observed);
        let mut calls = 0;
        builder = builder.with_system(move |lifecycle: Lifecycle<'_>, DeltaTime(delta_time)| {
            calls += 1;
            observed
                .borrow_mut()
                .push((id, calls, lifecycle.iter().count(), delta_time));
        });
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
fn structural_records_preserve_order_and_only_record_successful_removals() {
    use dirk_universe::lifecycle::LifecycleEvent;
    let mut universe = Universe::builder().build();
    let mut cmd = universe.handle().command_buffer();
    let first = cmd.create_world(World::builder("first"));
    let second = cmd.create_world(World::builder("second"));
    let entity = cmd.spawn(first, Entity::builder().with_component(Position(1, 0)));
    cmd.set_component(entity, Position(2, 0));
    cmd.remove_component::<Position>(entity);
    cmd.remove_component::<Position>(entity);
    cmd.set_component(entity, Position(3, 0));
    cmd.send(entity, second);
    cmd.destroy_world(second);
    cmd.despawn(entity);
    cmd.destroy_world(first);
    cmd.submit();
    universe.tick(0.0);
    assert_eq!(
        universe.lifecycle().copied().collect::<Vec<_>>(),
        vec![
            LifecycleEvent::WorldCreated { world: first },
            LifecycleEvent::WorldCreated { world: second },
            LifecycleEvent::EntitySpawned {
                entity,
                world: first
            },
            LifecycleEvent::ComponentRemoved {
                entity,
                type_id: std::any::TypeId::of::<Position>()
            },
            LifecycleEvent::EntityMoved {
                entity,
                from: first,
                to: second
            },
            LifecycleEvent::ComponentRemoved {
                entity,
                type_id: std::any::TypeId::of::<Position>()
            },
            LifecycleEvent::EntityDespawned {
                entity,
                world: second
            },
            LifecycleEvent::WorldDestroyed { world: second },
            LifecycleEvent::WorldDestroyed { world: first },
        ]
    );
    assert_eq!(universe.alive_count(), 0);
    universe.tick(0.0);
    assert_eq!(universe.lifecycle().count(), 0);
}

#[test]
fn removed_non_clone_components_are_released_immediately() {
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
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn additions_replacements_and_reinsertions_have_distinct_change_semantics() {
    use std::{cell::RefCell, rc::Rc};
    let seen = Rc::new(RefCell::new(Vec::new()));
    let removed_seen = Rc::new(RefCell::new(Vec::new()));
    let observer_seen = seen.clone();
    let observer_removed = removed_seen.clone();
    let mut universe = Universe::builder()
        .with_system(
            move |added: Query<&Counter, Added<Counter>>,
                  changed: Query<&Counter, Changed<Counter>>,
                  removed: RemovedComponents<Counter>| {
                observer_seen.borrow_mut().push((
                    added.iter().map(|value| value.0).collect::<Vec<_>>(),
                    changed.iter().map(|value| value.0).collect::<Vec<_>>(),
                ));
                observer_removed
                    .borrow_mut()
                    .push(removed.iter().collect::<Vec<_>>());
            },
        )
        .build();
    let mut cmd = universe.handle().command_buffer();
    let world = cmd.create_world(World::builder("w"));
    let entity = cmd.spawn(world, Entity::builder().with_component(Counter(1)));
    cmd.submit();
    universe.tick(0.0);
    let mut cmd = universe.handle().command_buffer();
    cmd.set_component(entity, Counter(2));
    cmd.set_component(entity, Counter(3));
    cmd.submit();
    universe.tick(0.0);
    let mut cmd = universe.handle().command_buffer();
    cmd.remove_component::<Counter>(entity);
    cmd.set_component(entity, Counter(4));
    cmd.submit();
    universe.tick(0.0);
    let mut cmd = universe.handle().command_buffer();
    cmd.despawn(entity);
    cmd.submit();
    universe.tick(0.0);
    universe.tick(0.0);
    assert_eq!(
        *seen.borrow(),
        vec![
            (vec![1], vec![1]),
            (vec![], vec![3]),
            (vec![4], vec![4]),
            (vec![], vec![]),
            (vec![], vec![]),
        ]
    );
    assert_eq!(
        *removed_seen.borrow(),
        vec![vec![], vec![], vec![entity], vec![entity], vec![]]
    );
}

#[test]
fn change_detection_survives_an_observer_skipping_ticks() {
    use std::{cell::RefCell, rc::Rc};
    let seen = Rc::new(RefCell::new(Vec::new()));
    let observer_seen = seen.clone();
    let mut observer = (move |query: Query<&Counter, Changed<Counter>>| {
        observer_seen
            .borrow_mut()
            .push(query.iter().map(|counter| counter.0).collect::<Vec<_>>());
    })
    .to_system();
    let mut universe = Universe::builder()
        .with_world(World::builder("w").with_entity(Entity::builder().with_component(Counter(0))))
        .with_system(|mut query: Query<&mut Counter>| {
            for mut counter in &mut query {
                counter.0 += 1;
            }
        })
        .build();
    let commands = RefCell::new(universe.handle().command_buffer());
    universe.tick(0.0);
    observer.run(&universe, 0.0, &commands);
    for _ in 0..3 {
        universe.tick(0.0);
    }
    observer.run(&universe, 0.0, &commands);
    observer.run(&universe, 0.0, &commands);
    assert_eq!(*seen.borrow(), vec![vec![1], vec![4], vec![]]);
}

#[test]
fn mutable_iteration_and_lookups_edit_live_values_in_one_system() {
    let mut universe = Universe::builder()
        .with_system(
            |mut query: Query<(Entity, &mut Counter, &Step), Without<Hidden>>| {
                let world = WorldId::default();
                let entities: Vec<_> = query
                    .iter_in_world_mut(world)
                    .map(|(entity, mut counter, step)| {
                        counter.0 += step.0;
                        entity
                    })
                    .collect();
                for entity in entities {
                    let (_, mut counter, _) = query.get_mut(entity).unwrap();
                    assert_eq!(counter.0, 2);
                    counter.0 += 3;
                }
            },
        )
        .build();
    let mut cmd = universe.handle().command_buffer();
    let first = cmd.create_world(World::builder("first"));
    let second = cmd.create_world(World::builder("second"));
    let builder = || {
        Entity::builder()
            .with_component(Counter(0))
            .with_component(Step(2))
    };
    let included = cmd.spawn(first, builder());
    let hidden = cmd.spawn(first, builder().with_component(Hidden));
    let other_world = cmd.spawn(second, builder());
    cmd.submit();
    universe.tick(0.0);
    assert_eq!(universe.component::<Counter>(included).unwrap().0, 5);
    assert_eq!(universe.component::<Counter>(hidden).unwrap().0, 0);
    assert_eq!(universe.component::<Counter>(other_world).unwrap().0, 0);
    let query = Query::<(Entity, &Counter), Without<Hidden>>::new(&universe);
    assert!(query.get(hidden).is_none());
    assert_eq!(query.iter_in_world(first).count(), 1);
}
