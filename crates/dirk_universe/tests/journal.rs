//! Tests for the editor journal.
#![cfg(feature = "journal")]

use std::cell::RefCell;

use dirk_universe::{journal::ChangeKind, prelude::*};

#[derive(Debug, Clone, PartialEq, Component)]
struct Health(i32);

/// Builds a journaled universe with one world.
fn journaled(capacity: usize) -> (Universe, WorldId) {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .with_journal(capacity)
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let world = universe.worlds().next().expect("one world").id();
    (universe, world)
}

/// Applies `edit` through an undoable command buffer, then ticks once.
fn apply<T>(universe: &mut Universe, edit: impl FnOnce(&mut CommandBuffer) -> T) -> T {
    let mut cmd = universe.handle().command_buffer().undoable();
    let result = edit(&mut cmd);
    cmd.submit();
    universe.tick(0.0);
    result
}

fn health(universe: &Universe, entity: Entity) -> Option<i32> {
    universe.component::<Health>(entity).map(|health| health.0)
}

fn entry_count(universe: &Universe) -> usize {
    universe
        .journal()
        .expect("journal enabled")
        .entries()
        .count()
}

#[test]
fn each_undoable_buffer_records_one_entry() {
    let (mut universe, world) = journaled(16);
    // Building the universe is not an edit.
    assert_eq!(entry_count(&universe), 0);
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(10)))
    });
    universe.tick(0.0);
    apply(&mut universe, |cmd| cmd.set_component(entity, Health(5)));

    assert_eq!(entry_count(&universe), 2);
    let journal = universe.journal().expect("journal enabled");
    let kinds: Vec<Vec<_>> = journal
        .entries()
        .map(|entry| {
            let mut kinds: Vec<_> = entry
                .changes()
                .map(|change| (change.component(), change.kind()))
                .collect();
            kinds.sort_by_key(|(component, _)| *component);
            kinds
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            vec![
                (std::any::type_name::<InWorld>(), ChangeKind::Added),
                (std::any::type_name::<Health>(), ChangeKind::Added),
            ],
            vec![(std::any::type_name::<Health>(), ChangeKind::Changed)],
        ]
    );
}

#[test]
fn undo_and_redo_step_through_values_and_structure() {
    let (mut universe, world) = journaled(16);
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(10)))
    });
    apply(&mut universe, |cmd| cmd.set_component(entity, Health(5)));
    apply(&mut universe, |cmd| cmd.despawn(entity));
    assert!(!universe.is_alive(entity));

    apply(&mut universe, CommandBuffer::undo);
    assert!(universe.is_alive(entity));
    assert_eq!(universe.get_world(entity), Some(world));
    assert_eq!(health(&universe, entity), Some(5));

    apply(&mut universe, CommandBuffer::undo);
    assert_eq!(health(&universe, entity), Some(10));

    apply(&mut universe, CommandBuffer::undo);
    assert!(!universe.is_alive(entity));

    apply(&mut universe, CommandBuffer::redo);
    apply(&mut universe, CommandBuffer::redo);
    assert_eq!(health(&universe, entity), Some(5));
    let journal = universe.journal().expect("journal enabled");
    assert!(journal.can_undo() && journal.can_redo());
}

#[test]
fn systems_observe_restored_values_as_changes() {
    thread_local! {
        static SEEN: RefCell<Vec<Option<i32>>> = const { RefCell::new(Vec::new()) };
    }
    fn observe(healths: Query<(Entity, Delta<Health>)>) {
        for (_, delta) in &healths {
            SEEN.with_borrow_mut(|seen| seen.push(delta.value().map(|health| health.0)));
        }
    }
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .with_journal(16)
        .with_system(observe)
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let world = universe.worlds().next().expect("one world").id();
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(1)))
    });
    apply(&mut universe, |cmd| cmd.remove_component::<Health>(entity));
    apply(&mut universe, CommandBuffer::undo);
    apply(&mut universe, CommandBuffer::redo);

    assert_eq!(
        SEEN.with_borrow(Clone::clone),
        vec![Some(1), None, Some(1), None]
    );
}

#[test]
fn new_changes_discard_undone_entries() {
    let (mut universe, world) = journaled(16);
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(1)))
    });
    apply(&mut universe, |cmd| cmd.set_component(entity, Health(2)));
    apply(&mut universe, CommandBuffer::undo);
    assert!(universe.journal().expect("journal enabled").can_redo());

    apply(&mut universe, |cmd| cmd.set_component(entity, Health(3)));
    let journal = universe.journal().expect("journal enabled");
    assert!(!journal.can_redo());

    apply(&mut universe, CommandBuffer::redo);
    assert_eq!(health(&universe, entity), Some(3));
}

#[test]
fn undoing_a_world_destruction_restores_its_entities() {
    let (mut universe, world) = journaled(16);
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(7)))
    });
    apply(&mut universe, |cmd| cmd.destroy_world(world));
    assert!(universe.world(world).is_none());

    apply(&mut universe, CommandBuffer::undo);

    assert_eq!(
        universe.world(world).map(|world| world.name().to_owned()),
        Some("w".to_owned())
    );
    assert!(universe.is_in_world(world, entity));
    assert_eq!(health(&universe, entity), Some(7));
}

#[test]
fn the_history_keeps_the_latest_entries() {
    let (mut universe, world) = journaled(2);
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(0)))
    });
    for value in 1..=3 {
        apply(&mut universe, |cmd| {
            cmd.set_component(entity, Health(value))
        });
    }
    assert_eq!(entry_count(&universe), 2);

    apply(&mut universe, CommandBuffer::undo);
    apply(&mut universe, CommandBuffer::undo);
    apply(&mut universe, CommandBuffer::undo);
    assert_eq!(health(&universe, entity), Some(1));
}

#[test]
fn changes_by_systems_and_other_buffers_are_not_recorded() {
    #[derive(Debug, Clone, Component)]
    struct Age(u32);
    fn grow(mut ages: Query<&mut Age>) {
        for mut age in &mut ages {
            age.0 += 1;
        }
    }
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .with_journal(16)
        .with_system(grow)
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let world = universe.worlds().next().expect("one world").id();
    let entity = apply(&mut universe, |cmd| {
        cmd.spawn(
            world,
            Entity::builder()
                .with_component(Health(1))
                .with_component(Age(0)),
        )
    });
    apply(&mut universe, |cmd| cmd.set_component(entity, Health(2)));
    apply(&mut universe, CommandBuffer::undo);
    let mut plain = universe.handle().command_buffer();
    plain.spawn(world, Entity::builder());
    plain.submit();
    for _ in 0..3 {
        universe.tick(0.0);
    }

    let journal = universe.journal().expect("journal enabled");
    assert_eq!(journal.entries().count(), 1);
    assert!(journal.can_redo());
    apply(&mut universe, CommandBuffer::redo);
    assert_eq!(health(&universe, entity), Some(2));
    assert!(
        universe
            .component::<Age>(entity)
            .is_some_and(|age| age.0 > 3)
    );
}

#[test]
fn restoring_isolated_worlds_stays_hidden_from_shared_queries() {
    thread_local! {
        static SHARED: RefCell<Vec<Entity>> = const { RefCell::new(Vec::new()) };
    }
    fn observe(
        healths: Query<(Entity, Delta<Health>)>,
        locations: Query<(Entity, Delta<InWorld>)>,
    ) {
        let seen = healths.iter().map(|(entity, _)| entity);
        let moved = locations.iter().map(|(entity, _)| entity);
        SHARED.with_borrow_mut(|shared| shared.extend(seen.chain(moved)));
    }
    let mut universe = Universe::builder()
        .with_journal(16)
        .with_system(observe)
        .build()
        .expect("systems should schedule");
    let world = apply(&mut universe, |cmd| {
        cmd.create_world(World::builder("hidden").isolated())
    });
    apply(&mut universe, |cmd| {
        cmd.spawn(world, Entity::builder().with_component(Health(3)));
    });
    apply(&mut universe, |cmd| cmd.destroy_world(world));
    apply(&mut universe, CommandBuffer::undo);
    apply(&mut universe, CommandBuffer::redo);
    apply(&mut universe, CommandBuffer::undo);

    assert!(universe.world(world).is_some());
    assert!(SHARED.with_borrow(Vec::is_empty));
}
