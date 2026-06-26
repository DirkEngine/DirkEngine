#![cfg(test)]

use std::any::TypeId;

use crate::{
    Entity, EntityBuilder, Universe, World, WorldId,
    components::Component,
    query::{
        Query,
        experimental::{QueryItem, Read},
        filter::{Not, With},
    },
    systems::experimental::{StandaloneSystem, filtered_system, system as default_system},
};

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Health(u32);

#[derive(Debug, serde::Serialize, serde::Deserialize, Component)]
struct Mana(u32);

fn spawn_entity(universe: &mut Universe, world: WorldId, builder: EntityBuilder) -> Entity {
    let mut command_buffer = universe.handle().command_buffer();
    let entity = command_buffer.spawn(world, builder);
    command_buffer.submit();
    universe.tick(0.0);
    entity
}

#[test]
fn allocator_clones_share_world_and_entity_sequences() {
    let allocator = crate::Allocator::new();
    let clone = allocator.clone();

    assert_eq!(allocator.allocate_world().raw(), 0);
    assert_eq!(clone.allocate_world().raw(), 1);
    assert_eq!(allocator.allocate_entity().raw(), 0);
    assert_eq!(clone.allocate_entity().raw(), 1);
}

#[test]
fn builder_creates_worlds_and_initial_entities() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha").with_entity(Entity::builder()))
        .with_world(World::builder("beta").with_entity(Entity::builder()))
        .build();
    universe.tick(0.0);

    assert_eq!(universe.alive_count(), 2);
    assert_eq!(
        universe.world(crate::WorldId::default()).map(World::name),
        Some("alpha")
    );
    assert_eq!(
        universe
            .world(crate::WorldId::default() + 1)
            .map(World::name),
        Some("beta")
    );
}

#[test]
fn spawn_entity_in_missing_world_is_ignored() {
    let mut universe = Universe::builder().build();
    universe.tick(0.0);
    let missing = crate::WorldId::default() + 99;

    let spawned = spawn_entity(&mut universe, missing, Entity::builder());

    assert!(!universe.is_alive(spawned));
    assert_eq!(universe.alive_count(), 0);
}

#[test]
fn query_filters_by_components_and_world_membership() {
    let mut universe = Universe::builder()
        .with_world(World::builder("a"))
        .with_world(World::builder("b"))
        .build();
    universe.tick(0.0);

    let world_a = crate::WorldId::default();
    let world_b = world_a + 1;

    let e1 = spawn_entity(
        &mut universe,
        world_a,
        Entity::builder().with_component(Health(10)),
    );
    let e2 = spawn_entity(
        &mut universe,
        world_b,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(5)),
    );

    let q = Query::empty()
        .with_component::<Health>()
        .without_component::<Mana>()
        .with_world(world_a)
        .without_world(world_b);

    assert!(q.matches(&universe, e1));
    assert!(!q.matches(&universe, e2));
}

#[test]
fn component_getter_returns_expected_values() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    let e = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(123)),
    );

    assert_eq!(universe.component::<Health>(e).map(|h| h.0), Some(123));
    assert_eq!(universe.component::<Mana>(e).map(|m| m.0), None);
}

#[test]
fn worlds_returns_all_live_worlds() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let mut worlds: Vec<_> = universe
        .worlds()
        .map(|world| (world.id().raw(), world.name().to_owned()))
        .collect();
    worlds.sort_by_key(|(id, _)| *id);

    assert_eq!(
        worlds,
        vec![(0, "alpha".to_owned()), (1, "beta".to_owned())]
    );
}

#[test]
fn entities_returns_live_entity_world_pairs() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let first_world = crate::WorldId::default();
    let second_world = first_world + 1;
    let first = spawn_entity(&mut universe, first_world, Entity::builder());
    let second = spawn_entity(&mut universe, second_world, Entity::builder());

    let mut entities: Vec<_> = universe
        .entities()
        .map(|(entity, world)| (entity.raw(), world.raw()))
        .collect();
    entities.sort_by_key(|(entity, _)| *entity);

    assert_eq!(
        entities,
        vec![
            (first.raw(), first_world.raw()),
            (second.raw(), second_world.raw())
        ]
    );
}

#[test]
fn entities_in_world_filters_correctly() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let first_world = crate::WorldId::default();
    let second_world = first_world + 1;
    let first = spawn_entity(&mut universe, first_world, Entity::builder());
    let _second = spawn_entity(&mut universe, second_world, Entity::builder());

    let entities: Vec<_> = universe
        .entities_in_world(first_world)
        .map(Entity::raw)
        .collect();

    assert_eq!(entities, vec![first.raw()]);
}

#[test]
fn component_infos_exposes_type_name_and_debug_value() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();
    let entity = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(77)),
    );

    let infos: Vec<_> = universe.component_infos(entity).collect();

    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].type_id, TypeId::of::<Health>());
    assert_eq!(infos[0].type_name, std::any::type_name::<Health>());
    assert_eq!(format!("{:?}", infos[0].debug), "Health(77)");
}

#[test]
fn inspection_helpers_update_after_despawn_and_world_destruction() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let first_world = crate::WorldId::default();
    let second_world = first_world + 1;
    let despawned = spawn_entity(
        &mut universe,
        first_world,
        Entity::builder().with_component(Health(10)),
    );
    let destroyed = spawn_entity(
        &mut universe,
        second_world,
        Entity::builder().with_component(Mana(20)),
    );

    let mut command_buffer = universe.handle().command_buffer();
    command_buffer.despawn(despawned);
    command_buffer.destroy_world(second_world);
    command_buffer.submit();
    universe.tick(0.016);

    assert_eq!(universe.component_infos(despawned).count(), 0);
    assert_eq!(universe.component_infos(destroyed).count(), 0);
    assert!(!universe.entities().any(|(entity, _)| entity == despawned));
    assert!(!universe.entities().any(|(entity, _)| entity == destroyed));
    assert!(!universe.worlds().any(|world| world.id() == second_world));
}

#[test]
fn query_item_iter_applies_filters() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let world_a = crate::WorldId::default();
    let world_b = world_a + 1;
    let e1 = spawn_entity(
        &mut universe,
        world_a,
        Entity::builder().with_component(Health(10)),
    );
    let _e2 = spawn_entity(
        &mut universe,
        world_b,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(5)),
    );
    let _e3 = spawn_entity(
        &mut universe,
        world_b,
        Entity::builder().with_component(Mana(15)),
    );

    let matched: Vec<_> = QueryItem::<Read<Health>, Not<Mana>>::iter(&universe)
        .map(|query| (query.entity().raw(), query.params().0))
        .collect();

    assert_eq!(matched, vec![(e1.raw(), 10)]);
}

#[test]
fn query_item_tuple_params_require_every_component() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    let _health_only = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    let both = spawn_entity(
        &mut universe,
        world,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(5)),
    );

    let matched: Vec<_> = QueryItem::<(Read<Health>, Read<Mana>)>::iter(&universe)
        .map(|query| {
            let entity = query.entity();
            let (health, mana) = query.into_params();
            (entity.raw(), health.0, mana.0)
        })
        .collect();

    assert_eq!(matched, vec![(both.raw(), 20, 5)]);
}

#[test]
fn query_item_fetch_skips_entities_missing_parameters() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    let mana_1 = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Mana(1)),
    );
    let mana_2 = spawn_entity(
        &mut universe,
        world,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(2)),
    );

    let mut matched: Vec<_> = QueryItem::<Read<Mana>>::iter(&universe)
        .map(|query| query.entity().raw())
        .collect();
    matched.sort_unstable();

    assert_eq!(matched, vec![mana_1.raw(), mana_2.raw()]);
}

#[test]
fn query_item_matches_entities_across_all_worlds() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build();
    universe.tick(0.0);

    let first_world = crate::WorldId::default();
    let second_world = first_world + 1;
    let first = spawn_entity(
        &mut universe,
        first_world,
        Entity::builder().with_component(Health(10)),
    );
    let second = spawn_entity(
        &mut universe,
        second_world,
        Entity::builder().with_component(Health(20)),
    );

    let mut matched: Vec<_> = QueryItem::<Read<Health>>::iter(&universe)
        .map(|query| query.entity().raw())
        .collect();
    matched.sort_unstable();

    assert_eq!(matched, vec![first.raw(), second.raw()]);
}

#[test]
fn query_item_on_empty_universe_yields_nothing() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);

    assert_eq!(QueryItem::<Read<Health>>::iter(&universe).count(), 0);
    assert_eq!(QueryItem::<()>::iter(&universe).count(), 0);
}

#[test]
fn query_item_excludes_despawned_entities() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    let despawned = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    assert_eq!(QueryItem::<Read<Health>>::iter(&universe).count(), 1);

    let mut command_buffer = universe.handle().command_buffer();
    command_buffer.despawn(despawned);
    command_buffer.submit();
    universe.tick(0.016);

    assert_eq!(QueryItem::<Read<Health>>::iter(&universe).count(), 0);
}

#[test]
fn with_and_not_filters_compose() {
    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    let health_only = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    let both = spawn_entity(
        &mut universe,
        world,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(5)),
    );
    let mana_only = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Mana(15)),
    );

    let mut with_health: Vec<_> = QueryItem::<(), With<Health>>::iter(&universe)
        .map(|query| query.entity().raw())
        .collect();
    with_health.sort_unstable();
    assert_eq!(with_health, vec![health_only.raw(), both.raw()]);

    let mut not_health: Vec<_> = QueryItem::<(), (Not<Health>,)>::iter(&universe)
        .map(|query| query.entity().raw())
        .collect();
    not_health.sort_unstable();
    assert_eq!(not_health, vec![mana_only.raw(),]);

    let mut everything: Vec<_> = QueryItem::<(), ()>::iter(&universe)
        .map(|query| query.entity().raw())
        .collect();
    everything.sort_unstable();
    assert_eq!(
        everything,
        vec![health_only.raw(), both.raw(), mana_only.raw()]
    );
}

#[test]
fn func_system_runs_for_matching_queries_only() {
    use std::sync::{Arc, Mutex};

    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    let _included = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    let _excluded = spawn_entity(
        &mut universe,
        world,
        Entity::builder()
            .with_component(Health(20))
            .with_component(Mana(5)),
    );

    let seen = Arc::new(Mutex::new(Vec::new()));
    let system_seen = Arc::clone(&seen);
    let system = filtered_system(move |query: QueryItem<'_, Read<Health>, Not<Mana>>| {
        system_seen
            .lock()
            .expect("seen mutex poisoned")
            .push(query.params().0);
    });

    system.run(&universe);

    let seen = seen.lock().expect("seen mutex poisoned");
    assert_eq!(*seen, vec![10]);
}

#[test]
fn default_system_helper_runs_for_every_matching_entity() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let mut universe = Universe::builder().with_world(World::builder("w")).build();
    universe.tick(0.0);
    let world = crate::WorldId::default();

    for health in [1, 2, 3] {
        spawn_entity(
            &mut universe,
            world,
            Entity::builder().with_component(Health(health)),
        );
    }
    spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Mana(4)),
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let system_calls = Arc::clone(&calls);
    let sys = default_system(move |query: QueryItem<'_, Read<Health>>| {
        system_calls.fetch_add(query.params().0 as usize, Ordering::SeqCst);
    });

    sys.run(&universe);

    assert_eq!(calls.load(Ordering::SeqCst), 6);
}
