#![cfg(test)]

use std::any::TypeId;

use crate::{
    Entity, EntityBuilder, InWorld, Universe, World, WorldId,
    components::Component,
    query::{
        Query,
        filter::{With, Without},
    },
    systems::IntoSystem,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Component)]
struct Health(u32);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Component)]
struct Mana(u32);

/// Returns the IDs of the universe's worlds, in creation order.
fn world_ids<const N: usize>(universe: &Universe) -> [WorldId; N] {
    let mut ids: Vec<_> = universe.worlds().map(|world| world.id()).collect();
    ids.sort_by_key(|id| id.raw());
    ids.try_into().expect("unexpected world count")
}

fn spawn_entity(universe: &mut Universe, world: WorldId, builder: EntityBuilder) -> Entity {
    let mut command_buffer = universe.handle().command_buffer();
    let entity = command_buffer.spawn(world, builder);
    command_buffer.submit();
    universe.tick(0.0);
    entity
}

#[test]
fn allocator_clones_share_one_sequence_for_worlds_and_entities() {
    let allocator = crate::Allocator::new();
    let clone = allocator.clone();

    assert_eq!(allocator.allocate_world().raw(), 0);
    assert_eq!(clone.allocate_entity().raw(), 1);
    assert_eq!(clone.allocate_world().raw(), 2);
}

#[test]
fn builder_creates_worlds_and_initial_entities() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha").with_entity(Entity::builder()))
        .with_world(World::builder("beta").with_entity(Entity::builder()))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [alpha, beta] = world_ids(&universe);
    assert_eq!(universe.entities().count(), 2);
    assert_eq!(
        universe
            .world(alpha)
            .map(|w| w.name().to_owned())
            .as_deref(),
        Some("alpha")
    );
    assert_eq!(
        universe.world(beta).map(|w| w.name().to_owned()).as_deref(),
        Some("beta")
    );
    assert!(universe.is_alive(alpha.entity()));
}

#[test]
fn spawn_entity_in_missing_world_is_ignored() {
    let mut universe = Universe::builder()
        .with_world(World::builder("home"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [home] = world_ids(&universe);
    let mut cmd = universe.handle().command_buffer();
    let destroyed = cmd.create_world(World::builder("destroyed"));
    cmd.destroy_world(destroyed);
    cmd.submit();
    universe.tick(0.0);
    let ordinary = spawn_entity(&mut universe, home, Entity::builder());

    let in_destroyed = spawn_entity(&mut universe, destroyed, Entity::builder());
    let in_entity = spawn_entity(&mut universe, WorldId::new(ordinary), Entity::builder());

    assert!(!universe.is_alive(in_destroyed));
    assert!(!universe.is_alive(in_entity));
    assert_eq!(universe.entities().count(), 1);
}

#[test]
fn query_filters_by_components_and_world_membership() {
    let mut universe = Universe::builder()
        .with_world(World::builder("a"))
        .with_world(World::builder("b"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [world_a, world_b] = world_ids(&universe);

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

    let matched: Vec<_> = universe
        .query_filtered::<(Entity, &Health), Without<Mana>>()
        .iter()
        .filter(|item| {
            universe.is_in_world(world_a, item.0) && !universe.is_in_world(world_b, item.0)
        })
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(matched, vec![e1]);
    assert!(!matched.contains(&e2));
}

#[test]
fn component_getter_returns_expected_values() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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
        .build()
        .expect("systems should schedule");
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
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [first_world, second_world] = world_ids(&universe);
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
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [first_world, second_world] = world_ids(&universe);
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
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);
    let entity = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(77)),
    );

    let infos: Vec<_> = universe
        .component_infos(entity)
        .filter(|info| info.type_id != TypeId::of::<InWorld>())
        .collect();

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
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [first_world, second_world] = world_ids(&universe);
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
fn query_iter_applies_filters() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [world_a, world_b] = world_ids(&universe);
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

    let matched: Vec<_> = universe
        .query_filtered::<(Entity, &Health), Without<Mana>>()
        .iter()
        .map(|(entity, health)| (entity.raw(), health.0))
        .collect();

    assert_eq!(matched, vec![(e1.raw(), 10)]);
}

#[test]
fn query_tuple_params_require_every_component() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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

    let matched: Vec<_> = universe
        .query::<(Entity, &Health, &Mana)>()
        .iter()
        .map(|(entity, health, mana)| (entity.raw(), health.0, mana.0))
        .collect();

    assert_eq!(matched, vec![(both.raw(), 20, 5)]);
}

#[test]
fn query_fetch_skips_entities_missing_parameters() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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

    let mut matched: Vec<_> = universe
        .query::<(Entity, &Mana)>()
        .iter()
        .map(|(entity, _)| entity.raw())
        .collect();
    matched.sort_unstable();

    assert_eq!(matched, vec![mana_1.raw(), mana_2.raw()]);
}

#[test]
fn query_matches_entities_across_all_worlds() {
    let mut universe = Universe::builder()
        .with_world(World::builder("alpha"))
        .with_world(World::builder("beta"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    let [first_world, second_world] = world_ids(&universe);
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

    let mut matched: Vec<_> = universe
        .query::<(Entity, &Health)>()
        .iter()
        .map(|(entity, _)| entity.raw())
        .collect();
    matched.sort_unstable();

    assert_eq!(matched, vec![first.raw(), second.raw()]);
}

#[test]
fn query_on_empty_universe_yields_nothing() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);

    assert_eq!(universe.query::<(Entity, &Health)>().iter().count(), 0);
    assert_eq!(
        universe
            .query_filtered::<Entity, With<InWorld>>()
            .iter()
            .count(),
        0
    );
    // The world itself is an entity.
    assert_eq!(universe.query::<Entity>().iter().count(), 1);
}

#[test]
fn query_excludes_despawned_entities() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

    let despawned = spawn_entity(
        &mut universe,
        world,
        Entity::builder().with_component(Health(10)),
    );
    assert_eq!(universe.query::<(Entity, &Health)>().iter().count(), 1);

    let mut command_buffer = universe.handle().command_buffer();
    command_buffer.despawn(despawned);
    command_buffer.submit();
    universe.tick(0.016);

    assert_eq!(universe.query::<(Entity, &Health)>().iter().count(), 0);
}

#[test]
fn with_and_without_filters_compose() {
    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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
    let neither = spawn_entity(&mut universe, world, Entity::builder());

    let mut with_health: Vec<_> = universe
        .query_filtered::<Entity, With<Health>>()
        .iter()
        .map(Entity::raw)
        .collect();
    with_health.sort_unstable();
    assert_eq!(with_health, vec![health_only.raw(), both.raw()]);

    let mut without_health: Vec<_> = universe
        .query_filtered::<Entity, (With<InWorld>, Without<Health>)>()
        .iter()
        .map(Entity::raw)
        .collect();
    without_health.sort_unstable();
    assert_eq!(without_health, vec![mana_only.raw(), neither.raw()]);

    let combined: Vec<_> = universe
        .query_filtered::<Entity, (With<Health>, Without<Mana>)>()
        .iter()
        .collect();
    assert_eq!(combined, vec![health_only]);
    assert_eq!(
        universe
            .query_filtered::<Entity, (With<Health>, Without<Health>)>()
            .iter()
            .count(),
        0
    );

    let mut everything: Vec<_> = universe
        .query_filtered::<Entity, With<InWorld>>()
        .iter()
        .map(Entity::raw)
        .collect();
    everything.sort_unstable();
    assert_eq!(
        everything,
        vec![
            health_only.raw(),
            both.raw(),
            mana_only.raw(),
            neither.raw()
        ]
    );
}

#[test]
fn function_system_iterates_filtered_query() {
    use std::{cell::RefCell, rc::Rc};

    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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

    let seen = Rc::new(RefCell::new(Vec::new()));
    let system_seen = Rc::clone(&seen);
    let mut system = (move |query: Query<'_, &Health, Without<Mana>>| {
        system_seen
            .borrow_mut()
            .extend(query.iter().map(|health| health.0));
    })
    .into_system();

    let commands = std::cell::RefCell::new(universe.handle().command_buffer());
    system.run(&mut universe, 0.016, &commands);

    assert_eq!(*seen.borrow(), vec![10]);
}

#[test]
fn function_system_sums_all_matching_entities() {
    use std::{cell::Cell, rc::Rc};

    let mut universe = Universe::builder()
        .with_world(World::builder("w"))
        .build()
        .expect("systems should schedule");
    universe.tick(0.0);
    let [world] = world_ids(&universe);

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

    let total = Rc::new(Cell::new(0));
    let system_total = Rc::clone(&total);
    let mut system = (move |query: Query<'_, &Health>| {
        system_total.set(system_total.get() + query.iter().map(|health| health.0).sum::<u32>());
    })
    .into_system();

    let commands = std::cell::RefCell::new(universe.handle().command_buffer());
    system.run(&mut universe, 0.016, &commands);

    assert_eq!(total.get(), 6);
}
