# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Submitted command buffers queue
changes for the next call to `Universe::tick`.

The experimental query and function-system APIs use Rust types to describe the
data each system needs. `Read<C>` borrows a component and skips entities without
it. Tuples fetch multiple components. `With<C>` and `Without<C>` filter by
component presence; tuples combine filters with AND. The default filter, `()`,
accepts every entity.

```rust
use dirk_universe::{
    CommandBuffer, Entity, Universe, World,
    components::Component,
    query::{experimental::{QueryItem, Read}, filter::Without},
    systems::experimental::FuncSystem,
};

#[derive(Debug, Component)]
struct Position(f64);

#[derive(Debug, Component)]
struct Velocity(f64);

#[derive(Debug, Component)]
struct Frozen;

let movement = FuncSystem::new(
    |commands: &mut CommandBuffer,
     item: QueryItem<'_, (Read<Position>, Read<Velocity>), Without<Frozen>>,
     delta_time| {
        let (position, velocity) = item.params();
        commands.set_component(
            item.entity(),
            Position(position.0 + velocity.0 * delta_time),
        );
    },
);

let mut universe = Universe::builder()
    .with_world(World::builder("simulation").with_entity(
        Entity::builder()
            .with_component(Position(0.0))
            .with_component(Velocity(2.0)),
    ))
    .with_system(movement)
    .build();

universe.tick(0.5); // Creates the entity and queues its first movement.
let entity = QueryItem::<Read<Position>>::iter(&universe)
    .next().unwrap().entity();
assert_eq!(universe.component::<Position>(entity).unwrap().0, 0.0);

universe.tick(0.5); // Applies the previous tick's movement.
assert_eq!(universe.component::<Position>(entity).unwrap().0, 1.0);
```

`FuncSystem::new` infers its query and filter from the callback's `QueryItem`
argument. Callbacks implement `FnMut`, so captured state can change between
calls. Each callback runs once per matching entity; an empty universe produces
no calls, including for `QueryItem<()>`. Entity iteration order is unspecified.
Queries can also run directly with `QueryItem::iter`.

Registered function systems run sequentially after the existing universe and
ticking systems, in registration order. `UniverseBuilder::with_other` appends
the other builder's systems. Tick callbacks read the same component snapshot
and queue updates in the shared command buffer for the next tick.

These APIs remain experimental. Queries provide shared component references;
updates go through commands. Parallel scheduling and lifecycle callbacks are
not part of the function-system API yet. Existing system traits remain
available for lifecycle callbacks.
