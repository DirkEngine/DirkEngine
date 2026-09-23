# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Submitted command buffers queue
changes for the next call to `Universe::tick`.

The query and system APIs use Rust types to describe the
data each system needs. `Read<C>` borrows a component and skips entities without
it. Tuples fetch multiple components. `With<C>` and `Without<C>` filter by
component presence; tuples combine filters with AND. The default filter, `()`,
accepts every entity.

```rust
use dirk_universe::{
    Entity, Universe, World,
    components::Component,
    query::{QueryItem, Read, filter::Without},
    systems::FuncSystem,
};

#[derive(Debug, Component)]
struct Position(f64);

#[derive(Debug, Component)]
struct Velocity(f64);

#[derive(Debug, Component)]
struct Frozen;

let movement = FuncSystem::new(|commands, universe, delta_time| {
    for item in QueryItem::<(Read<Position>, Read<Velocity>), Without<Frozen>>::iter(universe) {
        let (position, velocity) = item.params();
        commands.set_component(
            item.entity(),
            Position(position.0 + velocity.0 * delta_time),
        );
    }
});

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

Every system implements the same `System` trait and registers with
`UniverseBuilder::with_system`. `FuncSystem::new` adapts functions and `FnMut`
closures to that trait. Each system runs once per tick, even in an empty
universe, and may iterate any number of typed queries. Entity iteration order
is unspecified. Use standard iterator filters with `Universe::is_in_world`
when a query should be restricted to a runtime world ID.

Systems run sequentially in registration order. `UniverseBuilder::with_other`
appends the other builder's systems. Systems read the same live state and
queue updates in the shared command buffer for the next tick.

Lifecycle and component changes use this same execution path. `Universe::changes`
returns the current tick's ordered log; `Universe::component_changes::<C>` returns
typed additions, updates, and removals. Updates carry both old and new values,
and removed values remain readable even when the entity no longer exists.

```rust
use dirk_universe::{
    Universe, components::Component, changes::ComponentChange, systems::FuncSystem,
};

#[derive(Debug, Component)]
struct Health(u32);

let observer = FuncSystem::new(|_, universe, _| {
    for change in universe.component_changes::<Health>() {
        match change {
            ComponentChange::Added { entity, component } => {
                println!("{entity:?} starts with {} health", component.0);
            }
            ComponentChange::Updated { entity, old, new } => {
                println!("{entity:?}: {} -> {} health", old.0, new.0);
            }
            ComponentChange::Removed { entity, component } => {
                println!("{entity:?} removed with {} health", component.0);
            }
        }
    }
});
let universe = Universe::builder().with_system(observer).build();
```

Command buffers are applied in submission order, with each buffer's commands
applied in insertion order. The log records every successful transition,
including repeated replacements and entities spawned and despawned in one tick.
A world creation precedes its entity spawns, component additions follow their
entity spawn, component removals precede despawn, and world destruction follows
its entity despawns. Within a bulk spawn or destruction, component/entity order
is unspecified. No-op commands produce no changes.

All systems can read the complete log without consuming it. It is cleared at
the start of the next tick. Component values are shared with the log so old
values survive without requiring `Clone`; removed resources may therefore live
until the next tick. Interior mutations through types such as `Cell` are not
tracked by the command log. Queries always inspect the final live state, while
the log describes how that state was reached.

The old system traits and experimental module paths have been removed. Move
entity loops into `System::run` or a `FuncSystem` closure, and use the change log
for lifecycle work. Keep writes in the supplied command buffer. Parallel
execution and mutable component queries are outside this API.
