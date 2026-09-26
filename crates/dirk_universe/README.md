# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Submitted command buffers queue
changes for the next call to `Universe::tick`.

The query and system APIs use Rust types to describe the
data each system needs. `&C` reads a component and `&mut C` edits it; both
skip entities without that component. Tuples fetch multiple components.
`With<C>` and `Without<C>` filter by
component presence; tuples combine filters with AND. The default filter, `()`,
accepts every entity.

```rust
use dirk_universe::{
    Entity, Universe, World,
    components::Component,
    query::{Query, QueryItem, filter::Without},
    systems::DeltaTime,
};

#[derive(Debug, Clone, Component)]
struct Position(f64);

#[derive(Debug, Component)]
struct Velocity(f64);

#[derive(Debug, Component)]
struct Frozen;

let movement = |query: Query<'_, (&Velocity, &mut Position), Without<Frozen>>,
                DeltaTime(delta_time)| {
    let (velocity, mut position) = query.into_params();
    position.0 += velocity.0 * delta_time;
};

let mut universe = Universe::builder()
    .with_world(World::builder("simulation").with_entity(
        Entity::builder()
            .with_component(Position(0.0))
            .with_component(Velocity(2.0)),
    ))
    .with_system(movement)
    .build();

universe.tick(0.5); // Creates the entity and updates its position.
let entity = QueryItem::<&Position>::iter(&universe)
    .next().unwrap().entity();
assert_eq!(universe.component::<Position>(entity).unwrap().0, 1.0);

universe.tick(0.5);
assert_eq!(universe.component::<Position>(entity).unwrap().0, 2.0);
```

`UniverseBuilder::with_system` accepts functions and `FnMut` closures directly.
Their argument types declare the queries, changes, and delta time they need;
the universe supplies those values before each run. `Query<&C>` reads a
component and `Query<&mut C>` edits one; mutable components must implement
`Clone` so the change log retains their prior value. Stateful types implement
`System<Params>`, for example:

```rust
use dirk_universe::{components::Component, query::Query, systems::{DeltaTime, System}};

#[derive(Debug, Clone, Component)]
struct Position(f64);

struct Movement { speed: f64 }

impl System<(Query<'_, &mut Position>, DeltaTime)> for Movement {
    fn run(&mut self, (query, DeltaTime(dt)): (Query<'_, &mut Position>, DeltaTime)) {
        query.into_params().0 += self.speed * dt;
    }
}
```

The engine invokes a query system once per matching entity. Multiple `Query`
parameters must match the same entity (intersection); no matches means no calls.
Entity order is unspecified. Systems without `Query` parameters run once per
tick, including in an empty universe. A `QueryView<&C>` provides read-only
iteration and entity lookups for aggregate or lifecycle work without triggering
per-entity invocation. It supports `iter_in_world` for a runtime world ID.

This differs from Bevy's execution model: Bevy calls a system once and lets it
iterate its query. Here, the engine owns that loop. Put work that must happen
once per tick in a separate system without a `Query` parameter.

Mutable query edits are visible to later systems in the same tick. Structural
changes use an optional `Commands` parameter and take effect on the next tick.
Overlapping read/write or write/write component access is rejected when the
system is registered, including across `Query` and `QueryView` parameters.

Systems run sequentially in registration order. `UniverseBuilder::with_other`
appends the other builder's systems.

Lifecycle and component changes use this same execution path. `Universe::changes`
returns the current tick's ordered log; `Universe::component_changes::<C>` returns
typed additions, updates, and removals. Updates carry both old and new values,
and removed values remain readable even when the entity no longer exists.

```rust
use dirk_universe::{
    Universe, components::Component, changes::ComponentChange, systems::Changes,
};

#[derive(Debug, Component)]
struct Health(u32);

let observer = |changes: Changes<'_>| {
    for change in changes.components::<Health>() {
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
};
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

The old system traits and experimental module paths have been removed. Declare
queries and changes in each system's signature. Mutable query edits are
coalesced into an update visible in the following tick's change log. Parallel
execution is outside this API.
