# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Systems run once per tick in
registration order, including when their queries are empty.

`Query` provides iteration and entity lookups. `&C` reads a component and
`&mut C` edits it in place; neither requires `Clone`. Tuples fetch multiple
components, and including `Entity` fetches the entity ID. `With<C>` and
`Without<C>` filter by component presence; filter tuples combine with AND.
Multiple queries in a system are independent collections.

```rust
use dirk_universe::{
    Entity, Universe, World,
    components::Component,
    query::{Query, filter::Without},
    systems::DeltaTime,
};

#[derive(Debug, Component)]
struct Position(f64);
#[derive(Debug, Component)]
struct Velocity(f64);
#[derive(Debug, Component)]
struct Frozen;

fn movement(
    mut query: Query<(&Velocity, &mut Position), Without<Frozen>>,
    DeltaTime(dt): DeltaTime,
) {
    for (velocity, mut position) in &mut query {
        position.0 += velocity.0 * dt;
    }
}

let mut universe = Universe::builder()
    .with_world(World::builder("simulation").with_entity(
        Entity::builder().with_component(Position(0.0)).with_component(Velocity(2.0)),
    ))
    .with_system(movement)
    .build();
universe.tick(0.5);
let entity = Query::<(Entity, &Position)>::new(&universe).iter().next().unwrap().0;
assert_eq!(universe.component::<Position>(entity).unwrap().0, 1.0);
universe.tick(0.5);
assert_eq!(universe.component::<Position>(entity).unwrap().0, 2.0);
```

`UniverseBuilder::with_system` accepts functions and `FnMut` closures directly.
Stateful types can also implement `System<Params>`:

```rust
use dirk_universe::{Universe, components::Component, query::Query, systems::{DeltaTime, System}};

#[derive(Debug, Component)]
struct Position(f64);
struct Movement { speed: f64 }

impl System<(Query<'_, &mut Position>, DeltaTime)> for Movement {
    fn run(&mut self, (mut query, DeltaTime(dt)): (Query<'_, &mut Position>, DeltaTime)) {
        for mut position in &mut query {
            position.0 += self.speed * dt;
        }
    }
}
let universe = Universe::builder().with_system(Movement { speed: 2.0 }).build();
```

Use `iter()` and `get(entity)` for read-only queries, or `iter_mut()` and
`get_mut(entity)` for mutable queries. `&query` and `&mut query` also support
iteration. `iter_in_world` and `iter_in_world_mut` restrict iteration to a world.
Entity order is unspecified. Outside a system, `Query::new(&universe)` provides
read-only access.

Component reads return `Ref<C>` guards and writes return `ComponentMut<C>`
guards, which dereference to the component. This keeps storage safe without
unsafe code. Guards must be dropped before borrowing the same component
incompatibly. Mutable query borrows are tied to the query borrow, preventing
simultaneous iteration and another mutable lookup through that query.
Overlapping read/write or write/write declarations are rejected at system
registration, conservatively even when filters would make them disjoint.

`Added<C>` matches components attached since the observing system last ran.
`Changed<C>` matches additions, replacements, and mutable dereferences since
that system last ran. Merely fetching or reading a mutable component does not
mark it changed. Assigning the same value does; there is no value comparison.
Interior mutation through types such as `Cell` is not automatically tracked.

Each system has its own execution counter. Reading changes never consumes them
for another system. A system after a writer sees the edit in the same tick; a
system before it sees it on its next run. Multiple edits between observations
produce one match containing the current value. No historical values are kept.
Replacing an existing component is changed but not added; removing and
reinserting it is both. On a system's first run, all present matching components
count as added and changed. Outside a system, these filters also match all
present components because there is no previous invocation.

```rust
use dirk_universe::{
    Entity, Universe, components::Component,
    query::{Query, filter::Changed}, systems::RemovedComponents,
};

#[derive(Debug, Component)]
struct Health(u32);

fn synchronize(health: Query<(Entity, &Health), Changed<Health>>, removed: RemovedComponents<Health>) {
    // Clean up first: a removed component may have been reinserted this tick.
    for entity in removed.iter() {
        println!("remove cached health for {entity:?}");
    }
    for (entity, value) in &health {
        println!("{entity:?} now has {} health", value.0);
    }
}
let universe = Universe::builder().with_system(synchronize).build();
```

Structural changes use `Commands` or submitted command buffers and apply at the
start of the next tick. Buffers execute in submission order and commands in
insertion order. `RemovedComponents<C>` yields IDs for successful removals,
including despawns, without retaining removed values. Every system may read
these records; unlike component counters, removal records expire next tick.

`Lifecycle` exposes the current tick's ordered world creation/destruction,
entity spawn/move/despawn, and component removal notifications. Records contain
only IDs and component types. World creation precedes entity spawn; component
removals precede despawn; world destruction follows its entity despawns. Bulk
entity/component order is unspecified. No-op commands produce no records.
Ordinary component additions and updates are observed through query filters.
Consumers replay structure before synchronizing current component values, so
transient entities and remove/reinsert sequences are handled correctly.

Systems run sequentially. `UniverseBuilder::with_other` appends another
builder's systems; builders with outstanding handles or queued commands cannot
be merged. Parallel execution is outside this API.
