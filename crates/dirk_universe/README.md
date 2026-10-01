# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Systems run once per tick in
registration order, including when their queries are empty.

`Query` provides iteration and entity lookups. `&C` reads a component and
`&mut C` edits it in place; neither requires `Clone`. `Option<&C>` fetches a
component when present without requiring it. Tuples fetch multiple
components, and including `Entity` fetches the entity ID. `With<C>` and
`Without<C>` filter by component presence; filter tuples combine with AND.
Multiple queries in a system are independent collections.

```rust
use dirk_universe::prelude::*;

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
let entity = universe.query::<(Entity, &Position)>().iter().next().unwrap().0;
assert_eq!(universe.component::<Position>(entity).unwrap().0, 1.0);
universe.tick(0.5);
assert_eq!(universe.component::<Position>(entity).unwrap().0, 2.0);
```

`UniverseBuilder::with_system` accepts functions and `FnMut` closures directly.
Stateful types implement `System`, naming their parameters once. `'u` stands
for the universe borrow of one run:

```rust
use dirk_universe::prelude::*;

#[derive(Debug, Component)]
struct Position(f64);
struct Movement { speed: f64 }

impl System for Movement {
    type Params<'u> = (Query<'u, &'u mut Position>, DeltaTime);

    fn run(&mut self, (mut query, DeltaTime(dt)): Self::Params<'_>) {
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
Entity order is unspecified. Outside a system, `universe.query::<D>()` and
`universe.query_filtered::<D, F>()` provide read-only access.

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

`Delta<C>` reports what happened to `C` since the observing system last ran:
`Delta::Set` with the current value, or `Delta::Removed`. It reports the net
change, so a component removed and re-added is `Set`, and one added and removed
again between two runs is not reported at all. Despawned entities report
`Removed`, which is why a `Delta` can only be combined with `Entity`; the
compiler rejects other combinations. Removals are kept until every system has
run once after them.

```rust
use dirk_universe::prelude::*;

#[derive(Debug, Component)]
struct Health(u32);

fn synchronize(health: Query<(Entity, Delta<Health>)>) {
    for (entity, delta) in &health {
        match delta {
            Delta::Set(value) => println!("{entity:?} now has {} health", value.0),
            Delta::Removed => println!("remove cached health for {entity:?}"),
        }
    }
}
let universe = Universe::builder().with_system(synchronize).build();
```

Structure is data too. A world is an entity carrying a `World` component, and
may carry any other world-level components. Every other entity carries an
`InWorld` component naming its world: spawning adds it, `send` replaces it and
despawning removes it. `Delta<World>` and `Delta<InWorld>` therefore observe
world creation and destruction and entity spawns, moves and despawns. `World`
and `InWorld` are read-only: queries can read them, but only the engine writes
them. Because worlds are entities, `Query<Entity>` includes them; add
`With<InWorld>` to match only entities inside worlds.

Structural changes use `Commands` or submitted command buffers and apply at the
start of the next tick. Buffers execute in submission order and commands in
insertion order. Destroying or despawning a world despawns its entities too.

Systems run sequentially. `UniverseBuilder::with_other` appends another
builder's systems; builders with outstanding handles or queued commands cannot
be merged. Parallel execution is outside this API.
