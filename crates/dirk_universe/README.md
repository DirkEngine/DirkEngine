# universe

`dirk_universe` is `DirkEngine`'s entity-component system. A `Universe` contains
worlds, entities, components, and systems. Systems run once per tick,
including when their queries are empty, in an order derived from the data they
access.

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
    .build()
    .expect("systems should schedule");
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
assert!(universe.is_ok());
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
Overlapping read/write or write/write borrows within one system are rejected
by `UniverseBuilder::build`, conservatively even when filters would make them
disjoint.

`Added<C>` matches components attached since the observing system last ran.
`Changed<C>` matches additions, replacements, and mutable dereferences since
that system last ran. Merely fetching or reading a mutable component does not
mark it changed. Assigning the same value does; there is no value comparison.
Interior mutation through types such as `Cell` is not automatically tracked.

Each system has its own execution counter. Reading changes never consumes them
for another system. Multiple edits between observations
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
assert!(universe.is_ok());
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

## Scheduling

Nobody orders systems by hand. A system that reads a component, whether through
`&C`, `Delta<C>` or a filter such as `Changed<C>`, runs after every system that
writes `C`, so it sees this tick's values. `Lagged<D>` reads `D` before this
tick's systems write it instead: its system runs before the writers, which also
breaks dependency cycles. Commands applied at the start of the tick are already
visible, and an explicit order placing a writer first takes precedence.
Dependent systems are ordered the same whatever order plugins merge their
builders in; unrelated systems keep their registration order.

```rust
use dirk_universe::prelude::*;

#[derive(Debug, Component)]
struct Velocity(f64);
#[derive(Debug, Component)]
struct Position(f64);

fn render(positions: Query<&Position>) { /* runs second */ }
fn movement(mut query: Query<(&Velocity, &mut Position)>) { /* runs first */ }

let universe = Universe::builder()
    .with_system(render)
    .with_system(movement)
    .build()?;
assert_eq!(universe.schedule().systems()[1].name, std::any::type_name_of_val(&render));
# Ok::<(), dirk_universe::schedule::ScheduleError>(())
```

Two systems writing the same component must be ordered explicitly, unless
other dependencies already order them. `.before(other)` and `.after(other)`
name functions and closures directly, and `System` types through
`MySystem::label()`. Closures made by one factory share a type, so they share a
label. Explicit orders replace derived ones between the same pair.

`UniverseBuilder::build` reports every problem as a `ScheduleError`: borrows
that conflict within one system, writers in no defined order, dependency
cycles (naming each system and component involved), and orders against
systems that were never registered. `universe.schedule()` returns the final
order with the reason for each dependency, and displays as a readable list.

Commands do not order systems, since their changes apply on the next tick.
Systems run sequentially. `UniverseBuilder::with_other` appends another
builder's systems; builders with outstanding handles or queued commands cannot
be merged.
