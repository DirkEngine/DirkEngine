# `events` — Engine Event System

A lightweight, type-safe, **channel-backed** pub/sub event bus designed for
use inside a game-engine loop.

## Core Concepts

| Term | Type | Role |
|------|------|------|
| **Event** | any `T: [events::Event]` | A value that travels through the bus |
| **Dispatcher** | [`Dispatcher<T>`] | Queues events for immediate background routing |
| **Consumer** | [`Consumer<T>`] | Reads events that have already been routed |
| **Event Manager** | [`EventManager`] | Wires everything together |

## Quick Start

```rust
use dirk_events::{Event, EventManager};

// 1. Define your event type.
#[derive(Debug, Clone, Event)]
#[event("player scored {points} points")]
struct PlayerScored { points: u32 }

// 2. Create the shared manager.
let workers = dirk_threads::WorkerPool::new("pool");
let mgr = EventManager::new(workers);

// 3. Obtain a dispatcher (producer side) and a consumer (subscriber side).
let dispatcher = mgr.register::<PlayerScored>();
let mut consumer   = mgr.subscribe::<PlayerScored>();

// 4. Queue an event from anywhere that holds the dispatcher.
dispatcher.dispatch(PlayerScored { points: 42 });

// 5. Wait for the event; routing runs on a worker thread.
let event = consumer.consume_blocking().expect("event should arrive");
println!("score update: {}", event.debug()); // "player scored 42 points"
```

## Lifecycle & Delivery Guarantees

```text
 dispatcher.dispatch(e)
       │
       ▼
 [background worker queue]
       │
 [worker thread routes event]
       │
       ├──▶ consumer A
       ├──▶ consumer B
       └──▶ consumer C       ← every live consumer receives a clone
```

* **Immediate background routing** — events are forwarded as soon as a worker
  thread can route them; the game thread does not perform the fan-out work.
* **Fan-out** — every active [`Consumer`] for a given type receives its own
  independent clone of each event.
* **Type-isolated** — consumers only receive events of the exact type they
  subscribed to; other event types are never delivered to them.
* **Dropped-consumer pruning** — if a [`Consumer`] is dropped, its entry is
  removed immediately.
* **Subscription timing** — a consumer receives events dispatched after it
  subscribes, even if an older event is still waiting in a producer queue.
* **Producer closure** — after the final dispatcher for an event type is
  dropped, consumers receive any queued events and then `consume` and
  `consume_blocking` return `None`.

## Using the `#[derive(Event)]` Macro

The [`Event`] derive macro implements the [`Event`] trait for you and
lets you customise the string returned by [`Event::debug`] via an optional
`#[event("…")]` attribute.

### No attribute — falls back to `{self:?}`

```rust
use dirk_events::Event;

#[derive(Debug, Clone, Event)]
struct RawEvent { id: u32 }

assert!(RawEvent { id: 7 }.debug().contains("RawEvent"));
```

### Named-field struct — reference fields by name

```rust
use dirk_events::Event;

#[derive(Debug, Clone, Event)]
#[event("moved to ({x}, {y})")]
struct EntityMoved { x: f32, y: f32 }

assert_eq!(EntityMoved { x: 1.0, y: 2.5 }.debug(), "moved to (1, 2.5)");
```

### Tuple struct — reference fields by zero-based index

```rust
use dirk_events::Event;

#[derive(Debug, Clone, Event)]
#[event("key pressed: code={0}")]
struct KeyPressed(u32);

assert_eq!(KeyPressed(65).debug(), "key pressed: code=65");
```

### Enum — annotate each variant individually

```rust
use dirk_events::Event;

#[derive(Debug, Clone, Event)]
enum NetworkEvent {
    #[event("connected to {host}:{port}")]
    Connected { host: String, port: u16 },

    #[event("disconnected")]
    Disconnected,

    #[event("packet received: {0} bytes")]
    PacketReceived(usize),
}

assert_eq!(
    NetworkEvent::Connected { host: "example.com".into(), port: 443 }.debug(),
    "connected to example.com:443",
);
assert_eq!(NetworkEvent::Disconnected.debug(), "disconnected");
assert_eq!(NetworkEvent::PacketReceived(512).debug(), "packet received: 512 bytes");
```

### Partial field references

You do **not** have to reference every field. Unreferenced fields are simply
ignored; the macro handles the binding silently.

```rust
use dirk_events::Event;

#[derive(Debug, Clone, Event)]
#[event("x only: {x}")]
struct Position { x: i32, y: i32, z: i32 }

assert_eq!(Position { x: 5, y: 99, z: 0 }.debug(), "x only: 5");
```

## Sharing the Manager Across Systems

[`EventManager`] is cheaply cloneable — every clone shares the **same**
underlying state via `Arc<RwLock<…>>`. Pass it by value (or clone it freely)
into as many systems as you like:

```rust
use dirk_events::EventManager;

let workers = dirk_threads::WorkerPool::new("pool");
let mgr = EventManager::new(workers);

let input_system_mgr  = mgr.clone();
let render_system_mgr = mgr.clone();

// All three handles point to the same bus.
```

## Multiple Dispatchers for the Same Type

Several systems can independently produce events of the same type. All of
their events are delivered to all subscribers.

```rust
use dirk_events::{EventManager, Event};

#[derive(Debug, Clone, Event)]
struct DamageEvent(u32);

let workers = dirk_threads::WorkerPool::new("pool");
let mgr = EventManager::new(workers);
let d1 = mgr.register::<DamageEvent>(); // melee system
let d2 = mgr.register::<DamageEvent>(); // projectile system
let mut consumer = mgr.subscribe::<DamageEvent>();

d1.dispatch(DamageEvent(10));
d2.dispatch(DamageEvent(25));

let total: u32 = (0..2)
    .map(|_| consumer.consume_blocking().expect("event should arrive").0)
    .sum();
assert_eq!(total, 35);
```

## Cloning Dispatchers and Consumers

Cloning a [`Dispatcher`] registers a **new, independent producer** with the
same manager. Cloning a [`Consumer`] creates a **fresh, independent
subscription** — it does not share the receiver of the original.

## Thread Safety

[`Dispatcher`] is [`Send`], so it can be moved into background threads to
produce events off the main thread.

```rust
use dirk_events::{EventManager, Event};
use std::thread;

#[derive(Debug, Clone, Event)]
struct WorkDone(u32);

let workers = dirk_threads::WorkerPool::new("pool");
let mgr = EventManager::new(workers);
let dispatcher = mgr.register::<WorkDone>();
let mut consumer   = mgr.subscribe::<WorkDone>();

thread::spawn(move || {
    dispatcher.dispatch(WorkDone(1));
    dispatcher.dispatch(WorkDone(2));
}).join().unwrap();

assert_eq!(consumer.consume_blocking().expect("first event").0, 1);
assert_eq!(consumer.consume_blocking().expect("second event").0, 2);
```
