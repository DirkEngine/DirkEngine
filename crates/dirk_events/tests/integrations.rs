//! Integration tests for the `events` crate.
//!
//! These tests live in `events/tests/` and therefore exercise only the public API.
//! They cover cross-crate usage of the Event derive macro, realistic engine-loop
//! simulations, and concurrency safety.

use dirk_events::{Consumer, Dispatcher, Event, EventManager};
use dirk_threads::WorkerPool;
use std::thread;

// =============================================================================
// Helper
// =============================================================================

fn collect_all<T: Event>(consumer: &mut Consumer<T>, count: usize) -> Vec<T> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut events = Vec::with_capacity(count);
    while events.len() < count {
        if let Some(event) = consumer.try_consume() {
            events.push(event);
        } else {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for event"
            );
            std::thread::yield_now();
        }
    }
    assert!(
        consumer.try_consume().is_none(),
        "received more than {count} events"
    );
    events
}

fn wait_for<T: Event>(consumer: &mut Consumer<T>) -> T {
    collect_all(consumer, 1)
        .pop()
        .expect("one event was collected")
}

// =============================================================================
// Section A – Derive macro: Event types defined outside the crate
//
// Ensures the macro works when the user crate defines its own event types
// (the most common real-world scenario).
// =============================================================================

// ── A.1  A realistic application event hierarchy ──────────────────────────────

#[derive(Debug, Clone, Event)]
#[event("window resized to {width}×{height} px")]
struct WindowResized {
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Event)]
#[event("key pressed: code={0}")]
struct KeyPressed(u32);

#[derive(Debug, Clone, Event)]
#[event("mouse moved to ({x:.1}, {y:.1})")]
struct MouseMoved {
    x: f64,
    y: f64,
}

/// Unit event – fired once, carries no payload.
#[derive(Debug, Clone, Event)]
#[event("application shutdown requested")]
struct ShutdownRequested;

/// Enum event mixing all three variant kinds.
#[derive(Debug, Clone, Event)]
enum NetworkEvent {
    #[event("connected to {host}:{port}")]
    Connected { host: String, port: u16 },

    #[event("disconnected")]
    Disconnected,

    #[event("packet received: {0} bytes")]
    PacketReceived(usize),
}

// ── A.2  Types whose `debug()` relies on `{self:?}` (no attribute) ────────────

#[derive(Debug, Clone, Event)]
struct RawEvent {
    id: u32,
    tag: &'static str,
}

#[derive(Debug, Clone, Event)]
enum RawEnumEvent {
    A,
    // this field is used
    #[allow(unused)]
    B(i32),
    C {
        label: String,
    },
}

// =============================================================================
// Section B – `debug()` correctness for integration-level types
// =============================================================================

#[test]
fn window_resized_debug_contains_dimensions() {
    let e = WindowResized {
        width: 1920,
        height: 1080,
    };
    assert_eq!(e.debug(), "window resized to 1920×1080 px");
}

#[test]
fn key_pressed_debug_contains_code() {
    assert_eq!(KeyPressed(65).debug(), "key pressed: code=65");
}

#[test]
fn mouse_moved_debug_contains_coordinates() {
    let e = MouseMoved { x: 12.3, y: 45.6 };
    let dbg = e.debug();
    assert!(dbg.contains("12.3"), "debug output: {dbg}");
    assert!(dbg.contains("45.6"), "debug output: {dbg}");
}

#[test]
fn shutdown_requested_is_static_string() {
    assert_eq!(ShutdownRequested.debug(), "application shutdown requested");
}

#[test]
fn network_event_connected_interpolates_host_and_port() {
    let e = NetworkEvent::Connected {
        host: "127.0.0.1".into(),
        port: 8080,
    };
    assert_eq!(e.debug(), "connected to 127.0.0.1:8080");
}

#[test]
fn network_event_disconnected_is_static() {
    assert_eq!(NetworkEvent::Disconnected.debug(), "disconnected");
}

#[test]
fn network_event_packet_received_shows_byte_count() {
    assert_eq!(
        NetworkEvent::PacketReceived(512).debug(),
        "packet received: 512 bytes"
    );
}

#[test]
fn raw_event_falls_back_to_debug_repr() {
    let e = RawEvent { id: 1, tag: "test" };
    let dbg = e.debug();
    // The default uses `{self:?}` so the Debug impl must round-trip.
    assert!(dbg.contains("RawEvent"));
    assert!(dbg.contains('1'));
    assert!(dbg.contains("test"));
}

#[test]
fn raw_enum_event_each_variant_falls_back_to_debug() {
    assert_eq!(RawEnumEvent::A.debug(), format!("{:?}", RawEnumEvent::A));
    assert_eq!(
        RawEnumEvent::B(7).debug(),
        format!("{:?}", RawEnumEvent::B(7))
    );
    let v = RawEnumEvent::C { label: "hi".into() };
    assert_eq!(v.debug(), format!("{v:?}"));
}

// =============================================================================
// Section C – EventManager: realistic "engine loop" simulations
// =============================================================================

/// Simulates a two-system engine: an input system produces `KeyPressed` events,
/// a UI system produces `WindowResized` events, and two unrelated consumers
/// listen to each.
#[test]
fn realistic_multi_system_engine_loop() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);

    let key_dispatcher: Dispatcher<KeyPressed> = mgr.register();
    let win_dispatcher: Dispatcher<WindowResized> = mgr.register();

    let mut key_consumer: Consumer<KeyPressed> = mgr.subscribe();
    let mut win_consumer: Consumer<WindowResized> = mgr.subscribe();

    // Simulate three frames.
    for frame in 0u32..3 {
        key_dispatcher.dispatch(KeyPressed(65 + frame));

        if frame == 1 {
            win_dispatcher.dispatch(WindowResized {
                width: 800,
                height: 600,
            });
        }
    }

    // Drain and verify.
    let keys: Vec<u32> = collect_all(&mut key_consumer, 3)
        .into_iter()
        .map(|e| e.0)
        .collect();
    assert_eq!(keys, vec![65, 66, 67]);

    let wins = collect_all(&mut win_consumer, 1);
    assert_eq!(wins.len(), 1);
    assert_eq!(wins[0].width, 800);
}

/// Producer fires many events; multiple consumers should each receive all of them.
#[test]
fn fan_out_to_many_consumers() {
    const N: usize = 500;

    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<KeyPressed>();

    let mut consumers: Vec<Consumer<KeyPressed>> = (0..8).map(|_| mgr.subscribe()).collect();

    for i in 0..u32::try_from(N).expect("test count should fit in u32") {
        dispatcher.dispatch(KeyPressed(i));
    }

    for consumer in &mut consumers {
        let events = collect_all(consumer, N);
        assert_eq!(events.len(), N, "consumer did not receive all events");
        for (i, event) in events.iter().enumerate() {
            assert_eq!(
                event.0,
                u32::try_from(i).expect("event index should fit in u32")
            );
        }
    }
}

/// Dispatcher is moved to a background thread; the event manager dispatches on the
/// main thread. Tests `Send` bounds on Dispatcher and Event.
#[test]
fn dispatcher_is_send_and_can_be_moved_to_thread() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<KeyPressed>();
    let mut consumer = mgr.subscribe::<KeyPressed>();

    let handle = thread::spawn(move || {
        for i in 0..10u32 {
            dispatcher.dispatch(KeyPressed(i));
        }
    });

    handle.join().expect("thread panicked");

    let events = collect_all(&mut consumer, 10);
    assert_eq!(events.len(), 10);
}

/// Verifies that `ShutdownRequested` (a unit event) correctly clones and reaches
/// all subscribers.
#[test]
fn unit_event_reaches_all_consumers() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<ShutdownRequested>();
    let mut c1 = mgr.subscribe::<ShutdownRequested>();
    let mut c2 = mgr.subscribe::<ShutdownRequested>();

    dispatcher.dispatch(ShutdownRequested);

    assert_eq!(collect_all(&mut c1, 1).len(), 1);
    assert_eq!(collect_all(&mut c2, 1).len(), 1);
}

// =============================================================================
// Section D – Edge cases on the public API
// =============================================================================

/// Registering but never subscribing must not panic on dispatch.
#[test]
fn register_without_subscribe_is_safe() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<KeyPressed>();
    dispatcher.dispatch(KeyPressed(1));
    // No assertion needed – we just must not panic.
}

/// Subscribing but never registering any dispatcher: consumer stays empty.
#[test]
fn subscribe_without_register_yields_empty_consumer() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let mut consumer = mgr.subscribe::<KeyPressed>();
    assert!(collect_all(&mut consumer, 0).is_empty());
}

/// Consumers that are dropped mid-simulation are silently pruned; remaining
/// consumers and subsequent dispatches must be unaffected.
#[test]
fn mid_simulation_consumer_drop_is_handled() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<KeyPressed>();
    let mut alive = mgr.subscribe::<KeyPressed>();

    {
        let _dying = mgr.subscribe::<KeyPressed>();
        dispatcher.dispatch(KeyPressed(1));
        // `_dying` dropped here.
    }

    // After the dropped consumer is pruned, further dispatches must work fine.
    dispatcher.dispatch(KeyPressed(2));

    let events: Vec<u32> = collect_all(&mut alive, 2)
        .into_iter()
        .map(|e| e.0)
        .collect();
    // Both events delivered to `alive` (which was never dropped).
    assert_eq!(events, vec![1, 2]);
}

/// Events are routed without an explicit flush.
#[test]
fn events_are_visible_without_explicit_flush() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<KeyPressed>();
    let mut consumer = mgr.subscribe::<KeyPressed>();

    dispatcher.dispatch(KeyPressed(42));
    assert_eq!(wait_for(&mut consumer).0, 42);
}

/// Verifies that an enum event (`NetworkEvent`) goes through the full pipeline.
#[test]
fn enum_event_round_trips_through_manager() {
    let workers = WorkerPool::new("test");
    let mgr = EventManager::new(workers);
    let dispatcher = mgr.register::<NetworkEvent>();
    let mut consumer = mgr.subscribe::<NetworkEvent>();

    dispatcher.dispatch(NetworkEvent::Connected {
        host: "example.com".into(),
        port: 443,
    });
    dispatcher.dispatch(NetworkEvent::PacketReceived(1024));
    dispatcher.dispatch(NetworkEvent::Disconnected);

    let events = collect_all(&mut consumer, 3);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].debug(), "connected to example.com:443");
    assert_eq!(events[1].debug(), "packet received: 1024 bytes");
    assert_eq!(events[2].debug(), "disconnected");
}
