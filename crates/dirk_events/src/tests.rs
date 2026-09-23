//! Unit tests for the `events` crate.
//!
//! Covers:
//!   • `EventManager`: register, subscribe, multi-type, multi-subscriber,
//!     dropped-consumer pruning, buffering, zero-subscriber robustness.
//!   • `#[derive(Event)]` macro: every combination of struct / enum × unit / named / unnamed
//!     fields, with and without `#[event("…")]` format strings, partial field references,
//!     and fields that appear in the format string vs. those that are silently ignored.

#![cfg(test)]

use crate::{Consumer, Dispatcher, Event, EventManager};

// =========================================================================
// Helper: drain every pending event from a Consumer into a Vec.
// =========================================================================

fn collect<T: Event>(consumer: &mut Consumer<T>, count: usize) -> Vec<T> {
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
    collect(consumer, 1).pop().expect("one event was collected")
}

// =========================================================================
// Section 1 – Derive-macro test types
//
// Rule: the macro must compile and `debug()` must return the expected string.
// Every combination of struct / enum shape and attribute presence is exercised.
// =========================================================================

// ── 1.1  Unit struct ─────────────────────────────────────────────────────

/// No `#[event]` attribute → falls back to `format!("{self:?}")`.
#[derive(Debug, Clone, Event)]
struct UnitStructDefault;

/// Explicit static message.
#[derive(Debug, Clone, Event)]
#[event("unit-struct-event")]
struct UnitStructWithAttr;

// ── 1.2  Named-field struct ───────────────────────────────────────────────

/// No attribute → `{self:?}`.
#[derive(Debug, Clone, Event)]
struct NamedStructDefault {
    x: i32,
    y: i32,
}

/// All fields referenced in the format string.
#[derive(Debug, Clone, Event)]
#[event("moved to ({x}, {y})")]
struct NamedStructAllFields {
    x: i32,
    y: i32,
}

/// Only *some* fields referenced – the rest must be silently allowed (the
/// macro uses `..` in the destructure pattern).
#[derive(Debug, Clone, Event)]
#[event("x only: {x}")]
struct NamedStructPartialFields {
    x: i32,
    _y: i32,
}

/// Static string (no field interpolation), but struct *has* fields.
#[derive(Debug, Clone, Event)]
#[event("something happened")]
struct NamedStructStaticMessage {
    _payload: String,
}

// ── 1.3  Unnamed-field struct ─────────────────────────────────────────────

/// No attribute → `{self:?}`.
#[derive(Debug, Clone, Event)]
struct UnnamedStructDefault(i32, String);

/// All positional fields referenced.
#[derive(Debug, Clone, Event)]
#[event("code={0} label={1}")]
struct UnnamedStructAllFields(i32, String);

/// Only the first field referenced; the second must not cause a compile error.
#[derive(Debug, Clone, Event)]
#[event("first={0}")]
struct UnnamedStructFirstOnly(i32, String, f64);

/// Only the last field of three referenced.
#[derive(Debug, Clone, Event)]
#[event("last={2}")]
struct UnnamedStructLastOnly(i32, String, f64);

/// Static string, no positional references.
#[derive(Debug, Clone, Event)]
#[event("unnamed static")]
struct UnnamedStructStatic(u8, u8);

// ── 1.4  Unit enum ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Event)]
enum UnitEnum {
    #[event("alpha occurred")]
    Alpha,
    #[event("beta occurred")]
    Beta,
}

/// No attribute on variants → `{self:?}`.
#[derive(Debug, Clone, Event)]
enum UnitEnumDefault {
    Foo,
    Bar,
}

// ── 1.5  Named-field enum ─────────────────────────────────────────────────

#[derive(Debug, Clone, Event)]
enum NamedEnum {
    #[event("resized to {width}×{height}")]
    Resized { width: u32, height: u32 },

    /// Only one field used; `_z` is bound but then ignored.
    #[event("moved to ({x}, {y})")]
    Moved { x: f32, y: f32, _z: f32 },

    /// Static message, all fields ignored.
    #[event("reset")]
    Reset { _dummy: bool },
}

// ── 1.6  Unnamed-field enum ───────────────────────────────────────────────

#[derive(Debug, Clone, Event)]
enum UnnamedEnum {
    #[event("single={0}")]
    Single(i32),

    #[event("pair: {0} / {1}")]
    Pair(String, String),

    /// First field only, second silently discarded.
    #[event("head={0}")]
    Head(u8, u8),
}

// ── 1.7  Mixed enum (unit + named + unnamed variants) ────────────────────

#[derive(Debug, Clone, Event)]
enum MixedEnum {
    #[event("just a unit")]
    Unit,

    #[event("named: id={id}")]
    Named { id: u64 },

    #[event("unnamed: {0}")]
    Unnamed(String),
}

// ── 1.8  Enum – no attributes at all (every variant defaults to {self:?}) ─

#[derive(Debug, Clone, Event)]
enum NoAttrEnum {
    Alpha,
    Beta(i32),
}

// can't really be tested as it doesn't have any fields.
// just an easy feature to have when in development
#[allow(unused)]
#[derive(Debug, Clone, Event)]
#[event("empty enum")]
enum EmptyEnum {}

// =========================================================================
// Section 2 – Derive-macro: `debug()` output assertions
// =========================================================================

mod macro_debug_output {
    use super::*;

    // ── Unit structs ─────────────────────────────────────────────────────

    #[test]
    fn unit_struct_default_uses_debug_repr() {
        let e = UnitStructDefault;
        // The default branch produces `format!("{self:?}")`.
        assert_eq!(e.debug(), format!("{UnitStructDefault:?}"));
    }

    #[test]
    fn unit_struct_with_attr_returns_literal() {
        assert_eq!(UnitStructWithAttr.debug(), "unit-struct-event");
    }

    // ── Named-field structs ───────────────────────────────────────────────

    #[test]
    fn named_struct_default_uses_debug_repr() {
        let e = NamedStructDefault { x: 3, y: 7 };
        assert_eq!(e.debug(), format!("{e:?}"));
    }

    #[test]
    fn named_struct_all_fields_interpolated() {
        let e = NamedStructAllFields { x: 10, y: -5 };
        assert_eq!(e.debug(), "moved to (10, -5)");
    }

    #[test]
    fn named_struct_partial_fields_only_referenced_one() {
        let e = NamedStructPartialFields { x: 42, _y: 99 };
        assert_eq!(e.debug(), "x only: 42");
    }

    #[test]
    fn named_struct_static_message_ignores_fields() {
        let e = NamedStructStaticMessage {
            _payload: "ignored".into(),
        };
        assert_eq!(e.debug(), "something happened");
    }

    // ── Unnamed-field structs ─────────────────────────────────────────────

    #[test]
    fn unnamed_struct_default_uses_debug_repr() {
        let e = UnnamedStructDefault(7, "hello".into());
        assert_eq!(e.debug(), format!("{e:?}"));
    }

    #[test]
    fn unnamed_struct_all_fields_interpolated() {
        let e = UnnamedStructAllFields(404, "not found".into());
        assert_eq!(e.debug(), "code=404 label=not found");
    }

    #[test]
    fn unnamed_struct_first_field_only() {
        let e = UnnamedStructFirstOnly(1, "ignored".into(), 3.5);
        assert_eq!(e.debug(), "first=1");
    }

    #[test]
    fn unnamed_struct_last_field_only() {
        // Only {2} is referenced; _0 and _1 are not bound.
        let e = UnnamedStructLastOnly(0, "skip".into(), 2.72);
        assert!(e.debug().contains("2.72"));
    }

    #[test]
    fn unnamed_struct_static_ignores_all_fields() {
        let e = UnnamedStructStatic(1, 2);
        assert_eq!(e.debug(), "unnamed static");
    }

    // ── Unit enum ─────────────────────────────────────────────────────────

    #[test]
    fn unit_enum_each_variant_returns_its_message() {
        assert_eq!(UnitEnum::Alpha.debug(), "alpha occurred");
        assert_eq!(UnitEnum::Beta.debug(), "beta occurred");
    }

    #[test]
    fn unit_enum_default_uses_debug_repr() {
        assert_eq!(
            UnitEnumDefault::Foo.debug(),
            format!("{:?}", UnitEnumDefault::Foo)
        );
        assert_eq!(
            UnitEnumDefault::Bar.debug(),
            format!("{:?}", UnitEnumDefault::Bar)
        );
    }

    // ── Named-field enum ──────────────────────────────────────────────────

    #[test]
    fn named_enum_resized_interpolates_both_fields() {
        let e = NamedEnum::Resized {
            width: 1920,
            height: 1080,
        };
        assert_eq!(e.debug(), "resized to 1920×1080");
    }

    #[test]
    fn named_enum_moved_interpolates_two_of_three_fields() {
        let e = NamedEnum::Moved {
            x: 1.0,
            y: 2.5,
            _z: 0.0,
        };
        assert_eq!(e.debug(), "moved to (1, 2.5)");
    }

    #[test]
    fn named_enum_reset_static_message() {
        let e = NamedEnum::Reset { _dummy: true };
        assert_eq!(e.debug(), "reset");
    }

    // ── Unnamed-field enum ────────────────────────────────────────────────

    #[test]
    fn unnamed_enum_single_field() {
        assert_eq!(UnnamedEnum::Single(99).debug(), "single=99");
    }

    #[test]
    fn unnamed_enum_pair_both_fields() {
        let e = UnnamedEnum::Pair("hello".into(), "world".into());
        assert_eq!(e.debug(), "pair: hello / world");
    }

    #[test]
    fn unnamed_enum_head_first_field_only() {
        let e = UnnamedEnum::Head(7, 255);
        assert_eq!(e.debug(), "head=7");
    }

    // ── Mixed enum ────────────────────────────────────────────────────────

    #[test]
    fn mixed_enum_unit_variant() {
        assert_eq!(MixedEnum::Unit.debug(), "just a unit");
    }

    #[test]
    fn mixed_enum_named_variant() {
        let e = MixedEnum::Named { id: 1_234_567_890 };
        assert_eq!(e.debug(), "named: id=1234567890");
    }

    #[test]
    fn mixed_enum_unnamed_variant() {
        let e = MixedEnum::Unnamed("payload".into());
        assert_eq!(e.debug(), "unnamed: payload");
    }

    // ── Enum without any attributes ───────────────────────────────────────

    #[test]
    fn no_attr_enum_falls_back_to_debug() {
        assert_eq!(
            NoAttrEnum::Alpha.debug(),
            format!("{:?}", NoAttrEnum::Alpha)
        );
        assert_eq!(
            NoAttrEnum::Beta(42).debug(),
            format!("{:?}", NoAttrEnum::Beta(42))
        );
    }
}

// =========================================================================
// Section 3 – EventManager: core behaviour
// =========================================================================

mod event_manager {
    use dirk_threads::WorkerPool;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };
    use std::time::Duration;

    use super::*;

    // A minimal event used by most tests below.
    #[derive(Debug, Clone, Event)]
    #[event("counter={0}")]
    struct CounterEvent(u32);

    // A second, distinct event type to test type isolation.
    #[derive(Debug, Clone, Event)]
    #[event("label={0}")]
    struct LabelEvent(String);

    // ── 3.1  Basic round-trip ─────────────────────────────────────────────

    #[test]
    fn single_event_reaches_single_subscriber() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher: Dispatcher<CounterEvent> = mgr.register();
        let mut consumer: Consumer<CounterEvent> = mgr.subscribe();

        dispatcher.dispatch(CounterEvent(1));

        let events = collect(&mut consumer, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 1);
    }

    #[test]
    fn multiple_events_all_reach_subscriber() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        for i in 0..5 {
            dispatcher.dispatch(CounterEvent(i));
        }

        let values: Vec<u32> = collect(&mut consumer, 5).into_iter().map(|e| e.0).collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4]);
    }

    // ── 3.2  Async routing ────────────────────────────────────────────────

    #[test]
    fn events_are_delivered_without_explicit_flush() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        dispatcher.dispatch(CounterEvent(42));
        assert_eq!(wait_for(&mut consumer).0, 42);
    }

    #[test]
    fn successive_dispatches_reach_consumer() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        // Each receive waits for the event dispatched so far.
        dispatcher.dispatch(CounterEvent(1));

        let first = collect(&mut consumer, 1);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].0, 1);

        dispatcher.dispatch(CounterEvent(2));

        let second = collect(&mut consumer, 1);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].0, 2);
    }

    #[test]
    fn no_events_means_empty_consumer() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let _dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        assert!(collect(&mut consumer, 0).is_empty());
    }

    // ── 3.3  Fan-out: multiple subscribers ────────────────────────────────

    #[test]
    fn single_event_reaches_all_subscribers() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut c1 = mgr.subscribe::<CounterEvent>();
        let mut c2 = mgr.subscribe::<CounterEvent>();
        let mut c3 = mgr.subscribe::<CounterEvent>();

        dispatcher.dispatch(CounterEvent(99));

        for consumer in [&mut c1, &mut c2, &mut c3] {
            let events = collect(consumer, 1);
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].0, 99);
        }
    }

    #[test]
    fn multiple_events_fan_out_to_all_subscribers() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut c1 = mgr.subscribe::<CounterEvent>();
        let mut c2 = mgr.subscribe::<CounterEvent>();

        for i in 0..3 {
            dispatcher.dispatch(CounterEvent(i));
        }

        for consumer in [&mut c1, &mut c2] {
            let values: Vec<u32> = collect(consumer, 3).into_iter().map(|e| e.0).collect();
            assert_eq!(values, vec![0, 1, 2]);
        }
    }

    // ── 3.4  Type isolation ───────────────────────────────────────────────

    #[test]
    fn subscribers_only_receive_their_event_type() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let counter_dispatcher = mgr.register::<CounterEvent>();
        let label_dispatcher = mgr.register::<LabelEvent>();

        let mut counter_consumer = mgr.subscribe::<CounterEvent>();
        let mut label_consumer = mgr.subscribe::<LabelEvent>();

        counter_dispatcher.dispatch(CounterEvent(7));
        label_dispatcher.dispatch(LabelEvent("hello".into()));

        let counters = collect(&mut counter_consumer, 1);
        assert_eq!(counters.len(), 1);
        assert_eq!(counters[0].0, 7);

        let labels = collect(&mut label_consumer, 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0].0, "hello");
    }

    #[test]
    fn no_cross_contamination_between_event_types() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let counter_dispatcher = mgr.register::<CounterEvent>();
        let _label_dispatcher = mgr.register::<LabelEvent>();

        let mut counter_consumer = mgr.subscribe::<CounterEvent>();
        let mut label_consumer = mgr.subscribe::<LabelEvent>();

        // Only fire a CounterEvent.
        counter_dispatcher.dispatch(CounterEvent(1));

        assert_eq!(collect(&mut counter_consumer, 1).len(), 1);
        assert!(collect(&mut label_consumer, 0).is_empty()); // Must not receive anything.
    }

    // ── 3.5  No subscribers registered ───────────────────────────────────

    #[test]
    fn dispatching_with_no_subscribers_does_not_panic() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();

        dispatcher.dispatch(CounterEvent(0));
        // Must not panic even though nobody is listening.
    }

    // ── 3.6  Dropped-consumer pruning ────────────────────────────────────

    #[test]
    fn dropped_consumer_is_pruned_silently() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();

        let mut alive = mgr.subscribe::<CounterEvent>();
        {
            let _dead = mgr.subscribe::<CounterEvent>();
            // `_dead` is dropped here; its receiver is gone.
        }

        // Subsequent dispatches must not panic.
        dispatcher.dispatch(CounterEvent(5));

        let events = collect(&mut alive, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 5);
    }

    #[test]
    fn all_consumers_dropped_does_not_panic() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();

        {
            let _c1 = mgr.subscribe::<CounterEvent>();
            let _c2 = mgr.subscribe::<CounterEvent>();
        } // Both dropped here.

        dispatcher.dispatch(CounterEvent(1));
    }

    // ── 3.7  Dropped Dispatcher ────────────────────────────────────────────

    #[test]
    fn subscribing_without_dispatcher_gives_empty_consumer() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        // Subscribe before any dispatcher is registered for this type.
        let mut consumer = mgr.subscribe::<CounterEvent>();

        let _dispatcher = mgr.register::<CounterEvent>();
        // No events dispatched.

        assert!(collect(&mut consumer, 0).is_empty());
    }

    // ── 3.8  High-volume stress ───────────────────────────────────────────

    #[test]
    fn high_volume_events_all_delivered() {
        const N: u32 = 10_000;

        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        for i in 0..N {
            dispatcher.dispatch(CounterEvent(i));
        }

        let events = collect(&mut consumer, N as usize);
        assert_eq!(
            u32::try_from(events.len()).expect("event count should fit in u32"),
            N
        );
        for (i, event) in events.iter().enumerate() {
            assert_eq!(
                event.0,
                u32::try_from(i).expect("event index should fit in u32")
            );
        }
    }

    // ── 3.9  Multiple ticks accumulate correctly ──────────────────────────

    #[test]
    fn events_accumulate_correctly_across_many_ticks() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        let ticks = 10u32;
        let per_tick = 3u32;

        for tick in 0..ticks {
            for j in 0..per_tick {
                dispatcher.dispatch(CounterEvent(tick * per_tick + j));
            }
        }

        // Drain everything accumulated.
        let all: Vec<u32> = collect(&mut consumer, (ticks * per_tick) as usize)
            .into_iter()
            .map(|e| e.0)
            .collect();
        let expected: Vec<u32> = (0..ticks * per_tick).collect();
        assert_eq!(all, expected);
    }

    // ── 3.10  try_consume ─────────────────────────────────────────────────

    #[test]
    fn try_consume_returns_none_when_empty() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let _dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        assert!(consumer.try_consume().is_none());
    }

    #[test]
    fn try_consume_drains_one_at_a_time() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();
        let mut barrier = consumer.clone();

        dispatcher.dispatch(CounterEvent(1));
        dispatcher.dispatch(CounterEvent(2));

        // Both events use one producer queue; the second receipt is a routing barrier.
        collect(&mut barrier, 2);

        assert_eq!(
            consumer
                .try_consume()
                .expect("first event should be present")
                .0,
            1
        );
        assert_eq!(
            consumer
                .try_consume()
                .expect("second event should be present")
                .0,
            2
        );
        assert!(consumer.try_consume().is_none());
    }

    // ── 3.11  Multiple dispatchers for the same type ──────────────────────

    #[test]
    fn two_dispatchers_for_same_type_both_reach_subscriber() {
        let workers = WorkerPool::new("test");
        let mgr = EventManager::new(workers);
        let d1 = mgr.register::<CounterEvent>();
        let d2 = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();

        d1.dispatch(CounterEvent(1));
        d2.dispatch(CounterEvent(2));

        let mut values: Vec<u32> = collect(&mut consumer, 2).into_iter().map(|e| e.0).collect();
        values.sort_unstable(); // Order across producers is not guaranteed.
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn final_dispatcher_closes_consumer_after_queued_events() {
        let mgr = EventManager::new(WorkerPool::new("test"));
        let first = mgr.register::<CounterEvent>();
        let second = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();
        first.dispatch(CounterEvent(1));
        second.dispatch(CounterEvent(2));
        drop(first);
        drop(second);

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut events = Vec::new();
            while let Some(event) = consumer.consume_blocking() {
                events.push(event.0);
            }
            done_tx
                .send(events)
                .expect("test receiver should remain open");
        });
        let mut values = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("consumer should close after pending events");
        values.sort_unstable();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn async_consumer_observes_final_dispatcher_closure() {
        let mgr = EventManager::new(WorkerPool::new("test"));
        let dispatcher = mgr.register::<CounterEvent>();
        let mut consumer = mgr.subscribe::<CounterEvent>();
        dispatcher.dispatch(CounterEvent(9));
        drop(dispatcher);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should build");
        runtime.block_on(async {
            let event = tokio::time::timeout(Duration::from_secs(2), consumer.consume())
                .await
                .expect("queued event should arrive")
                .expect("queued event should remain available");
            assert_eq!(event.0, 9);
            assert!(
                tokio::time::timeout(Duration::from_secs(2), consumer.consume())
                    .await
                    .expect("consumer should close")
                    .is_none()
            );
        });
    }

    #[test]
    fn new_subscriber_is_excluded_from_already_dispatched_event() {
        let mgr = EventManager::new(WorkerPool::new("test"));
        let mut dispatcher = mgr.register::<CounterEvent>();
        let mut early = mgr.subscribe::<CounterEvent>();
        // Intercept the producer queue so routing cannot race with subscription.
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        dispatcher.sender = sender;
        dispatcher.dispatch(CounterEvent(7));
        let mut late = mgr.subscribe::<CounterEvent>();

        let routed = receiver.try_recv().expect("event should be queued");
        assert_eq!(routed.subscribers.len(), 1);
        for subscriber in routed.subscribers {
            subscriber
                .send(routed.event.clone())
                .expect("early subscriber should remain open");
        }
        assert_eq!(
            early
                .try_consume()
                .expect("early subscriber should receive")
                .0,
            7
        );
        assert!(late.try_consume().is_none());
    }

    #[test]
    fn consuming_does_not_clone_event_for_tracing() {
        #[derive(Debug)]
        struct TrackedEvent(Arc<AtomicUsize>);

        impl Clone for TrackedEvent {
            fn clone(&self) -> Self {
                self.0.fetch_add(1, Ordering::Relaxed);
                Self(Arc::clone(&self.0))
            }
        }

        impl Event for TrackedEvent {
            fn debug(&self) -> String {
                "tracked".into()
            }
        }

        let mgr = EventManager::new(WorkerPool::new("test"));
        let dispatcher = mgr.register::<TrackedEvent>();
        let mut consumer = mgr.subscribe::<TrackedEvent>();
        let clones = Arc::new(AtomicUsize::new(0));
        dispatcher.dispatch(TrackedEvent(Arc::clone(&clones)));
        consumer.consume_blocking().expect("event should arrive");
        assert_eq!(clones.load(Ordering::Relaxed), 1);
    }
}
