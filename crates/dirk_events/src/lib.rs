#![doc = include_str!("../README.md")]

use parking_lot::RwLock;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, Weak},
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tracing::trace;

use dirk_threads::WorkerPool;

mod tests;

/// The marker trait that every event type must implement.
///
/// In practice you will almost never implement this by hand — use the
/// `#[derive(Event)]` proc-macro instead, which also gives you the
/// `#[event("…")]` format-string attribute.
///
/// # Requirements
///
/// | Bound | Reason |
/// |-------|--------|
/// | [`Send`] | Events may be queued from background threads |
/// | [`Clone`] | Each subscriber receives its own independent copy |
/// | `'static` | Events are stored inside type-erased trait objects |
///
/// # Manual Implementation
///
/// ```rust
/// use dirk_events::Event;
///
/// #[derive(Clone)]
/// struct MyEvent { value: i32 }
///
/// impl Event for MyEvent {
///     fn debug(&self) -> String {
///         format!("MyEvent({})", self.value)
///     }
/// }
/// ```
pub trait Event: Send + Clone + 'static {
    /// Returns a human-readable description of this event instance, used by
    /// the internal tracing instrumentation.
    ///
    /// When you use `#[derive(Event)]` this is generated automatically from the
    /// optional `#[event("…")]` format string (or falls back to `{self:?}`).
    fn debug(&self) -> String;
}

#[doc(hidden)]
pub use dirk_proc::Event;

/// The central event bus.
///
/// `EventManager` owns the internal channel infrastructure and is responsible
/// for routing dispatched events to the right subscribers on background worker
/// threads.
///
/// # Cloning
///
/// `EventManager` is **cheaply cloneable** — all clones share the same
/// underlying state through an [`Arc`]. Clone it freely and pass it into every
/// system that needs to produce or consume events.
///
/// ```rust
/// use dirk_events::EventManager;
///
/// let workers = dirk_threads::WorkerPool::new("pool");
/// let mgr_a = EventManager::new(workers);
/// let mgr_b = mgr_a.clone(); // same bus, different handle
/// ```
///
/// # Shutdown
///
/// Subscriptions stay open for as long as any `EventManager` clone is alive,
/// independent of how many [`Dispatcher`]s exist. Dropping the last clone
/// shuts the bus down: every [`Consumer`] receives the events already
/// dispatched (while the [`WorkerPool`] is still running), after which
/// [`Consumer::consume`] and [`Consumer::consume_blocking`] return `None`.
/// Later dispatches are discarded. [`Dispatcher`]s and [`Consumer`]s do not
/// keep the bus alive, so threads blocked in a receive can observe shutdown.
#[derive(Clone)]
pub struct EventManager {
    bus: Arc<Bus>,
}

/// State shared by all [`EventManager`] clones.
struct Bus {
    topics: RwLock<HashMap<TypeId, Arc<dyn AnyTopic>>>,
    workers: WorkerPool,
}

impl Drop for Bus {
    fn drop(&mut self) {
        for topic in self.topics.get_mut().values() {
            topic.close();
        }
    }
}

/// Type-erased access to a [`Topic`] stored in the [`Bus`].
trait AnyTopic: Send + Sync {
    fn close(&self);
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

/// Routing state for one event type, shared by the bus, its dispatchers and
/// its consumers.
struct Topic<T: Event> {
    state: RwLock<TopicState<T>>,
    /// Queue drained by the single routing task of this event type.
    router: UnboundedSender<RoutedEvent<T>>,
}

struct TopicState<T: Event> {
    /// Copy-on-write so a dispatch can snapshot it with one reference count.
    subscribers: Arc<Vec<Subscriber<T>>>,
    next_id: u64,
    closed: bool,
}

/// The sending half of one [`Consumer`]'s channel.
struct Subscriber<T: Event> {
    id: u64,
    sender: UnboundedSender<T>,
}

impl<T: Event> Clone for Subscriber<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            sender: self.sender.clone(),
        }
    }
}

/// An event paired with the subscribers that existed when it was dispatched.
struct RoutedEvent<T: Event> {
    event: T,
    subscribers: Arc<Vec<Subscriber<T>>>,
}

impl<T: Event> Topic<T> {
    fn new(router: UnboundedSender<RoutedEvent<T>>) -> Self {
        Self {
            state: RwLock::new(TopicState {
                subscribers: Arc::default(),
                next_id: 0,
                closed: false,
            }),
            router,
        }
    }

    fn subscribe(self: &Arc<Self>) -> Consumer<T> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let mut state = self.state.write();
        let id = state.next_id;
        state.next_id += 1;
        if !state.closed {
            Arc::make_mut(&mut state.subscribers).push(Subscriber { id, sender });
        }
        Consumer {
            receiver,
            topic: Arc::clone(self),
            id,
        }
    }

    fn unsubscribe(&self, id: u64) {
        let mut state = self.state.write();
        if let Some(index) = state.subscribers.iter().position(|sub| sub.id == id) {
            Arc::make_mut(&mut state.subscribers).swap_remove(index);
        }
    }

    fn dispatch(&self, event: T) {
        let subscribers = Arc::clone(&self.state.read().subscribers);
        if !subscribers.is_empty() {
            let _ = self.router.send(RoutedEvent { event, subscribers });
        }
    }
}

impl<T: Event> AnyTopic for Topic<T> {
    fn close(&self) {
        let mut state = self.state.write();
        state.closed = true;
        // Routed events keep their own snapshots, so queued events still arrive.
        state.subscribers = Arc::default();
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}

/// Closes a topic when its routing task ends, including by panic (e.g. in the
/// event's [`Clone`]) or cancellation, so consumers observe closure instead of
/// waiting forever.
struct CloseOnExit<T: Event>(Weak<Topic<T>>);

impl<T: Event> Drop for CloseOnExit<T> {
    fn drop(&mut self) {
        if let Some(topic) = self.0.upgrade() {
            topic.close();
        }
    }
}

/// The single routing task of `topic`: forwards each queued event to the
/// subscribers captured when it was dispatched, in dispatch order.
async fn route<T: Event>(topic: Weak<Topic<T>>, mut receiver: UnboundedReceiver<RoutedEvent<T>>) {
    let _close = CloseOnExit(topic);
    while let Some(RoutedEvent { event, subscribers }) = receiver.recv().await {
        if let Some((last, rest)) = subscribers.split_last() {
            for subscriber in rest {
                let _ = subscriber.sender.send(event.clone());
            }
            let _ = last.sender.send(event);
        }
    }
}

impl EventManager {
    /// Creates a new, empty event manager with a [`WorkerPool`].
    #[must_use]
    pub fn new(workers: WorkerPool) -> Self {
        Self {
            bus: Arc::new(Bus {
                topics: RwLock::default(),
                workers,
            }),
        }
    }

    /// Returns a [`Dispatcher`] for events of type `T`.
    ///
    /// Any number of dispatchers may exist for the same type. They share one
    /// ordered routing queue per event type, and creating or dropping them
    /// never affects existing subscriptions.
    #[must_use]
    pub fn register<T: Event>(&self) -> Dispatcher<T> {
        Dispatcher {
            topic: self.topic(),
        }
    }

    /// Subscribes to an event type and returns a [`Consumer`] for it.
    ///
    /// Every call to `subscribe` creates an **independent** subscription. Each
    /// consumer receives its own clone of every event dispatched for that type,
    /// regardless of how many other consumers exist.
    ///
    /// A consumer can be created before or after a dispatcher is registered for
    /// the same type.
    #[must_use]
    pub fn subscribe<T: Event>(&self) -> Consumer<T> {
        self.topic::<T>().subscribe()
    }

    /// Returns the topic for `T`, creating it and its routing task on first use.
    fn topic<T: Event>(&self) -> Arc<Topic<T>> {
        let type_id = TypeId::of::<T>();
        let existing = self.bus.topics.read().get(&type_id).cloned();
        let topic = existing.unwrap_or_else(|| {
            Arc::clone(
                self.bus
                    .topics
                    .write()
                    .entry(type_id)
                    .or_insert_with(|| self.spawn_topic::<T>()),
            )
        });
        topic
            .into_any()
            .downcast::<Topic<T>>()
            .expect("topic type invariant violated: TypeId key must match Topic<T>")
    }

    fn spawn_topic<T: Event>(&self) -> Arc<dyn AnyTopic> {
        let (router, receiver) = mpsc::unbounded_channel();
        let topic = Arc::new(Topic::<T>::new(router));
        self.bus
            .workers
            .spawn(route(Arc::downgrade(&topic), receiver));
        topic
    }
}

/// Queues events to be forwarded to subscribers by a background worker task.
///
/// Created by [`EventManager::register`]. Cheap to clone — pass it into any
/// number of systems.
///
/// # Ordering
///
/// All dispatchers of one event type share a single routing queue, so each
/// consumer receives events in the order they were dispatched, across every
/// dispatcher of that type.
///
/// # Lifetime
///
/// A dispatcher does not keep subscriptions open. Dropping every dispatcher
/// leaves existing consumers subscribed, and events from dispatchers created
/// later still reach them. Dispatching after the [`EventManager`] shut down
/// discards the event.
pub struct Dispatcher<T: Event> {
    topic: Arc<Topic<T>>,
}

impl<T: Event> Dispatcher<T> {
    /// Queues `event` to be forwarded to all current subscribers as soon as a
    /// worker thread can route it.
    ///
    /// The set of subscribers is captured here: consumers created after this
    /// call do not receive `event`. This method is non-blocking and does not
    /// route on the caller thread.
    pub fn dispatch(&self, event: T) {
        trace!("dispatching event {}", event.debug());
        self.topic.dispatch(event);
    }
}

impl<T: Event> Clone for Dispatcher<T> {
    /// Returns another dispatcher for the same event type and routing queue.
    fn clone(&self) -> Self {
        Self {
            topic: Arc::clone(&self.topic),
        }
    }
}

impl<T: Event> std::fmt::Debug for Dispatcher<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dispatcher").finish_non_exhaustive()
    }
}

/// Receives events routed by background worker tasks.
///
/// Created by [`EventManager::subscribe`]. Each `Consumer` holds an independent
/// subscription — it is **not** shared with other consumers of the same type.
///
/// # Cloning
///
/// Cloning a `Consumer` creates a **fresh subscription** to the same event
/// type. The clone starts empty and receives events dispatched *after* it was
/// created; it does **not** inherit any events already queued in the original.
///
/// # Lifetime
///
/// The subscription stays open while the [`EventManager`] is alive, even when
/// no dispatcher exists. Once the last `EventManager` clone is dropped, the
/// consumer yields the remaining queued events and then reports closure. The
/// consumer also closes if routing for its event type fails, e.g. because the
/// event's [`Clone`] implementation panicked. When a `Consumer` is dropped its
/// subscription is removed immediately.
pub struct Consumer<T: Event> {
    receiver: UnboundedReceiver<T>,
    topic: Arc<Topic<T>>,
    id: u64,
}

impl<T: Event> Consumer<T> {
    /// Returns the **next** pending event, or `None` if the queue is currently
    /// empty or closed.
    ///
    /// This is non-blocking. Use [`consume_all`] if you want to drain every
    /// event that arrived so far.
    ///
    /// [`consume_all`]: Consumer::consume_all
    pub fn try_consume(&mut self) -> Option<T> {
        let res = self.receiver.try_recv().ok();
        if let Some(event) = res.as_ref() {
            trace!("consuming {}", event.debug());
        }
        res
    }

    /// Waits for the next event, or returns `None` once the [`EventManager`]
    /// has shut down and all queued events have been delivered.
    pub async fn consume(&mut self) -> Option<T> {
        let res = self.receiver.recv().await;
        if let Some(event) = res.as_ref() {
            trace!("consuming {}", event.debug());
        }
        res
    }

    /// Blocks the current thread until the next event arrives, or returns
    /// `None` once the [`EventManager`] has shut down and all queued events
    /// have been delivered.
    ///
    /// # Panics
    ///
    /// Panics if called within an asynchronous execution context; use
    /// [`consume`](Consumer::consume) there instead.
    pub fn consume_blocking(&mut self) -> Option<T> {
        let res = self.receiver.blocking_recv();
        if let Some(event) = res.as_ref() {
            trace!("consuming {}", event.debug());
        }
        res
    }

    /// Returns a lazy iterator that **drains all currently pending events**.
    ///
    /// The iterator calls [`try_consume`] repeatedly and stops as soon as the
    /// queue is empty. Collect it into a `Vec` or drive it with a `for` loop.
    ///
    /// [`try_consume`]: Consumer::try_consume
    pub fn consume_all(&mut self) -> impl Iterator<Item = T> {
        std::iter::from_fn(|| self.try_consume())
    }
}

impl<T: Event> Clone for Consumer<T> {
    /// Creates a **fresh, independent subscription** to the same event type.
    /// See the [type-level docs](Consumer) for details.
    fn clone(&self) -> Self {
        self.topic.subscribe()
    }
}

impl<T: Event> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.topic.unsubscribe(self.id);
    }
}

impl<T: Event> std::fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consumer").finish_non_exhaustive()
    }
}
