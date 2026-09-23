#![doc = include_str!("../README.md")]

use parking_lot::RwLock;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Arc,
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
/// underlying state through an `Arc<RwLock<…>>`. Clone it freely and pass it
/// into every system that needs to produce or consume events.
///
/// ```rust
/// use dirk_events::EventManager;
///
/// let workers = dirk_threads::WorkerPool::new("pool");
/// let mgr_a = EventManager::new(workers);
/// let mgr_b = mgr_a.clone(); // same bus, different handle
/// ```
#[derive(Clone)]
pub struct EventManager {
    subscribers: Arc<RwLock<HashMap<TypeId, EventTypeState>>>,
    workers: WorkerPool,
}

#[derive(Default)]
struct EventTypeState {
    dispatchers: usize,
    subscribers: Vec<Subscriber>,
}

struct RoutedEvent<T: Event> {
    event: T,
    subscribers: Vec<UnboundedSender<T>>,
}

impl EventManager {
    /// Creates a new, empty event manager with a [`WorkerPool`].
    #[must_use]
    pub fn new(workers: WorkerPool) -> Self {
        Self {
            subscribers: Arc::default(),
            workers,
        }
    }

    /// Registers a new event type and returns a [`Dispatcher`] for it.
    ///
    /// Every call to `register` creates an **independent** producer channel and
    /// a matching background routing task. Multiple dispatchers for the same
    /// event type are fully supported.
    ///
    #[must_use]
    pub fn register<T: Event>(&self) -> Dispatcher<T> {
        let (sender, receiver) = mpsc::unbounded_channel::<RoutedEvent<T>>();
        self.subscribers
            .write()
            .entry(TypeId::of::<T>())
            .or_default()
            .dispatchers += 1;
        self.spawn_router(receiver);
        Dispatcher {
            sender,
            manager: self.clone(),
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
        let (sender, receiver) = mpsc::unbounded_channel::<T>();
        let type_id = TypeId::of::<T>();
        let id = Arc::new(());
        self.subscribers
            .write()
            .entry(type_id)
            .or_default()
            .subscribers
            .push(Subscriber {
                id: Arc::clone(&id),
                sender: Box::new(sender),
            });
        Consumer {
            receiver,
            manager: self.clone(),
            id,
        }
    }

    fn spawn_router<T: Event>(&self, mut receiver: UnboundedReceiver<RoutedEvent<T>>) {
        self.workers.spawn(async move {
            while let Some(routed) = receiver.recv().await {
                for sender in routed.subscribers {
                    let _ = sender.send(routed.event.clone());
                }
            }
        });
    }

    fn unregister<T: Event>(&self) {
        let mut types = self.subscribers.write();
        let type_id = TypeId::of::<T>();
        if let Some(state) = types.get_mut(&type_id) {
            state.dispatchers -= 1;
            if state.dispatchers == 0 {
                // Pending routed events retain their own senders until delivered.
                types.remove(&type_id);
            }
        }
    }
}

/// A subscriber is a sender for events.
/// On event dispatching, will send the events through the
/// channels of every subscriber.
struct Subscriber {
    id: Arc<()>,
    sender: Box<dyn Any + Send + Sync>,
}

/// Queues events to be forwarded to subscribers by a background worker task.
///
/// Created by [`EventManager::register`]. Cheaply shareable — pass it by clone
/// into any number of systems.
///
/// # Cloning
///
/// Cloning a `Dispatcher` registers a **new, independent producer** with the
/// same [`EventManager`]. This means the clone and the original each have their
/// own internal routing channel, but both sets of events are delivered to all
/// subscribers.
pub struct Dispatcher<T: Event> {
    sender: UnboundedSender<RoutedEvent<T>>,
    manager: EventManager,
}

impl<T: Event> Dispatcher<T> {
    /// Queues `event` to be forwarded to all subscribers as soon as a worker
    /// thread can route it.
    ///
    /// This method is non-blocking and returns immediately. The event is not
    /// routed on the caller thread.
    pub fn dispatch(&self, event: T) {
        trace!("dispatching event {}", event.debug());
        let subscribers: Vec<_> = self
            .manager
            .subscribers
            .read()
            .get(&TypeId::of::<T>())
            .map(|state| {
                state
                    .subscribers
                    .iter()
                    .filter_map(|sub| sub.sender.downcast_ref::<UnboundedSender<T>>().cloned())
                    .collect()
            })
            .unwrap_or_default();
        if !subscribers.is_empty() {
            let _ = self.sender.send(RoutedEvent { event, subscribers });
        }
    }
}

impl<T: Event> Drop for Dispatcher<T> {
    fn drop(&mut self) {
        self.manager.unregister::<T>();
    }
}

impl<T: Event> Clone for Dispatcher<T> {
    /// Creates a **new, independent** dispatcher registered with the same
    /// [`EventManager`]. See the [type-level docs](Dispatcher) for details.
    fn clone(&self) -> Self {
        self.manager.register()
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
/// Cloning a `Consumer` creates a **fresh subscription** backed by the same
/// [`EventManager`]. The clone starts empty and receives events dispatched
/// *after* it was created; it does **not** inherit any events already queued
/// in the original.
///
/// # Dropping
///
/// When a `Consumer` is dropped its subscription is removed immediately.
pub struct Consumer<T: Event> {
    receiver: UnboundedReceiver<T>,
    manager: EventManager,
    id: Arc<()>,
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

    /// Waits for the next event, or returns `None` once the final dispatcher
    /// has been dropped and all queued events have been delivered.
    pub async fn consume(&mut self) -> Option<T> {
        let res = self.receiver.recv().await;
        if let Some(event) = res.as_ref() {
            trace!("consuming {}", event.debug());
        }
        res
    }

    /// Blocks the current thread until the next event arrives, or all
    /// dispatchers for this subscription are dropped.
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
    /// Creates a **fresh, independent subscription** with the same
    /// [`EventManager`]. See the [type-level docs](Consumer) for details.
    fn clone(&self) -> Self {
        self.manager.subscribe()
    }
}

impl<T: Event> Drop for Consumer<T> {
    fn drop(&mut self) {
        let mut types = self.manager.subscribers.write();
        let type_id = TypeId::of::<T>();
        if let Some(state) = types.get_mut(&type_id) {
            state
                .subscribers
                .retain(|sub| !Arc::ptr_eq(&sub.id, &self.id));
            if state.subscribers.is_empty() && state.dispatchers == 0 {
                types.remove(&type_id);
            }
        }
    }
}

impl<T: Event> std::fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Consumer").finish_non_exhaustive()
    }
}
