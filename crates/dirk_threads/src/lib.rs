//! This crate contains `DirkEngine`'s async threading primitives.

use std::{
    future::Future,
    num::NonZeroUsize,
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
};

use tokio::{
    runtime::{Builder, Handle},
    sync::{Mutex, oneshot},
    task::JoinHandle as TaskJoinHandle,
};
use tracing::info;

/// A cheap clonable handle to a pool of background worker threads.
///
/// The pool owns a dedicated Tokio runtime hosted on a non-game thread. Tasks
/// spawned through this handle are therefore executed away from the primary
/// engine thread.
#[derive(Clone)]
pub struct WorkerPool {
    inner: Arc<Inner>,
}

struct Inner {
    handle: Handle,
    /// Stored as an [`Option`] as the sender is consumed when sending
    /// shutdown message.
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    coordinator: Mutex<Option<JoinHandle<()>>>,
}

impl WorkerPool {
    /// Creates a pool with the specified thread name.
    ///
    /// # Panics
    ///
    /// Panics if the coordinator thread cannot start or the runtime cannot be built.
    #[must_use]
    pub fn new(name: &str) -> Self {
        let (handle_tx, handle_rx) = mpsc::sync_channel::<Handle>(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let thread_name = name.to_owned();

        let coordinator = thread::Builder::new()
            .name(thread_name.clone())
            .spawn(move || {
                let runtime = Builder::new_multi_thread()
                    .worker_threads(default_worker_count().get())
                    .thread_name(thread_name.clone())
                    .enable_all()
                    .build()
                    .expect("failed to build worker runtime");

                runtime.block_on(async move {
                    info!("starting worker pool: {thread_name}");
                    let handle = Handle::current();
                    handle_tx
                        .send(handle)
                        .expect("worker pool handle receiver should be open");
                    let _ = shutdown_rx.await;
                    info!("shutdown worker pool {thread_name}");
                });
            })
            .expect("failed to spawn worker coordinator thread");

        let handle = handle_rx
            .recv()
            .expect("worker pool runtime should initialize");

        Self {
            inner: Arc::new(Inner {
                handle,
                shutdown: Mutex::new(Some(shutdown_tx)),
                coordinator: Mutex::new(Some(coordinator)),
            }),
        }
    }

    /// Returns the underlying Tokio runtime handle.
    #[must_use]
    pub fn handle(&self) -> &Handle {
        &self.inner.handle
    }

    /// Spawns an async task onto the worker pool.
    pub fn spawn<F>(&self, future: F) -> TaskJoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.inner.handle.spawn(future)
    }

    /// Spawns a blocking closure onto the worker pool's blocking executor.
    pub fn spawn_blocking<F, R>(&self, operation: F) -> TaskJoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.inner.handle.spawn_blocking(operation)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.get_mut().take() {
            let _ = shutdown.send(());
        }

        if let Some(coordinator) = self.coordinator.get_mut().take() {
            // Joining from a task running on this pool would wait for itself.
            let on_own_runtime =
                Handle::try_current().is_ok_and(|current| current.id() == self.handle.id());
            if on_own_runtime {
                let _ = thread::Builder::new()
                    .name("dirk-pool-join".into())
                    .spawn(move || {
                        let _ = coordinator.join();
                    });
            } else {
                let _ = coordinator.join();
            }
        }
    }
}

fn default_worker_count() -> NonZeroUsize {
    thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .max(NonZeroUsize::MIN)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::Duration,
    };

    use super::WorkerPool;

    #[test]
    fn spawned_tasks_run_on_background_threads() {
        let pool = WorkerPool::new("test");
        let main_thread = std::thread::current().id();
        let (tx, rx) = mpsc::channel();

        pool.spawn(async move {
            tx.send(std::thread::current().id())
                .expect("receiver should still be alive");
        });

        let worker_thread = rx.recv().expect("task should complete");

        assert_ne!(worker_thread, main_thread);
    }

    #[test]
    fn blocking_tasks_complete() {
        let pool = WorkerPool::new("test");
        let counter = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let counter = Arc::clone(&counter);
            let (tx, rx) = mpsc::channel();
            pool.spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(5));
                counter.fetch_add(1, Ordering::SeqCst);
                tx.send(()).expect("receiver should still be alive");
            });
            tasks.push(rx);
        }

        for task in tasks {
            task.recv().expect("blocking task should complete");
        }

        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn constructor_works_inside_async_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should build");
        runtime.block_on(async {
            let pool = WorkerPool::new("nested");
            pool.spawn(async {}).await.expect("worker should run");
        });
    }

    #[test]
    fn last_handle_can_drop_inside_async_task() {
        let pool = WorkerPool::new("async-drop");
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let task_pool = pool.clone();
        pool.spawn(async move {
            let _ = release_rx.await;
            drop(task_pool);
            done_tx.send(()).expect("test receiver should remain open");
        });
        drop(pool);
        release_tx.send(()).expect("task should remain alive");
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker-side drop should not deadlock");
    }

    #[test]
    fn last_handle_can_drop_inside_blocking_task() {
        let pool = WorkerPool::new("blocking-drop");
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let task_pool = pool.clone();
        pool.spawn_blocking(move || {
            release_rx.recv().expect("test sender should remain open");
            drop(task_pool);
            done_tx.send(()).expect("test receiver should remain open");
        });
        drop(pool);
        release_tx.send(()).expect("task should remain alive");
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker-side drop should not deadlock");
    }
}
