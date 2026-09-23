//! This crate contains `DirkEngine`'s async threading primitives.

use std::{future::Future, num::NonZeroUsize, sync::Arc, thread, time::Duration};

use tokio::{
    runtime::{Builder, Handle, Runtime},
    task::JoinHandle as TaskJoinHandle,
};
use tracing::info;

/// How long dropping the last [`WorkerPool`] handle off-runtime waits for
/// running blocking tasks before detaching them.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// A cheap clonable handle to a pool of background worker threads.
///
/// The pool owns a dedicated multi-threaded Tokio runtime. Tasks spawned
/// through this handle run on the pool's worker threads, away from the
/// primary engine thread.
///
/// # Shutdown
///
/// The runtime shuts down when the last handle is dropped. Pending async
/// tasks are cancelled at their next `.await`. What happens to running
/// [`spawn_blocking`] tasks depends on where the last handle is dropped:
///
/// - **Outside any Tokio runtime** (e.g. the engine main thread): the drop
///   blocks for at most five seconds waiting for blocking tasks to finish,
///   then detaches any that are still running.
/// - **Inside a Tokio runtime** (a task of this pool, a task of another pool,
///   or code under `block_on`): the drop never blocks. Blocking tasks are
///   detached and run to completion in the background. Blocking there could
///   wait on the dropping task itself or stall another runtime's worker.
///
/// [`spawn_blocking`]: WorkerPool::spawn_blocking
#[derive(Clone)]
pub struct WorkerPool {
    inner: Arc<Inner>,
}

struct Inner {
    name: String,
    handle: Handle,
    /// Taken on drop to choose the shutdown strategy.
    runtime: Option<Runtime>,
}

impl WorkerPool {
    /// Creates a pool whose worker threads use the specified name.
    ///
    /// This may be called from inside another Tokio runtime.
    ///
    /// # Panics
    ///
    /// Panics if the runtime cannot be built.
    #[must_use]
    pub fn new(name: &str) -> Self {
        let runtime = Builder::new_multi_thread()
            .worker_threads(default_worker_count().get())
            .thread_name(name)
            .enable_all()
            .build()
            .expect("failed to build worker runtime");
        info!("starting worker pool: {name}");

        Self {
            inner: Arc::new(Inner {
                name: name.to_owned(),
                handle: runtime.handle().clone(),
                runtime: Some(runtime),
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
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        if Handle::try_current().is_ok() {
            runtime.shutdown_background();
            info!("worker pool {} shutting down in background", self.name);
        } else {
            runtime.shutdown_timeout(SHUTDOWN_TIMEOUT);
            info!("worker pool {} shut down", self.name);
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
        time::{Duration, Instant},
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
    fn drop_outside_runtime_waits_for_blocking_tasks() {
        let pool = WorkerPool::new("graceful-drop");
        let finished = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let task_finished = Arc::clone(&finished);
        pool.spawn_blocking(move || {
            started_tx
                .send(())
                .expect("test receiver should remain open");
            std::thread::sleep(Duration::from_millis(50));
            task_finished.fetch_add(1, Ordering::SeqCst);
        });
        started_rx.recv().expect("blocking task should start");
        drop(pool);
        assert_eq!(finished.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn drop_inside_other_runtime_does_not_block_its_worker() {
        let host = WorkerPool::new("host");
        let busy = WorkerPool::new("busy");
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        busy.spawn_blocking(move || {
            started_tx
                .send(())
                .expect("test receiver should remain open");
            let _ = release_rx.recv();
        });
        started_rx.recv().expect("blocking task should start");

        let (done_tx, done_rx) = mpsc::channel();
        host.spawn(async move {
            let start = Instant::now();
            drop(busy);
            done_tx
                .send(start.elapsed())
                .expect("test receiver should remain open");
        });
        let elapsed = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("drop on another runtime should not wait for blocking tasks");
        release_tx.send(()).expect("detached task should still run");
        assert!(elapsed < Duration::from_secs(1));
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
