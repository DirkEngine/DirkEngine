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
    #[expect(
        clippy::expect_used,
        reason = "Building a runtime only fails when the OS can't spawn threads, which the engine can't run without."
    )]
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
mod tests;
