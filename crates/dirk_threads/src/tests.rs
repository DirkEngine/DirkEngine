use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use super::{SHUTDOWN_TIMEOUT, WorkerPool};

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
        // wall-clock: the task outlasts a drop that doesn't wait for it, and
        // nothing signals that the drop is waiting.
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
        // wall-clock: a drop that waited would give up on the busy task only
        // after SHUTDOWN_TIMEOUT, so the time it took tells the two apart.
        let start = Instant::now();
        drop(busy);
        done_tx
            .send(start.elapsed())
            .expect("test receiver should remain open");
    });
    let elapsed = done_rx.recv().expect("the host task should finish");
    release_tx.send(()).expect("detached task should still run");
    assert!(
        elapsed < SHUTDOWN_TIMEOUT,
        "drop on another runtime should not wait for blocking tasks"
    );
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
