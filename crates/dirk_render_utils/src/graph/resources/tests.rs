use dirk_rhi::FRAMES_IN_FLIGHT;

use super::Pool;

const IN_FLIGHT: u64 = FRAMES_IN_FLIGHT as u64;

#[test]
fn pooled_resources_are_reused_once_their_frame_completed() {
    let mut pool = Pool::default();
    pool.release(1, ["a", "b"]);
    // Frames still running frame 1's work cannot reuse its resources.
    assert_eq!(pool.take(IN_FLIGHT, |_| true), None);
    assert_eq!(pool.take(1 + IN_FLIGHT, |&name| name == "b"), Some("b"));
    assert_eq!(pool.take(1 + IN_FLIGHT, |&name| name == "b"), None);
}

#[test]
fn resources_a_frame_did_not_reuse_are_evicted() {
    let mut pool = Pool::default();
    pool.release(1, ["a", "b"]);
    pool.evict(1 + IN_FLIGHT);
    assert_eq!(
        pool.take(1 + IN_FLIGHT, |&name| name == "a"),
        Some("a"),
        "the first frame that may reuse them keeps them"
    );
    pool.evict(2 + IN_FLIGHT);
    assert_eq!(pool.take(2 + IN_FLIGHT, |_| true), None);
}
