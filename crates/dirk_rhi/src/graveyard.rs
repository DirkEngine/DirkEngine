use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Native objects whose owners dropped them, kept until the GPU is done.
///
/// Each object is tagged with the frame open when it was dropped, or the last
/// frame begun. Encoders cannot outlive their frame, and a dropped image's
/// queued first transition is forgotten by the time the frame after its tag
/// begins, so once every submission up to and including its frame has
/// completed, nothing can use the object.
pub(crate) struct Graveyard<T> {
    /// The frame objects dropped now are tagged with.
    frame: AtomicU64,
    buried: Mutex<Vec<(u64, T)>>,
}

impl<T> Graveyard<T> {
    pub(crate) const fn new() -> Self {
        Self {
            frame: AtomicU64::new(0),
            buried: Mutex::new(Vec::new()),
        }
    }

    /// Keeps `object` until the current frame's work completes.
    pub(crate) fn bury(&self, object: T) {
        // Pairs with the release in `begin_frame`, so a drop that follows
        // recording in frame F sees F.
        let frame = self.frame.load(Ordering::Acquire);
        self.buried.lock().push((frame, object));
    }

    /// Tags objects dropped from now on with `frame`.
    pub(crate) fn begin_frame(&self, frame: u64) {
        self.frame.store(frame, Ordering::Release);
    }

    /// Hands every object dropped by the end of frame `completed` to `free`,
    /// after releasing the lock, so freeing does not block drops. Only call
    /// once all work up to and including that frame has completed.
    pub(crate) fn collect(&self, completed: u64, free: impl FnMut(T)) {
        let dead: Vec<_> = self
            .buried
            .lock()
            .extract_if(.., |(frame, _)| *frame <= completed)
            .map(|(_, object)| object)
            .collect();
        dead.into_iter().for_each(free);
    }
}

#[cfg(test)]
mod tests {
    use super::Graveyard;

    fn collect(graveyard: &Graveyard<&'static str>, completed: u64) -> Vec<&'static str> {
        let mut freed = Vec::new();
        graveyard.collect(completed, |object| freed.push(object));
        freed
    }

    #[test]
    fn objects_are_freed_only_once_their_frame_completed() {
        let graveyard = Graveyard::new();
        graveyard.bury("before any frame");
        graveyard.begin_frame(1);
        graveyard.bury("in frame 1");
        graveyard.begin_frame(2);
        graveyard.bury("in frame 2");

        assert_eq!(collect(&graveyard, 0), ["before any frame"]);
        assert_eq!(collect(&graveyard, 0), Vec::<&str>::new());
        assert_eq!(collect(&graveyard, 1), ["in frame 1"]);
        assert_eq!(collect(&graveyard, 2), ["in frame 2"]);
    }
}
