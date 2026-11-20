use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::heap::{Reads, Slot};

/// Native objects whose owners dropped them, kept until the GPU is done.
///
/// Each object is tagged with the frame open when it was dropped, or the last
/// frame begun. Encoders cannot outlive their frame, so once every submission
/// up to and including that frame has completed, nothing can use the object
/// but later frames reading its heap slots through indices kept past the
/// drop. So each object is kept until a frame after its last use closed
/// without reading its slots, and every frame that read them, as [`Reads`]
/// records, completed: indices used every frame keep their resources alive.
/// Indices used again after that are stale: they read whatever then fills
/// the slot.
pub(crate) struct Graveyard<T> {
    /// The frame objects dropped now are tagged with.
    frame: AtomicU64,
    buried: Mutex<Vec<(u64, T)>>,
    /// The last frame whose work may read each heap slot.
    pub(crate) reads: Reads,
}

/// What the graveyard keeps: a native object, and the heap slots it fills.
pub(crate) trait Buried {
    /// The heap slots the object fills, emptied when it is freed.
    fn slots(&self) -> impl Iterator<Item = Slot>;
}

impl<T: Buried> Graveyard<T> {
    pub(crate) fn new() -> Self {
        Self {
            frame: AtomicU64::new(0),
            buried: Mutex::new(Vec::new()),
            reads: Reads::new(),
        }
    }

    /// Keeps `object` until the current frame's work completes, and that of
    /// every later frame reading its slots.
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

    /// Hands every object whose last use, its drop or a read of its slots,
    /// came before the last frame begun and by the end of frame `completed`
    /// to `free`, after releasing the lock, so freeing does not block drops.
    /// Only call once all work up to and including that frame has completed,
    /// while no frame is open, so the last frame begun recorded its reads.
    pub(crate) fn collect(&self, completed: u64, free: impl FnMut(T)) {
        self.reads.complete(completed);
        // The frame after an object's last use may read it through an index
        // kept past its drop; until it closed, frees nothing.
        let Some(closed) = self.frame.load(Ordering::Relaxed).checked_sub(1) else {
            return;
        };
        let last = completed.min(closed);
        let dead: Vec<_> = self
            .buried
            .lock()
            .extract_if(.., |(frame, object)| {
                *frame <= last && {
                    *frame = self.reads.last(object.slots()).max(*frame);
                    *frame <= last
                }
            })
            .map(|(_, object)| object)
            .collect();
        dead.into_iter().for_each(free);
    }
}

#[cfg(test)]
mod tests {
    use dirk_shaders::bindless::{SamplerIndex, heap};

    use super::{Buried, Graveyard};
    use crate::heap::Slot;

    /// A named object filling a sampler slot, or none.
    struct Object(&'static str, Option<u32>);

    impl Buried for Object {
        fn slots(&self) -> impl Iterator<Item = Slot> {
            self.1.into_iter().map(|index| Slot {
                binding: heap::SAMPLERS,
                index,
            })
        }
    }

    fn collect(graveyard: &Graveyard<Object>, completed: u64) -> Vec<&'static str> {
        let mut freed = Vec::new();
        graveyard.collect(completed, |object| freed.push(object.0));
        freed
    }

    #[test]
    fn objects_are_freed_once_their_frame_completed_and_the_next_closed() {
        let graveyard = Graveyard::new();
        graveyard.bury(Object("before any frame", None));
        assert_eq!(collect(&graveyard, 0), Vec::<&str>::new());
        graveyard.begin_frame(1);
        graveyard.bury(Object("in frame 1", None));
        assert_eq!(collect(&graveyard, 0), ["before any frame"]);
        graveyard.begin_frame(2);
        graveyard.bury(Object("in frame 2", None));

        assert_eq!(collect(&graveyard, 0), Vec::<&str>::new());
        assert_eq!(collect(&graveyard, 2), ["in frame 1"]);
        graveyard.begin_frame(3);
        assert_eq!(collect(&graveyard, 1), Vec::<&str>::new());
        assert_eq!(collect(&graveyard, 2), ["in frame 2"]);
    }

    #[test]
    fn objects_whose_slots_later_frames_read_are_freed_once_those_completed() {
        let graveyard = Graveyard::new();
        graveyard.begin_frame(1);
        graveyard.bury(Object("read in frame 3", Some(5)));
        graveyard.bury(Object("unread", Some(6)));
        graveyard.reads.record(&SamplerIndex::new(5), 2);
        graveyard.begin_frame(3);
        graveyard.reads.record(&SamplerIndex::new(5), 3);
        graveyard.begin_frame(4);

        assert_eq!(collect(&graveyard, 1), ["unread"]);
        assert_eq!(collect(&graveyard, 2), Vec::<&str>::new());
        assert_eq!(collect(&graveyard, 3), ["read in frame 3"]);
    }
}
