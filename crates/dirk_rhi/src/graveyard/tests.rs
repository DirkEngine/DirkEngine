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
