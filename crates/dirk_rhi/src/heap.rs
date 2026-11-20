//! Heap slot bookkeeping shared by the backends: which slots of each array
//! are free, and the last frame whose work may read each slot.

use std::{
    collections::VecDeque,
    sync::atomic::{AtomicU64, Ordering},
};

use dirk_shaders::bindless::{
    BufferIndex, HeapIndex, HeapIndices, SamplerIndex, StorageImageIndex, TextureIndex, heap,
};

use crate::{Error, Result};

/// A slot in one of the heap's arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    pub(crate) binding: u32,
    pub(crate) index: u32,
}

impl Slot {
    /// The index shaders read the slot through.
    fn reader(self) -> HeapIndex {
        match self.binding {
            heap::TEXTURES_2D..=heap::TEXTURES_3D => {
                HeapIndex::Texture(TextureIndex::new(self.index))
            }
            heap::STORAGE_IMAGES_2D..=heap::STORAGE_IMAGES_3D => {
                HeapIndex::StorageImage(StorageImageIndex::new(self.index))
            }
            heap::SAMPLERS => HeapIndex::Sampler(SamplerIndex::new(self.index)),
            heap::BUFFERS => HeapIndex::Buffer(BufferIndex::new(self.index)),
            binding => unreachable!("the heap has no binding {binding}"),
        }
    }
}

/// The free slots of one of the heap's arrays.
#[derive(Default)]
pub(crate) struct FreeSlots {
    /// Slots never used yet start here.
    next: u32,
    /// Slots freed since, reused first, oldest first, so a stale index
    /// reads the empty descriptor for as long as possible.
    free: VecDeque<u32>,
}

impl FreeSlots {
    /// Takes a slot of the array at `binding`, of `capacity` slots, that no
    /// work still running or recording may read, as `reads` knows.
    ///
    /// Slots are freed only once the work reading them completed, but stale
    /// and forged indices may read them again later: those slots are
    /// skipped until that work completes too, so slots are never written
    /// while pending work reads them.
    pub(crate) fn take(&mut self, binding: u32, capacity: u32, reads: &Reads) -> Result<u32> {
        let unread = |index| reads.is_unread(Slot { binding, index });
        let freed = self.free.iter().position(|&index| unread(index));
        if let Some(index) = freed.and_then(|position| self.free.remove(position)) {
            return Ok(index);
        }
        while self.next < capacity {
            let index = self.next;
            self.next += 1;
            if unread(index) {
                return Ok(index);
            }
            self.free.push_back(index);
        }
        Err(Error::invalid(
            "bindless heap",
            &format!("binding {binding}"),
            format!("all {capacity} slots are in use"),
        ))
    }

    /// Makes `index` available again, once no work reads it.
    pub(crate) fn release(&mut self, index: u32) {
        self.free.push_back(index);
    }
}

/// Where the reads of each kind of index start in [`Reads::frames`]. Every
/// array an index kind reads shares that kind's frames: shaders may read
/// any of them through it.
const TEXTURES: usize = 0;
const STORAGE_IMAGES: usize = TEXTURES + heap::IMAGE_CAPACITY as usize;
const SAMPLERS: usize = STORAGE_IMAGES + heap::IMAGE_CAPACITY as usize;
const BUFFERS: usize = SAMPLERS + heap::SAMPLER_CAPACITY as usize;
const SLOTS: usize = BUFFERS + heap::BUFFER_CAPACITY as usize;

/// The last frame whose work may read each heap slot, recorded whenever the
/// RHI sees an index: in the push data of draws, and in frame
/// allocations. Objects are freed, and their slots emptied, once that frame
/// completed too (see [`Graveyard`](crate::graveyard::Graveyard)), and
/// slots are reused only then.
///
/// Reads are recorded by the open frame, on its thread, and checked when
/// frames begin, when no frame is open, or under the heap's lock while
/// taking a slot, which submissions hold too: either the read was submitted
/// first and the check sees it, or the slot was written before the
/// submission, which may then read it.
pub(crate) struct Reads {
    /// One frame per slot of each kind of index, at [`TEXTURES`] and the
    /// others.
    frames: Box<[AtomicU64]>,
    /// The last frame whose work, and all work before it, completed.
    completed: AtomicU64,
}

impl Reads {
    pub(crate) fn new() -> Self {
        Self {
            frames: (0..SLOTS).map(|_| AtomicU64::new(0)).collect(),
            completed: AtomicU64::new(0),
        }
    }

    /// Records that work of the open frame `frame` reads the slots of the
    /// indices `value` holds. A few relaxed loads, and a store per slot the
    /// first time the frame reads it.
    pub(crate) fn record(&self, value: &(impl HeapIndices + ?Sized), frame: u64) {
        value.heap_indices(&mut |index| {
            let read = &self.frames[Self::position(index)];
            // Only the open frame's thread stores, and frames only grow, so
            // nothing stores a later frame between the load and the store.
            if read.load(Ordering::Relaxed) < frame {
                read.store(frame, Ordering::Relaxed);
            }
        });
    }

    /// The last frame whose work may read any of `slots`.
    pub(crate) fn last(&self, slots: impl IntoIterator<Item = Slot>) -> u64 {
        slots
            .into_iter()
            .map(|slot| self.frames[Self::position(slot.reader())].load(Ordering::Relaxed))
            .max()
            .unwrap_or_default()
    }

    /// Records that all work up to and including frame `frame` completed.
    pub(crate) fn complete(&self, frame: u64) {
        self.completed.fetch_max(frame, Ordering::Relaxed);
    }

    /// Whether no work still running or recording may read `slot`.
    fn is_unread(&self, slot: Slot) -> bool {
        self.last([slot]) <= self.completed.load(Ordering::Relaxed)
    }

    /// Where the reads of the slot `index` reads are, in `frames`.
    fn position(index: HeapIndex) -> usize {
        let (first, slot) = match index {
            HeapIndex::Texture(index) => (TEXTURES, index.slot()),
            HeapIndex::StorageImage(index) => (STORAGE_IMAGES, index.slot()),
            HeapIndex::Sampler(index) => (SAMPLERS, index.slot()),
            HeapIndex::Buffer(index) => (BUFFERS, index.slot()),
        };
        first + slot as usize
    }
}
