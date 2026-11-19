//! Device memory: every buffer and texture is allocated from an untracked
//! `MTLHeap`, so render and compute encoders make every resource shaders
//! may reach through the bindless heap resident with one `useHeap` per heap,
//! and Metal tracks no hazards between them: command buffers order their
//! encoders with fences instead (see `command`).

use std::sync::atomic::{AtomicU64, Ordering};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLResourceOptions, MTLSizeAndAlign, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
};
use parking_lot::Mutex;

use super::{Check, Object};
use crate::{Memory, Result};

/// Size of the heaps resources share. Larger resources get a heap of their
/// own.
const BLOCK_SIZE: usize = 64 << 20;

/// A heap and the memory its resources live in.
struct Block {
    memory: Memory,
    raw: Object<dyn MTLHeap>,
    /// Whether resources share the heap, which is then [`BLOCK_SIZE`]
    /// bytes, rather than one larger resource owning it.
    shared: bool,
}

/// The heaps resources are allocated from.
pub(super) struct Allocator {
    /// Locked to allocate, and to add or release heaps.
    blocks: Mutex<Vec<Block>>,
    /// Changes whenever a heap is added or released, so command buffers
    /// know when to make the heaps resident again without locking.
    generation: AtomicU64,
}

/// The heaps encoders make resident, as of a generation of the allocator;
/// by default none, of generation zero, which the allocator never has.
#[derive(Default)]
pub(super) struct Residency {
    pub(super) generation: u64,
    pub(super) heaps: Vec<Object<dyn MTLHeap>>,
}

impl Memory {
    /// The options of resources in this memory: private, or shared with the
    /// host and coherent; write-combined when only the host writes it.
    fn options(self) -> MTLResourceOptions {
        let untracked = MTLResourceOptions::HazardTrackingModeUntracked;
        untracked
            | match self {
                Self::Device => MTLResourceOptions::StorageModePrivate,
                Self::Upload => {
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeWriteCombined
                }
                Self::Readback => MTLResourceOptions::StorageModeShared,
            }
    }
}

impl Allocator {
    pub(super) fn new() -> Self {
        Self {
            blocks: Mutex::new(Vec::new()),
            generation: AtomicU64::new(1),
        }
    }

    /// Allocates a buffer of `size` bytes in `memory`.
    pub(super) fn buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
        memory: Memory,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let options = memory.options();
        let needed = device.heapBufferSizeAndAlignWithLength_options(size, options);
        self.allocate(device, memory, needed, |heap| {
            heap.newBufferWithLength_options(size, options)
        })?
        .check("newBufferWithLength")
    }

    /// Allocates a private texture described by `desc`, whose storage and
    /// tracking modes it sets.
    pub(super) fn texture(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        desc: &MTLTextureDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTLTexture>>> {
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
        let needed = device.heapTextureSizeAndAlignWithDescriptor(desc);
        self.allocate(device, Memory::Device, needed, |heap| {
            heap.newTextureWithDescriptor(desc)
        })?
        .check("newTextureWithDescriptor")
    }

    /// Creates a resource with `create` in the first heap of `memory` with
    /// room for it, or in a new heap. `None` if Metal fails to create it.
    fn allocate<T: ?Sized>(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        memory: Memory,
        needed: MTLSizeAndAlign,
        create: impl Fn(&ProtocolObject<dyn MTLHeap>) -> Option<Retained<T>>,
    ) -> Result<Option<Retained<T>>> {
        let mut blocks = self.blocks.lock();
        let fits = |block: &&Block| {
            block.memory == memory && block.raw.maxAvailableSizeWithAlignment(needed.align) >= needed.size
        };
        if let Some(resource) = blocks.iter().filter(fits).find_map(|block| create(&block.raw)) {
            return Ok(Some(resource));
        }
        let desc = MTLHeapDescriptor::new();
        desc.setType(MTLHeapType::Automatic);
        desc.setResourceOptions(memory.options());
        desc.setSize(needed.size.max(BLOCK_SIZE));
        let raw: Object<_> = device
            .newHeapWithDescriptor(&desc)
            .check("newHeapWithDescriptor")?
            .into();
        let resource = create(&raw);
        blocks.push(Block {
            memory,
            raw,
            shared: needed.size <= BLOCK_SIZE,
        });
        self.generation.fetch_add(1, Ordering::Release);
        Ok(resource)
    }

    /// Releases heaps left empty, but one shared heap of each memory, so
    /// resources created and dropped every few frames reuse it. Resources
    /// are freed once the GPU is done with them, so nothing uses an empty
    /// heap's memory.
    pub(super) fn trim(&self) {
        let mut blocks = self.blocks.lock();
        let mut spares = Vec::new();
        let count = blocks.len();
        blocks.retain(|block| {
            let keep = block.raw.usedSize() > 0
                || (block.shared && !spares.contains(&block.memory));
            if keep && block.raw.usedSize() == 0 {
                spares.push(block.memory);
            }
            keep
        });
        if blocks.len() != count {
            self.generation.fetch_add(1, Ordering::Release);
        }
    }

    /// The generation of the heaps, which changes when heaps are added or
    /// released. Lock-free, for checks before every draw.
    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Every heap, and their generation.
    pub(super) fn residency(&self) -> Residency {
        let blocks = self.blocks.lock();
        Residency {
            generation: self.generation(),
            heaps: blocks.iter().map(|block| block.raw.clone()).collect(),
        }
    }
}
