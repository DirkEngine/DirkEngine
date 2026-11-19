use std::sync::Arc;

use dirk_shaders::bindless::heap;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLResourceOptions};

use super::{
    Check, Device, Object,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{BufferDesc, BufferUsage, Memory, Result};

/// Identifies a buffer while it is alive, to skip rebinding it.
pub(crate) type BufferId = *const ProtocolObject<dyn MTLBuffer>;

/// A Metal buffer.
pub(crate) struct Buffer {
    pub(super) device: Arc<Device>,
    pub(super) raw: Object<dyn MTLBuffer>,
    /// The buffer's slot in the heap, with storage usage.
    slot: Option<Slot>,
}

impl Buffer {
    /// Creates a buffer for a validated description. Host-visible memory is
    /// shared with the host and coherent.
    pub(crate) fn new(device: &Arc<Device>, desc: &BufferDesc<'_>) -> Result<Self> {
        let options = match desc.memory {
            Memory::Device => MTLResourceOptions::StorageModePrivate,
            Memory::Upload => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            }
            Memory::Readback => MTLResourceOptions::StorageModeShared,
        };
        // Sizes are within `maxBufferLength`, which is a `usize`.
        #[expect(clippy::cast_possible_truncation, reason = "the size fits")]
        let size = desc.size as usize;
        let raw: Object<_> = device
            .raw
            .newBufferWithLength_options(size, options)
            .check("newBufferWithLength")?
            .into();
        raw.setLabel(Some(&NSString::from_str(desc.label)));
        let slot = desc
            .usage
            .contains(BufferUsage::STORAGE)
            .then(|| device.add_to_heap(heap::BUFFERS, &Descriptor::Buffer(&raw)))
            .transpose()?;
        Ok(Self {
            device: device.clone(),
            raw,
            slot,
        })
    }

    /// The buffer's slot in the heap, if it has storage usage.
    pub(crate) fn slot(&self) -> Option<u32> {
        self.slot.map(|slot| slot.index)
    }

    /// Identifies the buffer while it is alive.
    pub(crate) fn id(&self) -> BufferId {
        &raw const *self.raw
    }

    /// Copies `data` into shared memory at `offset`.
    ///
    /// # Safety
    ///
    /// The range is in bounds, the memory is host-visible, and neither the
    /// host nor the GPU accesses the range concurrently.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Vulkan, where this can fail"
    )]
    pub(crate) unsafe fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        // SAFETY: the caller keeps the range in bounds of the shared memory
        // and excludes concurrent access.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped(offset), data.len()) };
        Ok(())
    }

    /// Copies shared memory at `offset` into `data`.
    ///
    /// # Safety
    ///
    /// The range is in bounds, the memory is host-visible, and GPU writes to
    /// it have completed.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Vulkan, where this can fail"
    )]
    pub(crate) unsafe fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        // SAFETY: the caller keeps the range in bounds of the shared memory
        // and excludes concurrent writes.
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped(offset), data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    /// Pointer to shared memory at `offset`.
    ///
    /// # Safety
    ///
    /// The memory is host-visible and `offset` is in bounds.
    unsafe fn mapped(&self, offset: u64) -> *mut u8 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the offset is within the buffer, which fits in the address space"
        )]
        let offset = offset as usize;
        // SAFETY: the caller keeps `offset` within the buffer.
        unsafe { self.raw.contents().cast::<u8>().as_ptr().add(offset) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.bury(Garbage {
            slots: [self.slot, None],
            object: Box::new(self.raw.clone()),
        });
    }
}
