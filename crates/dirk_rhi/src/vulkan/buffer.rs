use std::sync::Arc;

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};

use dirk_shaders::bindless::heap;

use super::{
    Check, Device,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{BufferDesc, BufferUsage, Memory, Result};

/// Identifies a buffer while it is alive, to skip rebinding it.
pub(crate) type BufferId = vk::Buffer;

/// A Vulkan buffer and its memory.
pub(crate) struct Buffer {
    pub(super) device: Arc<Device>,
    pub(super) raw: vk::Buffer,
    allocation: Allocation,
    /// The buffer's slot in the heap, with storage usage.
    slot: Option<Slot>,
}

impl Buffer {
    /// Creates a buffer for a validated description. Host-visible buffers are
    /// padded and aligned to whole non-coherent atoms so flushes of any range
    /// stay inside their allocation.
    pub(crate) fn new(device: &Arc<Device>, desc: &BufferDesc<'_>) -> Result<Self> {
        let atom = device.non_coherent_atom_size;
        let host_visible = desc.memory != Memory::Device;
        let size = if host_visible {
            desc.size.next_multiple_of(atom)
        } else {
            desc.size
        };
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(desc.usage.vk())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` is valid for the device.
        let raw = unsafe { device.raw.create_buffer(&info, None) }.check("vkCreateBuffer")?;
        // Owned from here on, so a failure below destroys the buffer.
        let mut buffer = Self {
            device: device.clone(),
            raw,
            allocation: Allocation::default(),
            slot: None,
        };
        // SAFETY: `raw` was created from this device.
        let mut requirements = unsafe { device.raw.get_buffer_memory_requirements(raw) };
        if host_visible {
            requirements.alignment = requirements.alignment.max(atom);
        }
        buffer.allocation = device.allocate(desc.label, requirements, desc.memory.vk(), true)?;
        // SAFETY: the allocation satisfies the buffer's requirements and is
        // bound once.
        unsafe {
            device.raw.bind_buffer_memory(
                raw,
                buffer.allocation.memory(),
                buffer.allocation.offset(),
            )
        }
        .check("vkBindBufferMemory")?;
        device.name(raw, desc.label);
        if desc.usage.contains(BufferUsage::STORAGE) {
            buffer.slot = Some(device.add_to_heap(heap::BUFFERS, Descriptor::Buffer(raw))?);
        }
        Ok(buffer)
    }

    /// The buffer's slot in the heap, if it has storage usage.
    pub(crate) fn slot(&self) -> Option<u32> {
        self.slot.map(|slot| slot.index)
    }

    /// Identifies the buffer while it is alive.
    pub(crate) fn id(&self) -> BufferId {
        self.raw
    }

    /// Copies `data` into mapped memory at `offset`, flushing it if the
    /// memory is not coherent.
    ///
    /// # Safety
    ///
    /// The range is in bounds, the memory is host-visible, and neither the
    /// host nor the GPU accesses the range concurrently.
    pub(crate) unsafe fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        // SAFETY: the caller keeps the range in bounds of the mapping and
        // excludes concurrent access.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped(offset), data.len());
        }
        if let Some(range) = self.non_coherent_range(offset, data.len()) {
            // SAFETY: the range lies within this buffer's mapped allocation.
            unsafe { self.device.raw.flush_mapped_memory_ranges(&[range]) }
                .check("vkFlushMappedMemoryRanges")?;
        }
        Ok(())
    }

    /// Copies mapped memory at `offset` into `data`, invalidating it first if
    /// the memory is not coherent.
    ///
    /// # Safety
    ///
    /// The range is in bounds, the memory is host-visible, and GPU writes to
    /// it have completed and are visible to the host.
    pub(crate) unsafe fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        if let Some(range) = self.non_coherent_range(offset, data.len()) {
            // SAFETY: the range lies within this buffer's mapped allocation.
            unsafe { self.device.raw.invalidate_mapped_memory_ranges(&[range]) }
                .check("vkInvalidateMappedMemoryRanges")?;
        }
        // SAFETY: the caller keeps the range in bounds of the mapping and
        // excludes concurrent writes.
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped(offset), data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    /// Pointer to mapped memory at `offset`.
    ///
    /// # Safety
    ///
    /// The memory is host-visible and `offset` is in bounds.
    unsafe fn mapped(&self, offset: u64) -> *mut u8 {
        let base = self
            .allocation
            .mapped_ptr()
            .map_or(std::ptr::null_mut(), |pointer| {
                pointer.as_ptr().cast::<u8>()
            });
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the offset is within the mapping, which fits in the address space"
        )]
        let offset = offset as usize;
        // SAFETY: the caller keeps `offset` within the mapping.
        unsafe { base.add(offset) }
    }

    /// The memory range covering `len` bytes at `offset`, widened to whole
    /// non-coherent atoms, or `None` if the memory is coherent.
    fn non_coherent_range(&self, offset: u64, len: usize) -> Option<vk::MappedMemoryRange<'_>> {
        if self
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return None;
        }
        let (offset, size) = Self::atom_range(
            self.allocation.offset() + offset,
            len as u64,
            self.device.non_coherent_atom_size,
        );
        Some(
            vk::MappedMemoryRange::default()
                // SAFETY: the memory outlives the returned range's use.
                .memory(unsafe { self.allocation.memory() })
                .offset(offset)
                .size(size),
        )
    }

    /// Widens `len` bytes at `offset` to whole atoms of `atom` bytes, as
    /// `(offset, size)`. Creation pads and aligns host-visible buffers to
    /// atoms, so the widened range stays inside the allocation.
    const fn atom_range(offset: u64, len: u64, atom: u64) -> (u64, u64) {
        let start = offset / atom * atom;
        let end = (offset + len).next_multiple_of(atom);
        (start, end - start)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.bury(Garbage::Buffer {
            raw: self.raw,
            allocation: std::mem::take(&mut self.allocation),
            slot: self.slot,
        });
    }
}

impl BufferUsage {
    /// The equivalent Vulkan usage.
    fn vk(self) -> vk::BufferUsageFlags {
        [
            (Self::COPY_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
            (Self::COPY_DST, vk::BufferUsageFlags::TRANSFER_DST),
            (Self::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
            (Self::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
            (Self::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
        ]
        .into_iter()
        .filter(|(usage, _)| self.contains(*usage))
        .fold(vk::BufferUsageFlags::empty(), |all, (_, usage)| all | usage)
    }
}

impl Memory {
    /// Where `gpu-allocator` places this memory.
    const fn vk(self) -> MemoryLocation {
        match self {
            Self::Device => MemoryLocation::GpuOnly,
            Self::Upload => MemoryLocation::CpuToGpu,
            Self::Readback => MemoryLocation::GpuToCpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Buffer;

    #[test]
    fn non_coherent_flushes_cover_only_the_written_atoms() {
        // A write inside one atom flushes that atom.
        assert_eq!(Buffer::atom_range(70, 10, 64), (64, 64));
        // A write straddling atoms flushes both, and no more.
        assert_eq!(Buffer::atom_range(1024 + 60, 8, 64), (1024, 128));
        // Atom-aligned writes flush exactly themselves.
        assert_eq!(Buffer::atom_range(256, 128, 64), (256, 128));
    }
}
