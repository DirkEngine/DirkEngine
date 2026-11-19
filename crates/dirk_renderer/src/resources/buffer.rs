//! This module houses the vulkan buffer abstraction.

use std::ptr::NonNull;
use std::{ffi::c_void, marker::PhantomData};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::resources::device::Garbage;
use crate::shaders::metadata::VertexInput;
use crate::{Result, resources::device::RenderDevice};

/// An abstraction around the vulkan buffer.
pub struct Buffer<Type: BuffType = Custom> {
    device: RenderDevice,
    raw: vk::Buffer,
    allocation: Option<Allocation>,
    _type: PhantomData<Type>,
}

pub trait BuffType {
    fn get_usage() -> vk::BufferUsageFlags;
}

macro_rules! define_buff_type {
    ($name:ident, $usage:expr) => {
        pub struct $name;

        impl BuffType for $name {
            fn get_usage() -> vk::BufferUsageFlags {
                $usage
            }
        }

        pastey::paste! {
            pub type [<$name Buffer>] = Buffer<$name>;
        }
    };
}

// flags don't matter
define_buff_type!(Custom, vk::BufferUsageFlags::STORAGE_BUFFER);

impl<Type: BuffType> Buffer<Type> {
    pub fn create_custom(
        device: &RenderDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };

        let allocation = device.allocate(&AllocationCreateDesc {
            name: "buffer",
            requirements,
            location,
            linear: true, // buffers are always linear
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            device
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        };

        Ok(Self {
            device: device.clone(),
            raw: buffer,
            allocation: Some(allocation),
            _type: PhantomData,
        })
    }
    pub fn create(
        device: &RenderDevice,
        size: vk::DeviceSize,
        location: MemoryLocation,
    ) -> Result<Self> {
        Buffer::create_custom(device, size, Type::get_usage(), location)
    }
    pub fn buffer(&self) -> vk::Buffer {
        self.raw
    }
    /// Returns a valid mapped pointer if the memory is host visible, otherwise it will return None.
    /// The pointer already points to the exact memory region of the suballocation, so no offset needs to be applied.
    pub fn mapped(&self) -> Option<NonNull<c_void>> {
        if let Some(allocation) = &self.allocation {
            allocation.mapped_ptr()
        } else {
            None
        }
    }

    pub fn upload_slice<T: Copy>(device: &RenderDevice, data: &[T]) -> Result<Self> {
        let size = std::mem::size_of_val(data) as vk::DeviceSize;

        let staging_buf = Buffer::create_custom(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;

        unsafe {
            #[expect(clippy::expect_used, reason = "CpuToGpu memory is always mapped.")]
            let ptr = staging_buf
                .mapped()
                .expect("should be host visible")
                .as_ptr()
                .cast::<T>();
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let device_buf = Buffer::create_custom(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | Type::get_usage(),
            MemoryLocation::GpuOnly,
        )?;

        device_buf.copy(&staging_buf, size)?;
        Ok(device_buf)
    }

    /// Copy src into self
    pub fn copy(&self, src: &Buffer, size: vk::DeviceSize) -> Result<()> {
        let cmd = self.device.transfer_pool.begin_single_time()?;

        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        cmd.copy_buffer(src.buffer(), self.buffer(), &[region]);

        cmd.end_and_submit(&self.device.queues)?;
        Ok(())
    }
}

impl<Type: BuffType> Drop for Buffer<Type> {
    fn drop(&mut self) {
        self.device.destroy(Garbage::Buffer(self.raw));
        if let Some(alloc) = self.allocation.take() {
            self.device.destroy(Garbage::Allocation(alloc));
        }
    }
}

// TODO: see about using uniform buffer binding in generics
define_buff_type!(Uniform, vk::BufferUsageFlags::UNIFORM_BUFFER);

impl UniformBuffer {
    /// Copies `data` into the persistently-mapped host-visible memory.
    ///
    /// # Safety
    /// The mapped pointer must be valid and the allocation must cover at least
    /// `size_of::<T>()` bytes — both invariants are guaranteed by every
    /// `UboData` constructed in this module.
    pub unsafe fn write<T: Copy>(&self, data: &T) {
        #[expect(
            clippy::expect_used,
            reason = "Uniform buffers are allocated in mapped memory."
        )]
        let ptr = self
            .mapped()
            .expect("UBO allocation must be host-visible")
            .as_ptr()
            .cast::<u8>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<T>(data).cast::<u8>(),
                ptr,
                size_of::<T>(),
            );
        }
    }
}

struct Vertex;
impl BuffType for Vertex {
    fn get_usage() -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::VERTEX_BUFFER
    }
}

pub struct VertexBuffer<I: VertexInput> {
    buff: Buffer<Vertex>,
    _vert_in: PhantomData<I>,
}

impl<I: VertexInput> VertexBuffer<I> {
    pub fn upload_slice(device: &RenderDevice, data: &[I]) -> Result<Self> {
        let buff = Buffer::<Vertex>::upload_slice(device, data)?;

        Ok(Self {
            buff,
            _vert_in: PhantomData,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buff.buffer()
    }
}

define_buff_type!(Index, vk::BufferUsageFlags::INDEX_BUFFER);
