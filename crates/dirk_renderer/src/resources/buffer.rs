//! Typed renderer buffers allocated through the RHI.

use std::{marker::PhantomData, mem::size_of_val};

use ash::vk;
use dirk_rhi::{
    Backend as _, Buffer as _, BufferCopy, BufferDesc, BufferUsages, CommandBuffer as _,
    MemoryDomain,
};
use dirk_rhi_vulkan::VulkanBuffer;
use gpu_allocator::MemoryLocation;

use crate::{Result, resources::device::RenderDevice, shaders::metadata::VertexInput};

pub struct Buffer<Type: BuffType = Custom> {
    inner: VulkanBuffer,
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

define_buff_type!(Custom, vk::BufferUsageFlags::STORAGE_BUFFER);

impl<Type: BuffType> Buffer<Type> {
    pub fn create_custom(
        device: &RenderDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self> {
        let inner = device.rhi.create_buffer(&BufferDesc {
            label: "renderer buffer",
            size,
            usage: buffer_usage(usage),
            memory: memory_domain(location),
        })?;
        Ok(Self {
            inner,
            _type: PhantomData,
        })
    }

    pub fn create(
        device: &RenderDevice,
        size: vk::DeviceSize,
        location: MemoryLocation,
    ) -> Result<Self> {
        Self::create_custom(device, size, Type::get_usage(), location)
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.inner.raw()
    }

    pub(crate) fn rhi(&self) -> &VulkanBuffer {
        &self.inner
    }

    pub fn upload_slice<T: Copy>(device: &RenderDevice, data: &[T]) -> Result<Self> {
        let size = u64::try_from(size_of_val(data))
            .map_err(|_| dirk_rhi::Error::InvalidResource("buffer upload is too large"))?;
        let staging = Buffer::<Custom>::create_custom(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
        staging.rhi().write(0, bytes)?;

        let output = Self::create_custom(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | Type::get_usage(),
            MemoryLocation::GpuOnly,
        )?;
        let mut command = device.transfer_pool.begin_single_time()?;
        command.rhi_mut().copy_buffer(
            staging.rhi(),
            output.rhi(),
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );
        command.end_and_submit()?;
        Ok(output)
    }
}

define_buff_type!(Uniform, vk::BufferUsageFlags::UNIFORM_BUFFER);

impl UniformBuffer {
    pub fn write<T: Copy>(&self, data: &T) -> Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(data).cast::<u8>(), size_of_val(data))
        };
        self.inner.write(0, bytes)?;
        Ok(())
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
        Ok(Self {
            buff: Buffer::<Vertex>::upload_slice(device, data)?,
            _vert_in: PhantomData,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buff.buffer()
    }
}

define_buff_type!(Index, vk::BufferUsageFlags::INDEX_BUFFER);

fn buffer_usage(usage: vk::BufferUsageFlags) -> BufferUsages {
    let mut result = BufferUsages::NONE;
    for (vulkan, rhi) in [
        (vk::BufferUsageFlags::TRANSFER_SRC, BufferUsages::COPY_SRC),
        (vk::BufferUsageFlags::TRANSFER_DST, BufferUsages::COPY_DST),
        (vk::BufferUsageFlags::VERTEX_BUFFER, BufferUsages::VERTEX),
        (vk::BufferUsageFlags::INDEX_BUFFER, BufferUsages::INDEX),
        (vk::BufferUsageFlags::UNIFORM_BUFFER, BufferUsages::UNIFORM),
        (vk::BufferUsageFlags::STORAGE_BUFFER, BufferUsages::STORAGE),
    ] {
        if usage.contains(vulkan) {
            result |= rhi;
        }
    }
    result
}

fn memory_domain(location: MemoryLocation) -> MemoryDomain {
    match location {
        MemoryLocation::GpuOnly | MemoryLocation::Unknown => MemoryDomain::Device,
        MemoryLocation::CpuToGpu => MemoryDomain::Upload,
        MemoryLocation::GpuToCpu => MemoryDomain::Readback,
    }
}
