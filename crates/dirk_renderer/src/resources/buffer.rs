//! Typed renderer buffers allocated through the RHI.

use std::{marker::PhantomData, mem::size_of_val};

use dirk_rhi::{
    Backend as _, Buffer as _, BufferCopy, BufferDesc, BufferUsages, CommandBuffer as _,
    MemoryDomain,
};

use crate::{
    Result,
    resources::{ActiveBuffer, device::RenderDevice},
    shaders::metadata::VertexInput,
};

pub struct Buffer<Type: BuffType = Custom> {
    inner: ActiveBuffer,
    _type: PhantomData<Type>,
}

pub trait BuffType {
    fn usage() -> BufferUsages;
}

macro_rules! define_buff_type {
    ($name:ident, $usage:expr) => {
        pub struct $name;

        impl BuffType for $name {
            fn usage() -> BufferUsages {
                $usage
            }
        }

        pastey::paste! {
            pub type [<$name Buffer>] = Buffer<$name>;
        }
    };
}

define_buff_type!(Custom, BufferUsages::STORAGE);

impl<Type: BuffType> Buffer<Type> {
    pub fn create_custom(
        device: &RenderDevice,
        size: u64,
        usage: BufferUsages,
        memory: MemoryDomain,
    ) -> Result<Self> {
        let inner = device.rhi.create_buffer(&BufferDesc {
            label: "renderer buffer",
            size,
            usage,
            memory,
        })?;
        Ok(Self {
            inner,
            _type: PhantomData,
        })
    }

    pub fn create(device: &RenderDevice, size: u64, memory: MemoryDomain) -> Result<Self> {
        Self::create_custom(device, size, Type::usage(), memory)
    }

    pub(crate) fn rhi(&self) -> &ActiveBuffer {
        &self.inner
    }

    pub fn upload_slice<T: Copy>(device: &RenderDevice, data: &[T]) -> Result<Self> {
        let size = u64::try_from(size_of_val(data))
            .map_err(|_| dirk_rhi::Error::InvalidResource(dirk_rhi::InvalidResource::OutOfRange))?;
        let staging = Buffer::<Custom>::create_custom(
            device,
            size,
            BufferUsages::COPY_SRC,
            MemoryDomain::Upload,
        )?;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
        staging.rhi().write(0, bytes)?;

        let output = Self::create_custom(
            device,
            size,
            BufferUsages::COPY_DST | Type::usage(),
            MemoryDomain::Device,
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

define_buff_type!(Uniform, BufferUsages::UNIFORM);

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
    fn usage() -> BufferUsages {
        BufferUsages::VERTEX
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

    pub(crate) fn buffer(&self) -> &ActiveBuffer {
        self.buff.rhi()
    }
}

define_buff_type!(Index, BufferUsages::INDEX);
