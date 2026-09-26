//! Typed host records and vertex data; raw allocations remain RHI resources.
use crate::upload::UploadBatch;
use bytemuck::NoUninit;
use dirk_rhi::{
    Buffer, BufferDesc, BufferUsages, MemoryDomain, Result, Rhi, VertexAttribute,
    VertexBufferLayout, VertexStepMode,
};
use std::marker::PhantomData;

/// Host vertex record with its authoritative input layout.
pub trait VertexInput: NoUninit {
    /// Attributes inside one vertex.
    const ATTRIBUTES: &'static [VertexAttribute];
    /// Describes the complete host record.
    ///
    /// # Panics
    /// Panics if the host record is larger than the RHI maximum vertex stride representation.
    #[must_use]
    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            stride: u32::try_from(size_of::<Self>()).expect("vertex stride fits u32"),
            step_mode: VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}
/// Vertex allocation whose record type matches a typed pipeline input.
pub struct VertexBuffer<I: VertexInput> {
    buffer: Buffer,
    input: PhantomData<I>,
}
impl<I: VertexInput> VertexBuffer<I> {
    /// Adds immutable vertices to the current upload batch.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn upload(rhi: &Rhi, uploads: &mut UploadBatch, vertices: &[I]) -> Result<Self> {
        Ok(Self {
            buffer: uploads.buffer(
                rhi,
                bytemuck::cast_slice(vertices),
                BufferUsages::VERTEX,
                dirk_rhi::ResourceAccess::Vertex,
            )?,
            input: PhantomData,
        })
    }
    /// Borrows the underlying allocation for recording.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}
/// One host-visible uniform record. GPU layout is validated independently by shader reflection.
pub struct UniformBuffer<T: NoUninit> {
    buffer: Buffer,
    record: PhantomData<T>,
}
impl<T: NoUninit> UniformBuffer<T> {
    /// Allocates a single record.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn new(rhi: &Rhi) -> Result<Self> {
        Ok(Self {
            buffer: rhi.create_buffer(&BufferDesc {
                label: "uniform record",
                size: size_of::<T>() as u64,
                usage: BufferUsages::UNIFORM,
                memory: MemoryDomain::Upload,
            })?,
            record: PhantomData,
        })
    }
    /// Borrows the allocation for descriptor creation and graph declarations.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    /// Writes this frame slot's record.
    ///
    /// # Safety
    /// All GPU work reading this slot must have completed.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn write(&mut self, data: &T) -> Result<()> {
        unsafe { self.buffer.write(0, bytemuck::bytes_of(data)) }
    }
}
