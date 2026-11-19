use bitflags::bitflags;

use crate::{Error, Result, Rhi, native};

bitflags! {
    /// Ways a buffer may be used. Declared when the buffer is created.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        /// Source of copies.
        const COPY_SRC = 1 << 0;
        /// Destination of copies.
        const COPY_DST = 1 << 1;
        /// Index buffer of indexed draws.
        const INDEX = 1 << 2;
        /// Read and written by shaders as a storage buffer.
        const STORAGE = 1 << 3;
    }
}

/// Where a buffer's memory lives, which decides whether the host can access
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Memory {
    /// Device-local memory the host cannot access. Fastest for the GPU.
    Device,
    /// Host-visible memory written by the host and read by the GPU.
    Upload,
    /// Host-visible, cached memory written by the GPU and read by the host.
    Readback,
}

/// Describes a buffer to create.
#[derive(Clone, Copy, Debug)]
pub struct BufferDesc<'a> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// Size in bytes.
    pub size: u64,
    /// Ways the buffer may be used.
    pub usage: BufferUsage,
    /// Where the buffer's memory lives.
    pub memory: Memory,
}

/// Immutable properties of a [`Buffer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferInfo {
    /// Size in bytes.
    pub size: u64,
    /// Ways the buffer may be used.
    pub usage: BufferUsage,
    /// Where the buffer's memory lives.
    pub memory: Memory,
}

/// A GPU buffer. Destroyed when dropped.
pub struct Buffer {
    pub(crate) raw: native::Buffer,
    info: BufferInfo,
    label: Box<str>,
}

impl Rhi {
    /// Creates a buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the buffer is empty, has no usage or
    /// exceeds the device's buffer size limit, or a native error if memory
    /// cannot be allocated.
    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<Buffer> {
        let invalid = |reason| Error::invalid("buffer", desc.label, reason);
        let limit = self.capabilities().limits.max_buffer_size;
        if desc.size == 0 {
            return Err(invalid("size is zero".to_owned()));
        }
        if desc.size > limit {
            return Err(invalid(format!(
                "size {} exceeds the device limit of {limit} bytes",
                desc.size
            )));
        }
        if desc.usage.is_empty() {
            return Err(invalid("usage is empty".to_owned()));
        }
        Ok(Buffer {
            raw: native::Buffer::new(&self.device, desc)?,
            info: BufferInfo {
                size: desc.size,
                usage: desc.usage,
                memory: desc.memory,
            },
            label: desc.label.into(),
        })
    }
}

impl Buffer {
    /// The buffer's immutable properties.
    #[must_use]
    pub fn info(&self) -> &BufferInfo {
        &self.info
    }

    /// The buffer's debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Copies `data` into host-visible memory at `offset`.
    ///
    /// # Safety
    ///
    /// The GPU must not access the written range until the write is done:
    /// work that used it must have completed. Work submitted afterwards sees
    /// the write.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the buffer's memory is not host-visible
    /// or the range exceeds the buffer, or a native error if non-coherent
    /// memory cannot be flushed.
    pub unsafe fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.check_host_range("write", offset, data.len())?;
        // SAFETY: the range is in bounds of host-visible memory, `&mut self`
        // excludes other host access and the caller excludes GPU access.
        unsafe { self.raw.write(offset, data) }
    }

    /// Copies host-visible memory at `offset` into `data`.
    ///
    /// # Safety
    ///
    /// GPU writes to the read range must have completed and been made visible
    /// to the host, with a barrier to host reads before the work completed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the buffer's memory is not host-visible
    /// or the range exceeds the buffer, or a native error if non-coherent
    /// memory cannot be invalidated.
    pub unsafe fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        self.check_host_range("read", offset, data.len())?;
        // SAFETY: the range is in bounds of host-visible memory and the caller
        // excludes GPU writes.
        unsafe { self.raw.read(offset, data) }
    }

    /// Checks that the host can access `len` bytes at `offset`.
    fn check_host_range(&self, access: &str, offset: u64, len: usize) -> Result<()> {
        let invalid = |reason| Error::invalid("buffer", &self.label, reason);
        if self.info.memory == Memory::Device {
            return Err(invalid(format!(
                "host {access} of device-local memory, which the host cannot access"
            )));
        }
        let end = u64::try_from(len)
            .ok()
            .and_then(|len| offset.checked_add(len));
        if end.is_none_or(|end| end > self.info.size) {
            return Err(invalid(format!(
                "host {access} of {len} bytes at offset {offset} exceeds its {} bytes",
                self.info.size
            )));
        }
        Ok(())
    }
}
