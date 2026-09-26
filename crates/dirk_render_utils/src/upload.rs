//! Batched uploads with an explicit transfer-to-graphics handoff.
use dirk_rhi::{
    Buffer, BufferCopy, BufferDesc, BufferImageCopy, BufferUsages, CommandEncoder, Completion,
    CopyQueue, DependencyInfo, Graphics, Image, ImageBarrier, ImageState, ImageSubresourceRange,
    MemoryDomain, Origin3d, QueueTransfer, QueueType, RecordedCommands, Result, Rhi, ShaderStages,
    SubmitInfo, UploadLayout,
};

/// One tick's uploads. Native copies record immediately; staging drops into the current cycle.
#[derive(Default)]
pub struct UploadBatch {
    encoders: Option<UploadEncoders>,
}

struct UploadEncoders {
    transfer: CommandEncoder<CopyQueue>,
    acquire: CommandEncoder<Graphics>,
}
impl UploadBatch {
    /// Starts a batch independently of renderer state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn encoders(&mut self, rhi: &Rhi) -> Result<&mut UploadEncoders> {
        if self.encoders.is_none() {
            self.encoders = Some(UploadEncoders {
                transfer: rhi.create_encoder("uploads")?,
                acquire: rhi.create_encoder("upload acquisitions")?,
            });
        }
        Ok(self
            .encoders
            .as_mut()
            .expect("upload encoders were created"))
    }
    /// Allocates and uploads immutable buffer data for subsequent graphics use.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn buffer(
        &mut self,
        rhi: &Rhi,
        bytes: &[u8],
        usage: BufferUsages,
        access: ImageState,
    ) -> Result<Buffer> {
        let mut staging = Self::staging(rhi, bytes)?;
        // SAFETY: this allocation has not been submitted and is exclusively borrowed.
        unsafe {
            staging.write(0, bytes)?;
        }
        let buffer = rhi.create_buffer(&BufferDesc {
            label: "uploaded buffer",
            size: bytes.len() as u64,
            usage: usage | BufferUsages::COPY_DST,
            memory: MemoryDomain::Device,
        })?;
        let encoders = self.encoders(rhi)?;
        // SAFETY: both ranges are fresh allocations, and both recordings submit in this cycle.
        unsafe {
            encoders.transfer.copy_buffer(
                &staging,
                &buffer,
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes.len() as u64,
                }],
            )?;
            let barrier = [dirk_rhi::BufferBarrier {
                buffer: &buffer,
                offset: 0,
                size: buffer.size(),
                old_state: ImageState::CopyDestination,
                new_state: access,
                queue_transfer: Some(Self::handoff()),
            }];
            let dependency = DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &barrier,
                image_barriers: &[],
            };
            encoders.transfer.barrier(&dependency)?;
            encoders.acquire.barrier(&dependency)?;
        }
        Ok(buffer)
    }
    /// Uploads all mip levels of a new color image and makes it readable by fragment shaders.
    ///
    /// # Safety
    /// The image must be uninitialized, not in GPU use, and remain alive through all subsequent uses.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn image(&mut self, rhi: &Rhi, image: &Image, levels: &[&[u8]]) -> Result<()> {
        if levels.len() != image.description().mip_levels as usize {
            return Err(dirk_rhi::InvalidResourceKind::Mismatch.into());
        }
        let encoders = self.encoders(rhi)?;
        unsafe {
            Self::image_barrier(
                &mut encoders.transfer,
                image,
                ImageState::Undefined,
                ImageState::CopyDestination,
                None,
            )?;
        }
        for (mip, pixels) in levels.iter().enumerate() {
            let mip = u32::try_from(mip).map_err(|_| dirk_rhi::InvalidResourceKind::OutOfRange)?;
            let extent = image.description().mip_extent(mip)?;
            unsafe {
                ImageUpload {
                    image,
                    mip,
                    origin: Origin3d::default(),
                    extent,
                    pixels,
                }
                .record(rhi, &mut encoders.transfer)?;
            }
        }
        let final_state = ImageState::ShaderRead(ShaderStages::FRAGMENT);
        unsafe {
            Self::image_barrier(
                &mut encoders.transfer,
                image,
                ImageState::CopyDestination,
                final_state,
                Some(Self::handoff()),
            )?;
            Self::image_barrier(
                &mut encoders.acquire,
                image,
                ImageState::CopyDestination,
                final_state,
                Some(Self::handoff()),
            )?;
        }
        Ok(())
    }
    /// Submits the transfer batch and returns graphics acquisitions and their GPU dependency.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn submit(self, rhi: &mut Rhi) -> Result<Option<(RecordedCommands, Completion)>> {
        let Some(encoders) = self.encoders else {
            return Ok(None);
        };
        let transfer = encoders.transfer.finish()?;
        let acquire = encoders.acquire.finish()?;
        // SAFETY: this batch records complete copy dependencies and submits once in its recording cycle.
        let completion = unsafe {
            rhi.queue::<CopyQueue>()
                .submit(vec![transfer], &SubmitInfo::default())?
        };
        Ok(Some((acquire, completion)))
    }
    /// Finishes initialization uploads, including ownership acquisition, before returning.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn finish(self, rhi: &mut Rhi) -> Result<()> {
        if let Some((acquire, transfer)) = self.submit(rhi)? {
            // SAFETY: acquisition waits for the matching, already-submitted release.
            unsafe {
                rhi.queue::<Graphics>().submit(
                    vec![acquire],
                    &SubmitInfo {
                        surface_frames: &[],
                        wait_for: &[&transfer],
                    },
                )?
            }
            .wait(u64::MAX)?;
        }
        Ok(())
    }
    fn staging(rhi: &Rhi, bytes: &[u8]) -> Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "upload staging",
            size: bytes.len() as u64,
            usage: BufferUsages::COPY_SRC,
            memory: MemoryDomain::Upload,
        })
    }
    fn handoff() -> QueueTransfer {
        QueueTransfer {
            source: QueueType::Copy,
            destination: QueueType::Graphics,
        }
    }
    unsafe fn image_barrier<Q: dirk_rhi::QueueKind>(
        cmd: &mut CommandEncoder<Q>,
        image: &Image,
        old_state: ImageState,
        new_state: ImageState,
        queue_transfer: Option<QueueTransfer>,
    ) -> Result<()> {
        let range = ImageSubresourceRange::WHOLE.resolve(&image.description())?;
        unsafe {
            cmd.barrier(&DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[ImageBarrier {
                    image,
                    old_state,
                    new_state,
                    aspects: range.aspects,
                    base_mip_level: 0,
                    mip_level_count: range.mip_level_count,
                    base_array_layer: 0,
                    array_layer_count: range.array_layer_count,
                    queue_transfer,
                }],
            })
        }
    }
}

/// One packed color region; the caller supplies its explicit image dependencies.
pub struct ImageUpload<'a> {
    /// Destination allocation.
    pub image: &'a Image,
    /// Destination mip.
    pub mip: u32,
    /// Destination texel origin.
    pub origin: Origin3d,
    /// Region extent.
    pub extent: dirk_rhi::Extent3d,
    /// Tightly packed source texels.
    pub pixels: &'a [u8],
}
impl ImageUpload<'_> {
    /// Packs aligned staging memory and records a copy into an already transitioned image.
    ///
    /// # Safety
    /// The image must be in `CopyDestination` state, ordered against prior uses, and alive through GPU use.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn record<Q: dirk_rhi::QueueKind>(
        &self,
        rhi: &Rhi,
        command: &mut CommandEncoder<Q>,
    ) -> Result<()> {
        let layout = UploadLayout::new(
            self.extent.width,
            self.extent.height,
            self.image.description().format,
            rhi.capabilities(),
        )?;
        let pixels = layout.pack(self.pixels)?;
        let mut staging = UploadBatch::staging(rhi, &pixels)?;
        unsafe {
            staging.write(0, &pixels)?;
            command.copy_buffer_to_image(
                &staging,
                self.image,
                &[BufferImageCopy {
                    buffer_offset: 0,
                    buffer_bytes_per_row: layout.bytes_per_row,
                    buffer_rows_per_image: layout.rows,
                    mip_level: self.mip,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    image_origin: self.origin,
                    extent: self.extent,
                    aspects: dirk_rhi::ImageAspects::COLOR,
                }],
            )?;
        }
        Ok(())
    }
}
