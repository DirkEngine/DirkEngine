//! Staged uploads: host data copied into new device-local buffers and images.

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CommandEncoder, Completion, CopyQueue, Error, Frame, Graphics, Image, ImageAccess,
    ImageBarrier, ImageDesc, ImageInfo, ImageRange, ImageUsage, Memory, PerFrame, QueueTransfer,
    QueueType, Result, Rhi,
};

/// Buffer copies move whole words: their offsets and sizes are multiples of
/// this many bytes.
const WORD: u64 = 4;

/// Uploads host data into new device-local buffers and images through one
/// persistently mapped staging buffer per frame slot.
///
/// Each frame packs its data straight into its slot's staging buffer, which
/// the slot's previous frame no longer uses. When it is full, it is replaced
/// by a larger one; the old buffer lives until the copies reading it
/// complete.
///
/// ```no_run
/// # use dirk_render_utils::Uploader;
/// # use dirk_rhi::{BufferAccess, BufferUsage, Result, Rhi, ShaderStages};
/// # fn upload(rhi: &mut Rhi, uploader: &mut Uploader, vertices: &[u8]) -> Result<()> {
/// let frame = rhi.begin_frame()?;
/// let mut uploads = uploader.begin(&frame);
/// let read = BufferAccess::ShaderRead(ShaderStages::VERTEX);
/// let mesh = uploads.buffer("mesh", BufferUsage::STORAGE, read, vertices)?;
/// uploads.submit()?;
/// // Graphics work submitted from here on reads `mesh`.
/// # Ok(())
/// # }
/// ```
pub struct Uploader {
    /// Each frame slot's staging buffer.
    staging: PerFrame<Buffer>,
    /// The frame whose data `used` counts.
    frame: u64,
    /// Bytes of that frame's staging buffer holding its data.
    used: u64,
}

/// One frame's uploads, recorded as they are requested and submitted
/// together by [`Uploads::submit`]. Dropping the batch discards them.
///
/// Copies run on the dedicated copy queue when the device has one, which
/// then hands the resources over to the graphics queue; otherwise they run on
/// the graphics queue. Either way, graphics work submitted after the batch
/// sees the uploaded data in its requested access, without waiting.
#[must_use = "uploads are discarded unless submitted"]
pub struct Uploads<'f> {
    uploader: &'f mut Uploader,
    frame: &'f Frame<'f>,
    /// Begun by the first upload.
    recording: Option<Recording<'f>>,
}

/// The commands a batch records.
enum Recording<'f> {
    /// Copies on the dedicated copy queue, which releases the resources to
    /// the graphics queue; `acquires` acquires them there.
    Transfer {
        copies: CommandEncoder<'f, CopyQueue>,
        acquires: CommandEncoder<'f, Graphics>,
    },
    /// Copies on the graphics queue.
    Graphics(CommandEncoder<'f, Graphics>),
}

/// Evaluates `$body` with `$encoder` bound to the encoder recording a
/// batch's copies, whichever queue it records for.
macro_rules! with_copies {
    ($recording:expr, $encoder:ident => $body:expr) => {
        match $recording {
            Recording::Transfer {
                copies: $encoder, ..
            } => $body,
            Recording::Graphics($encoder) => $body,
        }
    };
}

impl Uploader {
    /// Creates an uploader whose staging buffers start with `capacity` bytes
    /// each and grow on demand.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `capacity` is zero or exceeds the
    /// device's buffer size limit, or a native error if memory cannot be
    /// allocated.
    pub fn new(rhi: &Rhi, capacity: u64) -> Result<Self> {
        Ok(Self {
            staging: PerFrame::try_new(rhi, |_| Self::create_staging(rhi, capacity))?,
            frame: 0,
            used: 0,
        })
    }

    /// Begins a batch of uploads in `frame`. A frame may have several.
    ///
    /// # Panics
    ///
    /// Uploading panics if `frame` belongs to another RHI than the one the
    /// uploader was created for.
    pub fn begin<'f>(&'f mut self, frame: &'f Frame<'f>) -> Uploads<'f> {
        if self.frame != frame.number() {
            // Beginning the frame waited for the work of the frame that last
            // used its slot's staging buffer.
            self.frame = frame.number();
            self.used = 0;
        }
        Uploads {
            uploader: self,
            frame,
            recording: None,
        }
    }

    /// Reserves `size` bytes of `frame`'s staging buffer at a multiple of
    /// `alignment`, copies `data`, at most `size` bytes, to their start and
    /// returns their offset. Replaces a full staging buffer with a larger one.
    fn stage(&mut self, frame: &Frame<'_>, data: &[u8], size: u64, alignment: u64) -> Result<u64> {
        let staging = self.staging.get_mut(frame);
        let capacity = staging.info().size;
        let mut offset = self.used.next_multiple_of(alignment);
        if offset + size > capacity {
            // Copies recorded earlier in this frame keep the old buffer alive
            // until they complete, as dropping it defers its free.
            let limit = frame.rhi().capabilities().limits.max_buffer_size;
            *staging = Self::create_staging(frame.rhi(), (2 * capacity).min(limit).max(size))?;
            offset = 0;
        }
        self.used = offset + size;
        self.staging.write(frame, offset, data)?;
        Ok(offset)
    }

    /// Creates a staging buffer of `size` bytes.
    fn create_staging(rhi: &Rhi, size: u64) -> Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "upload staging",
            size,
            usage: BufferUsage::COPY_SRC,
            memory: Memory::Upload,
        })
    }
}

impl Uploads<'_> {
    /// Creates a device-local buffer holding `data`, which graphics work
    /// submitted after this batch reads through `access`. `usage` gains
    /// [`BufferUsage::COPY_DST`].
    ///
    /// Copies move whole four-byte words, so the buffer's size is `data`'s
    /// length rounded up to a multiple of four; the padding is unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `data` is empty or exceeds the device's
    /// buffer size limit, or `usage` lacks the usage `access` needs, or a
    /// native error if memory cannot be allocated.
    pub fn buffer(
        &mut self,
        label: &str,
        usage: BufferUsage,
        access: BufferAccess,
        data: &[u8],
    ) -> Result<Buffer> {
        let size = (data.len() as u64).next_multiple_of(WORD);
        let buffer = self.frame.rhi().create_buffer(&BufferDesc {
            label,
            size,
            usage: usage | BufferUsage::COPY_DST,
            memory: Memory::Device,
        })?;
        let src_offset = self.uploader.stage(self.frame, data, size, WORD)?;
        let recording = Recording::get(&mut self.recording, self.frame)?;
        let staging = self.uploader.staging.get(self.frame);
        let copy = BufferCopy {
            src_offset,
            dst_offset: 0,
            size,
        };
        with_copies!(recording, encoder => encoder.copy_buffer(staging, &buffer, &[copy]))?;
        recording.finish_buffer(&buffer, access)?;
        Ok(buffer)
    }

    /// Creates an image from `desc` holding `mips`, which graphics work
    /// submitted after this batch reads through `access`. `desc.usage` gains
    /// [`ImageUsage::COPY_DST`].
    ///
    /// `mips[level]` holds mip `level`'s texels tightly packed: rows, then
    /// depth slices, then array layers. Mips past the last one given are left
    /// undefined, for example to be generated on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `desc` is invalid or its format cannot
    /// be copied into (depth/stencil or multisampled images), if no mips or
    /// more than the image has are given, or a mip's length differs from its
    /// texels'; or a native error if memory cannot be allocated.
    pub fn image(
        &mut self,
        desc: &ImageDesc<'_>,
        access: ImageAccess,
        mips: &[&[u8]],
    ) -> Result<Image> {
        let image = self.frame.rhi().create_image(&ImageDesc {
            usage: desc.usage | ImageUsage::COPY_DST,
            ..*desc
        })?;
        let info = *image.info();
        Self::check_mips(&info, desc.label, mips)?;
        let recording = Recording::get(&mut self.recording, self.frame)?;
        recording.begin_image(&image)?;
        let alignment = info.format.copy_offset_alignment();
        for (level, data) in (0..).zip(mips) {
            let size = data.len() as u64;
            let offset = self.uploader.stage(self.frame, data, size, alignment)?;
            let staging = self.uploader.staging.get(self.frame);
            let copy = BufferImageCopy::packed(info.format, offset, info.mip_region(level));
            with_copies!(recording, encoder => {
                encoder.copy_buffer_to_image(staging, &image, &[copy])
            })?;
        }
        recording.finish_image(&image, access)?;
        Ok(image)
    }

    /// Submits the batch. Graphics work submitted afterwards sees every
    /// upload in its requested access without waiting.
    ///
    /// Returns the graphics queue's completion after which the uploads are
    /// done, or `None` if the batch is empty.
    ///
    /// # Errors
    ///
    /// Returns a native error, such as
    /// [`Error::DeviceLost`](dirk_rhi::Error::DeviceLost), if submission
    /// fails.
    pub fn submit(self) -> Result<Option<Completion>> {
        let Some(recording) = self.recording else {
            return Ok(None);
        };
        let done = match recording {
            Recording::Transfer {
                copies,
                mut acquires,
            } => {
                acquires.wait_for(self.frame.submit(copies)?)?;
                self.frame.submit(acquires)?
            }
            Recording::Graphics(encoder) => self.frame.submit(encoder)?,
        };
        Ok(Some(done))
    }

    /// Checks that `mips` holds one to all of the mips of the image `info`
    /// describes, each with every texel of its layers.
    fn check_mips(info: &ImageInfo, label: &str, mips: &[&[u8]]) -> Result<()> {
        let invalid = |reason| {
            Err(Error::Invalid {
                object: "image upload",
                label: label.to_owned(),
                reason,
            })
        };
        if mips.is_empty() || mips.len() > info.mip_levels as usize {
            return invalid(format!(
                "{} mips given for an image of {}",
                mips.len(),
                info.mip_levels
            ));
        }
        for (level, data) in (0..).zip(mips) {
            let extent = info.extent.mip(level);
            // Image limits keep this far from overflowing.
            let texels = u64::from(extent.width)
                * u64::from(extent.height)
                * u64::from(extent.depth)
                * u64::from(info.array_layers);
            let expected = texels * u64::from(info.format.bytes_per_texel());
            if data.len() as u64 != expected {
                return invalid(format!(
                    "mip {level} holds {} bytes instead of {expected}",
                    data.len()
                ));
            }
        }
        Ok(())
    }
}

impl<'f> Recording<'f> {
    /// The recording in `slot`, begun on the dedicated copy queue if the
    /// device has one.
    fn get<'s>(slot: &'s mut Option<Self>, frame: &'f Frame<'f>) -> Result<&'s mut Self> {
        Ok(match slot {
            Some(recording) => recording,
            empty @ None => empty.insert(if frame.rhi().capabilities().dedicated_copy_queue {
                Self::Transfer {
                    copies: frame.encoder("uploads")?,
                    acquires: frame.encoder("upload acquisitions")?,
                }
            } else {
                Self::Graphics(frame.encoder("uploads")?)
            }),
        })
    }

    /// Prepares a new image for copies into it.
    fn begin_image(&mut self, image: &Image) -> Result<()> {
        let barrier = [ImageBarrier {
            image,
            range: ImageRange::ALL,
            before: ImageAccess::Undefined,
            after: ImageAccess::CopyDestination,
            transfer: None,
        }];
        // SAFETY: the batch just created the image; nothing accessed it.
        with_copies!(self, encoder => unsafe { encoder.barrier(&[], &barrier) })
    }

    /// Hands a buffer the batch created and copied into over to graphics
    /// work accessing it through `access`.
    fn finish_buffer(&mut self, buffer: &Buffer, access: BufferAccess) -> Result<()> {
        let barrier = [BufferBarrier {
            buffer,
            before: BufferAccess::CopyDestination,
            after: access,
            transfer: self.transfer(),
        }];
        // SAFETY: the batch created the buffer and only copied into it.
        unsafe { self.hand_over(&barrier, &[]) }
    }

    /// Hands an image the batch created and copied into over to graphics
    /// work accessing it through `access`.
    fn finish_image(&mut self, image: &Image, access: ImageAccess) -> Result<()> {
        let barrier = [ImageBarrier {
            image,
            range: ImageRange::ALL,
            before: ImageAccess::CopyDestination,
            after: access,
            transfer: self.transfer(),
        }];
        // SAFETY: the batch created the image, prepared it for copies and
        // only copied into it.
        unsafe { self.hand_over(&[], &barrier) }
    }

    /// Records barriers from copies to graphics work: released by the copy
    /// queue and acquired by the graphics queue, or on the graphics queue
    /// alone.
    ///
    /// # Safety
    ///
    /// Each barrier's `before` describes every access to its resource since
    /// its previous barrier, and its transfer is [`Self::transfer`].
    unsafe fn hand_over(
        &mut self,
        buffers: &[BufferBarrier<'_>],
        images: &[ImageBarrier<'_>],
    ) -> Result<()> {
        // SAFETY: the caller's barriers describe the accesses, and the
        // release and acquire halves of a transfer record the same barrier.
        unsafe {
            match self {
                Self::Transfer { copies, acquires } => {
                    copies.barrier(buffers, images)?;
                    acquires.barrier(buffers, images)
                }
                Self::Graphics(encoder) => encoder.barrier(buffers, images),
            }
        }
    }

    /// The ownership transfer from copies to graphics work, if they run on
    /// different queues.
    fn transfer(&self) -> Option<QueueTransfer> {
        matches!(self, Self::Transfer { .. }).then_some(QueueTransfer {
            from: QueueType::Copy,
            to: QueueType::Graphics,
        })
    }
}
