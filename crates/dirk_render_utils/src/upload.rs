//! Staged uploads: host data copied into device-local buffers and images.

use std::fmt::Display;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CommandEncoder, Completion, CopyQueue, Error, Frame, Graphics, Image, ImageAccess,
    ImageBarrier, ImageDesc, ImageInfo, ImageRange, ImageRegion, ImageUsage, Memory, PerFrame,
    QueueTransfer, QueueType, Result, Rhi,
};

/// Buffer copies move whole words: their offsets and sizes are multiples of
/// this many bytes.
const WORD: u64 = 4;

/// Uploads host data into new device-local buffers and images, and into
/// regions of existing images, through one persistently mapped staging
/// buffer per frame slot.
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
/// let mut frame = rhi.begin_frame()?;
/// let mut uploads = uploader.begin(&mut frame);
/// let read = BufferAccess::ShaderRead(ShaderStages::VERTEX);
/// let mesh = uploads.buffer("mesh", BufferUsage::STORAGE, read, vertices)?;
/// uploads.submit()?;
/// // Graphics work submitted from here on reads `mesh`.
/// # Ok(())
/// # }
/// ```
pub struct Uploader {
    /// Staging for the copies into new resources.
    staging: Staging,
    /// Staging for writes into existing images, which the graphics queue
    /// copies, when new resources are copied on the dedicated copy queue:
    /// each queue family reads staging buffers of its own.
    writes: Option<Staging>,
    /// The frame whose data the staging buffers hold.
    frame: u64,
}

/// One persistently mapped staging buffer per frame slot.
struct Staging {
    buffers: PerFrame<Buffer>,
    /// Bytes of the current frame's buffer holding its data.
    used: u64,
}

/// One frame's uploads, recorded as they are requested and submitted
/// together by [`Uploads::submit`]. Dropping the batch discards them.
///
/// Copies run on the dedicated copy queue when the device has one, which
/// then hands the resources over to the graphics queue; otherwise they run on
/// the graphics queue. Either way, graphics work submitted after the batch
/// sees the uploaded data in its requested access, without waiting.
///
/// The batch borrows its frame mutably, so no other encoder can be created
/// or submitted until it is submitted or dropped: graphics work using the
/// uploads cannot be submitted before them.
///
/// ```compile_fail
/// # use dirk_render_utils::Uploader;
/// # use dirk_rhi::{Graphics, Result, Rhi};
/// # fn upload(rhi: &mut Rhi, uploader: &mut Uploader) -> Result<()> {
/// let mut frame = rhi.begin_frame()?;
/// let uploads = uploader.begin(&mut frame);
/// let encoder = frame.encoder::<Graphics>("too early")?;
/// uploads.submit()?;
/// # Ok(())
/// # }
/// ```
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
        let dedicated = rhi.capabilities().dedicated_copy_queue;
        Ok(Self {
            staging: Staging::new(rhi, capacity)?,
            writes: dedicated.then(|| Staging::new(rhi, capacity)).transpose()?,
            frame: 0,
        })
    }

    /// Begins a batch of uploads in `frame`, which it borrows until it is
    /// submitted. A frame may have several.
    ///
    /// # Panics
    ///
    /// Uploading panics if `frame` belongs to another RHI than the one the
    /// uploader was created for.
    pub fn begin<'f>(&'f mut self, frame: &'f mut Frame<'_>) -> Uploads<'f> {
        if self.frame != frame.number() {
            // Beginning the frame waited for the work of the frame that last
            // used its slot's staging buffers.
            self.frame = frame.number();
            for staging in std::iter::once(&mut self.staging).chain(&mut self.writes) {
                staging.used = 0;
            }
        }
        Uploads {
            uploader: self,
            frame,
            recording: None,
        }
    }

    /// Stages `data` for copies into new resources; see [`Staging::stage`].
    fn stage(&mut self, frame: &Frame<'_>, data: &[u8], size: u64, alignment: u64) -> Result<u64> {
        self.staging.stage(frame, data, size, alignment)
    }
}

impl Staging {
    /// Creates staging buffers of `capacity` bytes.
    fn new(rhi: &Rhi, capacity: u64) -> Result<Self> {
        Ok(Self {
            buffers: PerFrame::try_new(rhi, |_| Self::create(rhi, capacity))?,
            used: 0,
        })
    }

    /// `frame`'s staging buffer.
    fn get(&self, frame: &Frame<'_>) -> &Buffer {
        self.buffers.get(frame)
    }

    /// Reserves `size` bytes of `frame`'s staging buffer at a multiple of
    /// `alignment`, copies `data`, at most `size` bytes, to their start and
    /// returns their offset. Replaces a full staging buffer with a larger one.
    fn stage(&mut self, frame: &Frame<'_>, data: &[u8], size: u64, alignment: u64) -> Result<u64> {
        let staging = self.buffers.get_mut(frame);
        let capacity = staging.info().size;
        let mut offset = self.used.next_multiple_of(alignment);
        if offset + size > capacity {
            // Copies recorded earlier in this frame keep the old buffer alive
            // until they complete, as dropping it defers its free.
            let limit = frame.rhi().capabilities().limits.max_buffer_size;
            *staging = Self::create(frame.rhi(), (2 * capacity).min(limit).max(size))?;
            offset = 0;
        }
        self.used = offset + size;
        self.buffers.write(frame, offset, data)?;
        Ok(offset)
    }

    /// Creates a staging buffer of `size` bytes.
    fn create(rhi: &Rhi, size: u64) -> Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "upload staging",
            size,
            usage: BufferUsage::COPY_SRC,
            memory: Memory::Upload,
        })
    }
}

impl<'f> Uploads<'f> {
    /// The RHI the batch's frame belongs to, to create other objects while
    /// the batch borrows the frame.
    #[must_use]
    pub fn rhi(&self) -> &'f Rhi {
        self.frame.rhi()
    }

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

    /// Writes `texels` into `region` of `image`, keeping the rest of it.
    /// Graphics work submitted before this batch leaves the region in
    /// `access`, and graphics work submitted after it finds it there again.
    /// `texels` holds the region's texels tightly packed, as for
    /// [`Uploads::image`].
    ///
    /// The graphics queue copies them, after the batch's other copies.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `texels`' length differs from the
    /// region's texels, `region` is not inside `image`, `image` lacks
    /// [`ImageUsage::COPY_DST`] or the usage `access` needs, or `access` is
    /// [`ImageAccess::Undefined`].
    pub fn write_image(
        &mut self,
        image: &Image,
        access: ImageAccess,
        region: ImageRegion,
        texels: &[u8],
    ) -> Result<()> {
        let info = image.info();
        Self::check_texels(info, image.label(), &region, texels, "the region")?;
        let staging = match &mut self.uploader.writes {
            Some(writes) => writes,
            None => &mut self.uploader.staging,
        };
        let size = texels.len() as u64;
        let offset = staging.stage(
            self.frame,
            texels,
            size,
            info.format.copy_offset_alignment(),
        )?;
        let copy = BufferImageCopy::packed(info.format, offset, region);
        let encoder = Recording::get(&mut self.recording, self.frame)?.graphics();
        let range = ImageRange {
            base_mip_level: region.mip_level,
            mip_level_count: 1,
            base_layer: region.base_layer,
            layer_count: region.layer_count,
        };
        let barrier = |before, after| {
            [ImageBarrier {
                image,
                range,
                before,
                after,
                transfer: None,
            }]
        };
        // SAFETY: graphics work before the copy leaves the region in
        // `access`, and on the graphics queue: new images are handed over
        // to it before.
        unsafe { encoder.barrier(&[], &barrier(access, ImageAccess::CopyDestination))? };
        encoder.copy_buffer_to_image(staging.get(self.frame), image, &[copy])?;
        // SAFETY: the copy left the region in `CopyDestination`.
        unsafe { encoder.barrier(&[], &barrier(ImageAccess::CopyDestination, access)) }
    }

    /// Creates an image from `desc` holding `mips`, which graphics work
    /// submitted after this batch reads through `access`. `desc.usage` gains
    /// [`ImageUsage::COPY_DST`].
    ///
    /// `mips[level]` holds mip `level`'s texels tightly packed: rows, then
    /// depth slices, then array layers. Mips past the last one given are left
    /// undefined.
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
        let image = self.copy_image(desc, mips)?;
        Recording::get(&mut self.recording, self.frame)?.finish_image(&image, access)?;
        Ok(image)
    }

    /// Creates an image from `desc` holding `texels` in mip 0, packed as
    /// for [`Uploads::image`], and generates its other mips on the graphics
    /// queue by filtering mip 0 down. Graphics work submitted after this
    /// batch reads it through `access`. `desc.usage` gains
    /// [`ImageUsage::COPY_SRC`] and [`ImageUsage::COPY_DST`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Uploads::image`], or [`Error::Invalid`] if the
    /// format cannot be blitted with linear filtering.
    pub fn mipmapped_image(
        &mut self,
        desc: &ImageDesc<'_>,
        access: ImageAccess,
        texels: &[u8],
    ) -> Result<Image> {
        let desc = ImageDesc {
            usage: desc.usage | ImageUsage::COPY_SRC,
            ..*desc
        };
        let image = self.copy_image(&desc, &[texels])?;
        let recording = Recording::get(&mut self.recording, self.frame)?;
        // Hands mip 0 over to the graphics queue, which blits the others.
        recording.finish_image(&image, ImageAccess::CopyDestination)?;
        recording
            .graphics()
            .generate_mips(&image, ImageAccess::CopyDestination, access)?;
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

    /// Creates an image from `desc`, with [`ImageUsage::COPY_DST`], and
    /// records copies of `mips` into it, leaving it in
    /// [`ImageAccess::CopyDestination`] on the queue copying.
    fn copy_image(&mut self, desc: &ImageDesc<'_>, mips: &[&[u8]]) -> Result<Image> {
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
        Ok(image)
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
            let region = info.mip_region(level);
            Self::check_texels(info, label, &region, data, format_args!("mip {level}"))?;
        }
        Ok(())
    }

    /// Checks that `data` holds every texel of `region` of the image `info`
    /// describes, tightly packed. `name` names the region in errors.
    fn check_texels(
        info: &ImageInfo,
        label: &str,
        region: &ImageRegion,
        data: &[u8],
        name: impl Display,
    ) -> Result<()> {
        let extent = region.extent;
        let expected = [
            extent.width,
            extent.height,
            extent.depth,
            region.layer_count,
            info.format.bytes_per_texel(),
        ]
        .into_iter()
        .fold(1_u64, |size, factor| size.saturating_mul(u64::from(factor)));
        if data.len() as u64 != expected {
            return Err(Error::Invalid {
                object: "image upload",
                label: label.to_owned(),
                reason: format!("{name} holds {} bytes instead of {expected}", data.len()),
            });
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

    /// The encoder recording graphics work after the batch's copies.
    fn graphics(&mut self) -> &mut CommandEncoder<'f, Graphics> {
        match self {
            Self::Transfer { acquires, .. } => acquires,
            Self::Graphics(encoder) => encoder,
        }
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
