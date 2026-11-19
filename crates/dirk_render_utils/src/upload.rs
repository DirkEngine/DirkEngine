//! Staged uploads: host data copied into device-local buffers and images.

use std::fmt::Display;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CommandEncoder, Completion, CopyQueue, Error, Frame, FrameSlice, Graphics, Image, ImageAccess,
    ImageBarrier, ImageDesc, ImageInfo, ImageRange, ImageRegion, ImageUsage, Memory, QueueTransfer,
    QueueType, Result, Rhi,
};

/// Buffer copies move whole words: their offsets and sizes are multiples of
/// this many bytes.
const WORD: u64 = 4;

/// One frame's uploads: host data copied into new device-local buffers and
/// images, and into regions of existing images. The data is staged in the
/// frame's memory ([`Frame::alloc`]), which the frame's work alone reads,
/// so nothing grows. Uploads are recorded as they are requested and
/// submitted together by [`Uploads::submit`]. Dropping the batch discards
/// them: the buffers and images it returned stay usable, with undefined
/// contents, since a copy queue barrier takes over a new image's first
/// transition only once submitted (see
/// [`CommandEncoder::barrier`](dirk_rhi::CommandEncoder::barrier)), so the
/// graphics queue still moves it to its resting access.
///
/// Copies run on the dedicated copy queue when the device has one, which
/// then hands the resources over to the graphics queue; otherwise they run on
/// the graphics queue. Either way, graphics work submitted after the batch
/// sees the uploaded data, buffers in their requested access and images in
/// their resting access, without waiting.
///
/// ```no_run
/// # use dirk_render_utils::Uploads;
/// # use dirk_rhi::{BufferAccess, BufferUsage, Result, Rhi, ShaderStages};
/// # fn upload(rhi: &mut Rhi, vertices: &[u8]) -> Result<()> {
/// let mut frame = rhi.begin_frame()?;
/// let mut uploads = Uploads::new(&mut frame);
/// let read = BufferAccess::ShaderRead(ShaderStages::VERTEX);
/// let mesh = uploads.buffer("mesh", BufferUsage::STORAGE, read, vertices)?;
/// uploads.submit()?;
/// // Graphics work submitted from here on reads `mesh`.
/// # Ok(())
/// # }
/// ```
///
/// The batch borrows its frame mutably, so no other encoder can be created
/// or submitted until it is submitted or dropped: graphics work using the
/// uploads cannot be submitted before them.
///
/// ```compile_fail
/// # use dirk_render_utils::Uploads;
/// # use dirk_rhi::{Graphics, Result, Rhi};
/// # fn upload(rhi: &mut Rhi) -> Result<()> {
/// let mut frame = rhi.begin_frame()?;
/// let uploads = Uploads::new(&mut frame);
/// let encoder = frame.encoder::<Graphics>("too early")?;
/// uploads.submit()?;
/// # Ok(())
/// # }
/// ```
#[must_use = "uploads are discarded unless submitted"]
pub struct Uploads<'f> {
    frame: &'f Frame<'f>,
    /// Begun by the first upload.
    recording: Option<Recording<'f>>,
    /// Bytes staged so far.
    staged: u64,
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

impl<'f> Uploads<'f> {
    /// Begins a batch of uploads in `frame`, which it borrows until it is
    /// submitted. A frame may have several.
    pub fn new(frame: &'f mut Frame<'_>) -> Self {
        Self {
            frame,
            recording: None,
            staged: 0,
        }
    }

    /// The RHI the batch's frame belongs to, to create other objects while
    /// the batch borrows the frame.
    #[must_use]
    pub fn rhi(&self) -> &'f Rhi {
        self.frame.rhi()
    }

    /// Bytes the batch staged so far, to spread large uploads over frames.
    #[must_use]
    pub fn staged(&self) -> u64 {
        self.staged
    }

    /// Creates a device-local buffer holding `data`, which graphics work
    /// submitted after this batch reads through `access`. `usage` gains
    /// [`BufferUsage::COPY_DST`].
    ///
    /// Copies move whole four-byte words, so the buffer's size is `data`'s
    /// length rounded up to a multiple of four; the padding is unspecified.
    ///
    /// The RHI never sees heap indices `data` holds, such as a table of
    /// materials: work reading them relies on the caller to keep their
    /// resources alive until it completes, since their slots may otherwise
    /// be emptied and filled again while it runs (see
    /// [`HeapIndices`](trait@dirk_rhi::HeapIndices)). Indices in frame
    /// memory, such as per-frame records, need no such care.
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
        // Frame allocations reserve whole words, which the copy rounds to.
        let staging = self.stage(data)?;
        let recording = Recording::get(&mut self.recording, self.frame)?;
        let copy = BufferCopy {
            src_offset: staging.offset(),
            dst_offset: 0,
            size,
        };
        with_copies!(recording, encoder => {
            encoder.copy_buffer(staging.buffer(), &buffer, &[copy])
        })?;
        recording.finish_buffer(&buffer, access)?;
        Ok(buffer)
    }

    /// Writes `texels` into `region` of `image`, keeping the rest of it, in
    /// a copy that moves the region from its resting access and back.
    /// `texels` holds the region's texels tightly packed, as for
    /// [`Uploads::image`].
    ///
    /// The graphics queue copies them, after the batch's other copies.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `texels`' length differs from the
    /// region's texels, `region` is not inside `image`, or `image` lacks
    /// [`ImageUsage::COPY_DST`].
    pub fn write_image(&mut self, image: &Image, region: ImageRegion, texels: &[u8]) -> Result<()> {
        let info = image.info();
        Self::check_texels(info, image.label(), &region, texels, "the region")?;
        let staging = self.stage(texels)?;
        let copy = BufferImageCopy::packed(info.format, staging.offset(), region);
        let encoder = Recording::get(&mut self.recording, self.frame)?.graphics();
        encoder.copy_buffer_to_image(staging.buffer(), image, &[copy])
    }

    /// Creates an image from `desc` holding `mips`, in its resting access
    /// for graphics work submitted after this batch. `desc.usage` gains
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
    pub fn image(&mut self, desc: &ImageDesc<'_>, mips: &[&[u8]]) -> Result<Image> {
        self.copy_image(desc, mips)
    }

    /// Creates an image from `desc` holding `texels` in mip 0, packed as
    /// for [`Uploads::image`], and generates its other mips on the graphics
    /// queue by filtering mip 0 down, leaving it in its resting access for
    /// graphics work submitted after this batch. `desc.usage` gains
    /// [`ImageUsage::COPY_SRC`] and [`ImageUsage::COPY_DST`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Uploads::image`], or [`Error::Invalid`] if the
    /// format cannot be blitted with linear filtering.
    pub fn mipmapped_image(&mut self, desc: &ImageDesc<'_>, texels: &[u8]) -> Result<Image> {
        let desc = ImageDesc {
            usage: desc.usage | ImageUsage::COPY_SRC,
            ..*desc
        };
        // Hands mip 0 over to the graphics queue, which blits the others.
        let image = self.copy_image(&desc, &[texels])?;
        Recording::get(&mut self.recording, self.frame)?
            .graphics()
            .generate_mips(&image)?;
        Ok(image)
    }

    /// Submits the batch. Graphics work submitted afterwards sees every
    /// upload, in its requested or resting access, without waiting.
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

    /// Creates an image from `desc`, with [`ImageUsage::COPY_DST`], records
    /// copies of `mips` into it and hands it over to graphics work in its
    /// resting access.
    fn copy_image(&mut self, desc: &ImageDesc<'_>, mips: &[&[u8]]) -> Result<Image> {
        let image = self.frame.rhi().create_image(&ImageDesc {
            usage: desc.usage | ImageUsage::COPY_DST,
            ..*desc
        })?;
        let info = *image.info();
        Self::check_mips(&info, desc.label, mips)?;
        Recording::get(&mut self.recording, self.frame)?.begin_image(&image)?;
        let copied = (0..).zip(mips).try_for_each(|(level, data)| {
            // Frame allocations start at offsets valid for copies of any
            // format.
            let staging = self.stage(data)?;
            let copy =
                BufferImageCopy::packed(info.format, staging.offset(), info.mip_region(level));
            with_copies!(Recording::get(&mut self.recording, self.frame)?, encoder => {
                encoder.copy_buffer_to_image(staging.buffer(), &image, &[copy])
            })
        });
        // Hands the image over even if a copy failed, such as into a depth
        // image: the copy queue cannot return it to rest, so submitting the
        // batch's other uploads would fail.
        let finished = Recording::get(&mut self.recording, self.frame)?.finish_image(&image);
        copied.and(finished)?;
        Ok(image)
    }

    /// Copies `data` into the frame's memory, counting it as staged.
    fn stage(&mut self, data: &[u8]) -> Result<FrameSlice<'f, u8>> {
        self.staged += data.len() as u64;
        self.frame.alloc(data)
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

    /// Prepares a new image for copies into it, taking over its first
    /// transition from undefined contents.
    fn begin_image(&mut self, image: &Image) -> Result<()> {
        let barrier = [ImageBarrier {
            image,
            range: ImageRange::ALL,
            before: ImageAccess::Undefined,
            after: ImageAccess::CopyDestination,
            transfer: None,
        }];
        // SAFETY: the batch just created the image; nothing accessed it, and
        // `finish_image` returns it to rest. On the copy queue, the barrier
        // takes over its first transition, and the batch's graphics work,
        // submitted after the copies, finds it at rest.
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
    /// work, in its resting access.
    fn finish_image(&mut self, image: &Image) -> Result<()> {
        let barrier = [ImageBarrier {
            image,
            range: ImageRange::ALL,
            before: ImageAccess::CopyDestination,
            after: image.info().rest,
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dirk_rhi::{Extent3d, Format, ImageDimension, SampleCount};
    use dirk_rhi_test_shaders::support::with_rhi;

    use super::*;

    /// Records on the copy queue even without a dedicated one, which runs it
    /// on the graphics queue's family, so every device covers that path.
    #[test]
    fn failed_image_uploads_on_the_copy_queue_leave_the_batch_to_submit() -> Result<()> {
        with_rhi(|rhi| {
            let mut frame = rhi.begin_frame()?;
            let mut uploads = Uploads::new(&mut frame);
            uploads.recording = Some(Recording::Transfer {
                copies: uploads.frame.encoder("uploads")?,
                acquires: uploads.frame.encoder("upload acquisitions")?,
            });
            uploads.buffer(
                "before",
                BufferUsage::COPY_SRC,
                BufferAccess::CopySource,
                &[1; 16],
            )?;
            let depth = ImageDesc {
                label: "depth",
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(4, 4),
                format: Format::Depth32Float,
                usage: ImageUsage::SAMPLED,
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::X1,
                rest: None,
            };
            assert!(matches!(
                uploads.image(&depth, &[&[0; 64]]),
                Err(Error::Invalid { .. })
            ));
            let done = uploads.submit()?.expect("the buffer upload was recorded");
            frame.finish();
            rhi.wait(done, Duration::from_secs(10))
        })
    }
}
