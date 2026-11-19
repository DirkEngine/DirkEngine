use std::marker::PhantomData;

use crate::{
    Buffer, BufferAccess, BufferUsage, Completion, Error, Filter, Format, FormatFeatures, Frame,
    Graphics, Image, ImageAccess, ImageRange, ImageRegion, ImageUsage, QueueKind, QueueTransfer,
    QueueType, Result, SampleCount, native,
};

/// Copies `size` bytes between buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferCopy {
    /// Offset in the source buffer.
    pub src_offset: u64,
    /// Offset in the destination buffer.
    pub dst_offset: u64,
    /// Bytes to copy.
    pub size: u64,
}

/// Copies between a buffer and a region of a color image. The buffer holds
/// the region's rows back to back, `bytes_per_row` apart, and its layers or
/// depth slices `rows_per_image` rows apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferImageCopy {
    /// Offset of the first texel in the buffer; a multiple of the format's
    /// [`copy_offset_alignment`](Format::copy_offset_alignment).
    pub buffer_offset: u64,
    /// Bytes from one row to the next; a multiple of the texel size.
    pub bytes_per_row: u32,
    /// Rows from one layer or depth slice to the next.
    pub rows_per_image: u32,
    /// The texels copied.
    pub image: ImageRegion,
}

impl BufferImageCopy {
    /// A copy of `region` with rows packed tightly from `buffer_offset`.
    #[must_use]
    pub const fn packed(format: Format, buffer_offset: u64, region: ImageRegion) -> Self {
        Self {
            buffer_offset,
            bytes_per_row: region.extent.width * format.bytes_per_texel(),
            rows_per_image: region.extent.height,
            image: region,
        }
    }
}

/// Copies or blits between image regions. Copies need equal extents; blits
/// scale the source to the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageCopy {
    /// The texels read.
    pub src: ImageRegion,
    /// The texels written; as many layers as `src`.
    pub dst: ImageRegion,
}

/// Orders accesses to a whole buffer, optionally moving it between queues.
#[derive(Clone, Copy)]
pub struct BufferBarrier<'a> {
    /// The buffer.
    pub buffer: &'a Buffer,
    /// How it was accessed before the barrier.
    pub before: BufferAccess,
    /// How it is accessed after the barrier.
    pub after: BufferAccess,
    /// Queue ownership transfer, recorded on both queues.
    pub transfer: Option<QueueTransfer>,
}

/// Orders accesses to an image range and changes its layout, optionally
/// moving it between queues.
#[derive(Clone, Copy)]
pub struct ImageBarrier<'a> {
    /// The image.
    pub image: &'a Image,
    /// The mip levels and layers.
    pub range: ImageRange,
    /// How it was accessed before the barrier.
    pub before: ImageAccess,
    /// How it is accessed after the barrier.
    pub after: ImageAccess,
    /// Queue ownership transfer, recorded on both queues.
    pub transfer: Option<QueueTransfer>,
}

/// Records commands for queue `Q` within a [`Frame`], which it cannot
/// outlive. Submit it with [`Frame::submit`]; dropping it discards it.
///
/// Copy queues cannot blit:
/// ```compile_fail
/// use dirk_rhi::{CopyQueue, Filter, Frame, Image, ImageCopy, Result};
/// fn blit(frame: &Frame<'_>, image: &Image, region: ImageCopy) -> Result<()> {
///     let mut encoder = frame.encoder::<CopyQueue>("copies")?;
///     encoder.blit_image(image, image, &[region], Filter::Linear)
/// }
/// ```
///
/// Encoders cannot outlive their frame:
/// ```compile_fail
/// use dirk_rhi::{Graphics, Result, Rhi};
/// fn escape(rhi: &mut Rhi) -> Result<()> {
///     let frame = rhi.begin_frame()?;
///     let encoder = frame.encoder::<Graphics>("escaping")?;
///     frame.finish();
///     drop(encoder);
///     Ok(())
/// }
/// ```
pub struct CommandEncoder<'f, Q: QueueKind = Graphics> {
    pub(crate) frame: &'f Frame<'f>,
    pub(crate) raw: native::CommandBuffer,
    /// Timeline values on each queue the GPU waits for before running this.
    pub(crate) waits: [u64; QueueType::COUNT],
    queue: PhantomData<Q>,
}

impl<'f, Q: QueueKind> CommandEncoder<'f, Q> {
    pub(crate) fn new(frame: &'f Frame<'f>, raw: native::CommandBuffer) -> Self {
        Self {
            frame,
            raw,
            waits: [0; QueueType::COUNT],
            queue: PhantomData,
        }
    }

    /// Makes the GPU wait for `completion` before running these commands,
    /// making its writes visible to them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `completion` was not submitted on this
    /// device.
    pub fn wait_for(&mut self, completion: Completion) -> Result<()> {
        self.frame.rhi().check_submitted(completion)?;
        let wait = &mut self.waits[completion.queue as usize];
        *wait = (*wait).max(completion.value);
        Ok(())
    }

    /// Copies regions between buffers. Destinations may not overlap each
    /// other, nor, within one buffer, any source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a buffer is from another device or lacks
    /// the copy usage, or a region is empty, unaligned to four bytes, out of
    /// bounds or overlaps another in the same buffer.
    pub fn copy_buffer(
        &mut self,
        src: &Buffer,
        dst: &Buffer,
        regions: &[BufferCopy],
    ) -> Result<()> {
        let invalid = |reason: String| Err(Error::invalid("buffer copy", dst.label(), reason));
        self.check_buffer(src, BufferUsage::COPY_SRC)?;
        self.check_buffer(dst, BufferUsage::COPY_DST)?;
        if regions.is_empty() {
            return invalid("no regions".to_owned());
        }
        let in_bounds = |offset: u64, size: u64, buffer: &Buffer| {
            offset
                .checked_add(size)
                .is_some_and(|end| end <= buffer.info().size)
        };
        for region in regions {
            if region.size == 0
                || !(region.src_offset | region.dst_offset | region.size).is_multiple_of(4)
            {
                return invalid(format!("{region:?} is empty or not aligned to four bytes"));
            }
            if !in_bounds(region.src_offset, region.size, src)
                || !in_bounds(region.dst_offset, region.size, dst)
            {
                return invalid(format!("{region:?} is out of bounds"));
            }
        }
        let overlap = |start: u64, other: u64, size: u64, other_size: u64| {
            start < other + other_size && other < start + size
        };
        if let Err(reason) = Self::check_conflicts(
            regions,
            std::ptr::eq(src, dst),
            |write, other| overlap(write.dst_offset, other.dst_offset, write.size, other.size),
            |read, write| overlap(read.src_offset, write.dst_offset, read.size, write.size),
        ) {
            return invalid(reason);
        }
        self.raw
            .copy_buffer(self.device(), &src.raw, &dst.raw, regions);
        Ok(())
    }

    /// Copies buffer data into regions of a color image in the
    /// [`ImageAccess::CopyDestination`] layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a resource is from another device or
    /// lacks the copy usage, the image is multisampled or depth/stencil, or a
    /// region is out of bounds or misaligned.
    pub fn copy_buffer_to_image(
        &mut self,
        src: &Buffer,
        dst: &Image,
        regions: &[BufferImageCopy],
    ) -> Result<()> {
        self.check_buffer_image(
            src,
            BufferUsage::COPY_SRC,
            dst,
            ImageUsage::COPY_DST,
            regions,
        )?;
        self.raw.copy_buffer_to_image(
            self.device(),
            &src.raw,
            &dst.raw,
            dst.info().format,
            regions,
        );
        Ok(())
    }

    /// Copies regions of a color image in the [`ImageAccess::CopySource`]
    /// layout into a buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a resource is from another device or
    /// lacks the copy usage, the image is multisampled or depth/stencil, or a
    /// region is out of bounds or misaligned.
    pub fn copy_image_to_buffer(
        &mut self,
        src: &Image,
        dst: &Buffer,
        regions: &[BufferImageCopy],
    ) -> Result<()> {
        self.check_buffer_image(
            dst,
            BufferUsage::COPY_DST,
            src,
            ImageUsage::COPY_SRC,
            regions,
        )?;
        self.raw.copy_image_to_buffer(
            self.device(),
            &src.raw,
            &dst.raw,
            src.info().format,
            regions,
        );
        Ok(())
    }

    /// Copies regions between images of the same format and sample count,
    /// in the [`ImageAccess::CopySource`] and
    /// [`ImageAccess::CopyDestination`] layouts. Destinations may not overlap
    /// each other, nor, within one image, any source.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if an image is from another device or lacks
    /// the copy usage, the images differ in format or samples, or a region is
    /// out of bounds, overlaps another or differs in size from its source.
    pub fn copy_image(&mut self, src: &Image, dst: &Image, regions: &[ImageCopy]) -> Result<()> {
        self.check_image_copy("image copy", src, dst, regions)?;
        if let Some(region) = regions
            .iter()
            .find(|region| region.src.extent != region.dst.extent)
        {
            return Err(Error::invalid(
                "image copy",
                dst.label(),
                format!("{region:?} changes size; blit to scale"),
            ));
        }
        self.raw.copy_image(
            self.device(),
            &src.raw,
            &dst.raw,
            dst.info().format,
            regions,
        );
        Ok(())
    }

    /// Records a raw barrier, bypassing the render graph. Both halves of a
    /// queue ownership transfer record the same barrier, the destination
    /// after waiting for the source's [`Completion`].
    ///
    /// # Safety
    ///
    /// `before` must describe every access since the previous barrier on the
    /// same buffer or image range, including its current layout, or later
    /// accesses race with earlier ones.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a resource is from another device, lacks
    /// the usage an access needs, an image range is out of bounds, an image is
    /// transitioned to [`ImageAccess::Undefined`], or a transfer does not
    /// involve this encoder's queue.
    pub unsafe fn barrier(
        &mut self,
        buffers: &[BufferBarrier<'_>],
        images: &[ImageBarrier<'_>],
    ) -> Result<()> {
        let mut ranges = Vec::with_capacity(images.len());
        for barrier in buffers {
            let invalid = |reason: String| {
                Err(Error::invalid(
                    "buffer barrier",
                    barrier.buffer.label(),
                    reason,
                ))
            };
            self.check_buffer(
                barrier.buffer,
                barrier.before.usage() | barrier.after.usage(),
            )?;
            if let Err(reason) = Self::check_transfer(
                barrier.transfer,
                barrier.before.on_copy_queue(),
                barrier.after.on_copy_queue(),
            ) {
                return invalid(reason);
            }
        }
        for barrier in images {
            let invalid = |reason: String| {
                Err(Error::invalid(
                    "image barrier",
                    barrier.image.label(),
                    reason,
                ))
            };
            self.check_image(
                barrier.image,
                barrier.before.usage() | barrier.after.usage(),
            )?;
            if barrier.after == ImageAccess::Undefined {
                return invalid("cannot transition to Undefined".to_owned());
            }
            if let Err(reason) = Self::check_transfer(
                barrier.transfer,
                barrier.before.on_copy_queue(),
                barrier.after.on_copy_queue(),
            ) {
                return invalid(reason);
            }
            match barrier.image.info().resolve(barrier.range) {
                Ok(range) => ranges.push(range),
                Err(reason) => return invalid(reason),
            }
        }
        self.raw
            .barrier(self.device(), Q::TYPE, buffers, images, &ranges);
        Ok(())
    }

    /// The device commands are recorded for. Borrows the frame, not the
    /// encoder, so recording can borrow the native commands mutably.
    pub(crate) fn device(&self) -> &'f native::Device {
        &self.frame.rhi().device
    }

    /// Checks that `buffer` is from this device and has `usage`.
    pub(crate) fn check_buffer(&self, buffer: &Buffer, usage: BufferUsage) -> Result<()> {
        let invalid = |reason| Err(Error::invalid("buffer", buffer.label(), reason));
        if !buffer.raw.is_from(self.device()) {
            return invalid("belongs to another device".to_owned());
        }
        if !buffer.info().usage.contains(usage) {
            return invalid(format!("lacks the {usage:?} usage"));
        }
        Ok(())
    }

    /// Checks that `image` is from this device and has `usage`.
    pub(crate) fn check_image(&self, image: &Image, usage: ImageUsage) -> Result<()> {
        let invalid = |reason| Err(Error::invalid("image", image.label(), reason));
        if !image.raw.is_from(self.device()) {
            return invalid("belongs to another device".to_owned());
        }
        if !image.info().usage.contains(usage) {
            return invalid(format!("lacks the {usage:?} usage"));
        }
        Ok(())
    }

    /// Checks that a queue transfer moves between queues, one of them this
    /// encoder's, and that the copy queue only performs copy accesses: the
    /// access before the barrier unless it acquires, and after it unless it
    /// releases.
    fn check_transfer(
        transfer: Option<QueueTransfer>,
        before_on_copy_queue: bool,
        after_on_copy_queue: bool,
    ) -> Result<(), String> {
        match transfer {
            Some(QueueTransfer { from, to }) if from == to => {
                return Err(format!("transfers from {from:?} to itself"));
            }
            Some(QueueTransfer { from, to }) if Q::TYPE != from && Q::TYPE != to => {
                return Err(format!(
                    "transfers from {from:?} to {to:?}, not involving the {:?} queue",
                    Q::TYPE
                ));
            }
            _ => {}
        }
        let performs_before = transfer.is_none_or(|transfer| transfer.from == Q::TYPE);
        let performs_after = transfer.is_none_or(|transfer| transfer.to == Q::TYPE);
        if Q::TYPE == QueueType::Copy
            && ((performs_before && !before_on_copy_queue)
                || (performs_after && !after_on_copy_queue))
        {
            return Err(
                "the copy queue only copies; transfer the resource to the graphics queue"
                    .to_owned(),
            );
        }
        Ok(())
    }

    /// Rejects regions writing what another region writes and, when they
    /// copy within one resource, reading what any region writes.
    fn check_conflicts<T: std::fmt::Debug>(
        regions: &[T],
        same_resource: bool,
        writes_overlap: impl Fn(&T, &T) -> bool,
        read_overlaps_write: impl Fn(&T, &T) -> bool,
    ) -> Result<(), String> {
        for (index, region) in regions.iter().enumerate() {
            if let Some(other) = regions[index + 1..]
                .iter()
                .find(|other| writes_overlap(region, other))
            {
                return Err(format!("destinations of {region:?} and {other:?} overlap"));
            }
            if same_resource
                && let Some(write) = regions
                    .iter()
                    .find(|write| read_overlaps_write(region, write))
            {
                return Err(format!(
                    "source of {region:?} overlaps destination of {write:?}"
                ));
            }
        }
        Ok(())
    }

    /// Checks copies between `buffer` and the color `image`.
    fn check_buffer_image(
        &self,
        buffer: &Buffer,
        buffer_usage: BufferUsage,
        image: &Image,
        image_usage: ImageUsage,
        regions: &[BufferImageCopy],
    ) -> Result<()> {
        let invalid =
            |reason: String| Err(Error::invalid("buffer image copy", image.label(), reason));
        self.check_buffer(buffer, buffer_usage)?;
        self.check_image(image, image_usage)?;
        let info = image.info();
        if info.format.is_depth_stencil() || info.samples != SampleCount::X1 {
            return invalid(format!(
                "copies with {:?} images of {:?} samples are not supported",
                info.format, info.samples
            ));
        }
        if regions.is_empty() {
            return invalid("no regions".to_owned());
        }
        let texel = u64::from(info.format.bytes_per_texel());
        let alignment = info.format.copy_offset_alignment();
        for region in regions {
            if let Err(reason) = info.check_region(&region.image) {
                return invalid(reason);
            }
            let extent = region.image.extent;
            let row = u64::from(region.bytes_per_row);
            let rows = u64::from(region.rows_per_image);
            if row < u64::from(extent.width) * texel
                || !row.is_multiple_of(texel)
                || rows < u64::from(extent.height)
                || !region.buffer_offset.is_multiple_of(alignment)
            {
                return invalid(format!(
                    "{region:?} has rows shorter than the region, a row pitch that is not a \
                     multiple of {texel} bytes or an offset that is not a multiple of {alignment}"
                ));
            }
            // Rows before the last one, then the last row's texels.
            let slices = u64::from(extent.depth) * u64::from(region.image.layer_count);
            let end = (slices - 1)
                .checked_mul(rows)
                .and_then(|rows| rows.checked_add(u64::from(extent.height) - 1))
                .and_then(|rows| rows.checked_mul(row))
                .and_then(|size| size.checked_add(u64::from(extent.width) * texel))
                .and_then(|size| size.checked_add(region.buffer_offset));
            if end.is_none_or(|end| end > buffer.info().size) {
                return invalid(format!(
                    "{region:?} reaches beyond the buffer's {} bytes",
                    buffer.info().size
                ));
            }
        }
        Ok(())
    }

    /// Checks copies or blits between images of one format and sample count,
    /// whose regions do not overlap within one image.
    fn check_image_copy(
        &self,
        operation: &'static str,
        src: &Image,
        dst: &Image,
        regions: &[ImageCopy],
    ) -> Result<()> {
        let invalid = |reason: String| Err(Error::invalid(operation, dst.label(), reason));
        self.check_image(src, ImageUsage::COPY_SRC)?;
        self.check_image(dst, ImageUsage::COPY_DST)?;
        if src.info().format != dst.info().format || src.info().samples != dst.info().samples {
            return invalid(format!(
                "copies {:?} with {:?} samples into {:?} with {:?} samples",
                src.info().format,
                src.info().samples,
                dst.info().format,
                dst.info().samples
            ));
        }
        if Q::TYPE == QueueType::Copy && src.info().format.is_depth_stencil() {
            return invalid("depth/stencil images are copied on the graphics queue".to_owned());
        }
        if regions.is_empty() {
            return invalid("no regions".to_owned());
        }
        for region in regions {
            if let Err(reason) = src
                .info()
                .check_region(&region.src)
                .and_then(|()| dst.info().check_region(&region.dst))
            {
                return invalid(reason);
            }
            if region.src.layer_count != region.dst.layer_count {
                return invalid(format!("{region:?} changes the number of layers"));
            }
        }
        if let Err(reason) = Self::check_conflicts(
            regions,
            std::ptr::eq(src, dst),
            |write, other| write.dst.overlaps(&other.dst),
            |read, write| read.src.overlaps(&write.dst),
        ) {
            return invalid(reason);
        }
        Ok(())
    }
}

impl CommandEncoder<'_, Graphics> {
    /// Blits regions between images of the same format, scaling with
    /// `filter`, in the [`ImageAccess::CopySource`] and
    /// [`ImageAccess::CopyDestination`] layouts. Destinations may not overlap
    /// each other, nor, within one image, any source, so mips can be
    /// generated from the previous level.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if an image is from another device, lacks
    /// the copy usage or is multisampled, the format cannot be blitted or
    /// filtered with `filter`, or a region is out of bounds or overlaps
    /// another, or a native error if Metal cannot create what it draws
    /// blits with.
    pub fn blit_image(
        &mut self,
        src: &Image,
        dst: &Image,
        regions: &[ImageCopy],
        filter: Filter,
    ) -> Result<()> {
        self.check_image_copy("image blit", src, dst, regions)?;
        self.check_blit("image blit", src, filter)?;
        self.raw.blit_image(
            self.device(),
            &src.raw,
            &dst.raw,
            src.info().format,
            regions,
            filter,
        )
    }

    /// Generates every mip of every layer of `image` by filtering mip 0
    /// down, moving the whole image from `before` to `after` with the
    /// barriers it needs. Mips past the first may hold anything before.
    ///
    /// `before` must describe every access since the image's previous
    /// barrier, including its layout. Like render graph imports, which
    /// state the access an image is in rather than recording one, it is
    /// trusted: the typestate or uploads producing the image state it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `image` is from another device, lacks
    /// the copy usages or those `before` and `after` need, is multisampled,
    /// its format cannot be blitted with linear filtering, `before`
    /// discards mip 0 or `after` is [`ImageAccess::Undefined`].
    pub fn generate_mips(
        &mut self,
        image: &Image,
        before: ImageAccess,
        after: ImageAccess,
    ) -> Result<()> {
        let copies = ImageUsage::COPY_SRC | ImageUsage::COPY_DST;
        self.check_image(image, copies | before.usage() | after.usage())?;
        self.check_blit("mip generation", image, Filter::Linear)?;
        if matches!(before, ImageAccess::Undefined | ImageAccess::Present)
            || after == ImageAccess::Undefined
        {
            return Err(Error::invalid(
                "mip generation",
                image.label(),
                format!("cannot generate mips from {before:?} to {after:?}"),
            ));
        }
        self.raw.generate_mips(self.device(), image, before, after)
    }

    /// Checks that `image` is single-sampled and its format can be blitted
    /// with `filter`.
    fn check_blit(&self, operation: &'static str, image: &Image, filter: Filter) -> Result<()> {
        let info = image.info();
        let features = self.frame.rhi().capabilities().format(info.format).features;
        let filterable = filter == Filter::Nearest
            || (features.contains(FormatFeatures::FILTER_LINEAR)
                && !info.format.is_depth_stencil());
        if !features.contains(FormatFeatures::BLIT)
            || !filterable
            || info.samples != SampleCount::X1
        {
            return Err(Error::invalid(
                operation,
                image.label(),
                format!(
                    "{:?} with {:?} samples cannot be blitted with {filter:?} filtering",
                    info.format, info.samples
                ),
            ));
        }
        Ok(())
    }
}

impl ImageRegion {
    /// Whether the two regions share a texel.
    fn overlaps(&self, other: &Self) -> bool {
        let overlap = |start: u32, length: u32, other_start: u32, other_length: u32| {
            u64::from(start) < u64::from(other_start) + u64::from(other_length)
                && u64::from(other_start) < u64::from(start) + u64::from(length)
        };
        self.mip_level == other.mip_level
            && overlap(
                self.base_layer,
                self.layer_count,
                other.base_layer,
                other.layer_count,
            )
            && overlap(
                self.origin.x,
                self.extent.width,
                other.origin.x,
                other.extent.width,
            )
            && overlap(
                self.origin.y,
                self.extent.height,
                other.origin.y,
                other.extent.height,
            )
            && overlap(
                self.origin.z,
                self.extent.depth,
                other.origin.z,
                other.extent.depth,
            )
    }
}
