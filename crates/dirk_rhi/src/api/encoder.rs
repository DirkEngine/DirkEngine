use super::{
    Arc, Backend, Completion, GpuBindGroup, GpuBuffer, GpuGraphicsPipeline, GpuImage, GpuImageView,
    GpuPipelineLayout, GpuSurfaceFrame, Mutex, Object, Ordering, Payload, Rhi, Work,
};
use crate::{
    BufferUsages, ImageSubresourceRange, InvalidResourceKind as Ir, NativeCommandBuffer as _,
    QueueType, ResourceAccess as Access, Result,
};
use std::marker::PhantomData;
mod sealed {
    pub trait Sealed {}
}
/// Supported semantic recording queue. Native queues may alias on hardware without separate families.
pub trait QueueKind: sealed::Sealed + Send + Sync + 'static {
    /// Semantic queue role.
    const KIND: QueueType;
    #[doc(hidden)]
    fn queue<B: Backend>(rhi: &mut Rhi<B>) -> &mut Queue<B, Self>
    where
        Self: Sized;
}
/// Graphics recording and submission capability.
pub struct Graphics;
/// Transfer recording and submission capability.
pub struct CopyQueue;
impl sealed::Sealed for Graphics {}
impl sealed::Sealed for CopyQueue {}
impl QueueKind for Graphics {
    const KIND: QueueType = QueueType::Graphics;
    fn queue<B: Backend>(rhi: &mut Rhi<B>) -> &mut Queue<B, Self> {
        &mut rhi.graphics
    }
}
impl QueueKind for CopyQueue {
    const KIND: QueueType = QueueType::Copy;
    fn queue<B: Backend>(rhi: &mut Rhi<B>) -> &mut Queue<B, Self> {
        &mut rhi.transfer
    }
}
/// A typed queue belonging to one device.
pub struct Queue<B: Backend, Q: QueueKind = Graphics> {
    device: Arc<super::device::Device<B>>,
    kind: PhantomData<Q>,
    timeline: B::TimelineSemaphore,
    next_value: u64,
}
/// Explicit GPU dependencies for a submitted batch.
pub struct SubmitInfo<'a, B: Backend> {
    /// Frames touched by this batch; each frame can be associated exactly once.
    pub surface_frames: &'a [&'a GpuSurfaceFrame<B>],
    /// Completion points from already submitted producers, waited on by the GPU.
    pub wait_for: &'a [&'a Completion<B>],
}
impl<B: Backend> Default for SubmitInfo<'_, B> {
    fn default() -> Self {
        Self {
            surface_frames: &[],
            wait_for: &[],
        }
    }
}
/// Exclusive recording scope. Graphics methods are available only through a render pass.
///
/// A copy queue cannot begin graphics rendering:
/// ```compile_fail
/// use dirk_rhi::{Rhi, CopyQueue, RenderingInfo, Result};
/// unsafe fn invalid(device: &Rhi, info: &RenderingInfo<'_>) -> Result<()> {
///     let mut encoder = device.create_encoder::<CopyQueue>("copy")?;
///     let _pass = unsafe { encoder.begin_render_pass(info)? };
///     Ok(())
/// }
/// ```
/// An active pass exclusively borrows its encoder:
/// ```compile_fail
/// use dirk_rhi::{Rhi, Graphics, RenderingInfo, Result};
/// unsafe fn invalid(device: &Rhi, info: &RenderingInfo<'_>) -> Result<()> {
///     let mut encoder = device.create_encoder::<Graphics>("graphics")?;
///     let pass = unsafe { encoder.begin_render_pass(info)? };
///     let _commands = encoder.finish()?;
///     drop(pass);
///     Ok(())
/// }
/// ```
pub struct CommandEncoder<B: Backend, Q: QueueKind = Graphics> {
    cycle: u64,
    device: Arc<super::device::Device<B>>,
    raw: B::CommandBuffer,
    pool: B::CommandPool,
    in_pass: bool,
    failed: bool,
    kind: PhantomData<Q>,
}
/// Finished, single-submit commands. Consumed by their compatible queue.
pub struct RecordedCommands<B: Backend, Q: QueueKind = Graphics> {
    cycle: u64,
    device: Arc<super::device::Device<B>>,
    raw: B::CommandBuffer,
    pool: B::CommandPool,
    kind: PhantomData<Q>,
}
impl<B: Backend> Rhi<B> {
    /// Borrows this device's single queue object for the selected capability.
    pub fn queue<Q: QueueKind>(&mut self) -> &mut Queue<B, Q> {
        Q::queue(self)
    }
    /// Begins a new exclusive recording. Native pools are owned by the recording through completion.
    pub fn create_encoder<Q: QueueKind>(&self, label: &str) -> Result<CommandEncoder<B, Q>> {
        let _gate = self.device.gate.lock();
        let cached = self.device.pools.lock().entry(Q::KIND).or_default().pop();
        let (mut raw, pool) = if let Some(pair) = cached {
            pair
        } else {
            let mut pool = unsafe { self.device.backend.create_command_pool(Q::KIND)? };
            let raw = unsafe { self.device.backend.create_command_buffer(&mut pool)? };
            (raw, pool)
        };
        unsafe {
            raw.begin(label, true)?;
        }
        Ok(CommandEncoder {
            cycle: self.device.state.lock().cycle,
            device: self.device.clone(),
            raw,
            pool,
            in_pass: false,
            failed: false,
            kind: PhantomData,
        })
    }
}
impl<B: Backend, Q: QueueKind> CommandEncoder<B, Q> {
    fn outside(&self) -> Result<()> {
        if self.in_pass || self.failed {
            return Err(Ir::BadState.into());
        }
        Ok(())
    }
    fn record(&mut self, result: Result<()>) -> Result<()> {
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn buffer(
        buffer: &GpuBuffer<B>,
        usage: BufferUsages,
        range: crate::BufferRange,
    ) -> Result<crate::BufferRange> {
        if !buffer.info().usage.contains(usage) {
            return Err(Ir::Mismatch.into());
        }
        let range = range.resolve(buffer.size())?;

        Ok(range)
    }

    // Checks creation capabilities; dependencies are recorded only by explicit barriers.
    fn image_access(
        image: &GpuImage<B>,
        access: Access,
        range: ImageSubresourceRange,
    ) -> Result<()> {
        range.resolve(&image.description())?;
        if !image.description().usage.contains(access.image_usage()) {
            return Err(Ir::Mismatch.into());
        }
        Ok(())
    }

    /// Applies graph-generated semantic dependencies, checking all ranges and deriving native states.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn barrier(&mut self, info: &crate::DependencyInfo<'_, Rhi<B>>) -> Result<()> {
        self.outside()?;
        let images = info
            .image_barriers
            .iter()
            .map(|b| {
                let r = ImageSubresourceRange {
                    aspects: b.aspects,
                    base_mip_level: b.base_mip_level,
                    mip_level_count: b.mip_level_count,
                    base_array_layer: b.base_array_layer,
                    array_layer_count: b.array_layer_count,
                }
                .resolve(&b.image.description())?;
                Ok(crate::ImageBarrier::<B> {
                    image: b.image.raw(),
                    old_state: b.old_state,
                    new_state: b.new_state,
                    aspects: r.aspects,
                    base_mip_level: r.base_mip_level,
                    mip_level_count: r.mip_level_count,
                    base_array_layer: r.base_array_layer,
                    array_layer_count: r.array_layer_count,
                    queue_transfer: b.queue_transfer,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let buffers = info
            .buffer_barriers
            .iter()
            .map(|b| {
                let range = crate::BufferRange {
                    offset: b.offset,
                    size: b.size,
                }
                .resolve(b.buffer.size())?;
                Ok(crate::BufferBarrier::<B> {
                    buffer: b.buffer.raw(),
                    old_state: b.old_state,
                    new_state: b.new_state,
                    offset: range.offset,
                    size: range.size,
                    queue_transfer: b.queue_transfer,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = unsafe {
            self.raw.barrier(&crate::DependencyInfo {
                memory_barriers: info.memory_barriers,
                buffer_barriers: &buffers,
                image_barriers: &images,
            })
        };
        self.record(result)
    }

    /// Copies checked, non-overlapping byte ranges.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn copy_buffer(
        &mut self,
        src: &GpuBuffer<B>,
        dst: &GpuBuffer<B>,
        regions: &[crate::BufferCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            Self::buffer(
                src,
                BufferUsages::COPY_SRC,
                crate::BufferRange {
                    offset: region.src_offset,
                    size: region.size,
                },
            )?;
            Self::buffer(
                dst,
                BufferUsages::COPY_DST,
                crate::BufferRange {
                    offset: region.dst_offset,
                    size: region.size,
                },
            )?;
            if !region.src_offset.is_multiple_of(4)
                || !region.dst_offset.is_multiple_of(4)
                || !region.size.is_multiple_of(4)
                || std::ptr::eq(src, dst)
            {
                return Err(Ir::Mismatch.into());
            }
        }
        let result = unsafe { self.raw.copy_buffer(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    fn image_region(
        image: &GpuImage<B>,
        mip: u32,
        layer: u32,
        layers: u32,
        origin: crate::Origin3d,
        extent: crate::Extent3d,
        aspects: crate::ImageAspects,
    ) -> Result<ImageSubresourceRange> {
        let info = image.description();
        let limit = info.mip_extent(mip)?;
        let range = ImageSubresourceRange {
            aspects,
            base_mip_level: mip,
            mip_level_count: 1,
            base_array_layer: layer,
            array_layer_count: layers,
        }
        .resolve(&info)?;
        if extent.width == 0
            || extent.height == 0
            || extent.depth == 0
            || origin
                .x
                .checked_add(extent.width)
                .is_none_or(|n| n > limit.width)
            || origin
                .y
                .checked_add(extent.height)
                .is_none_or(|n| n > limit.height)
            || origin
                .z
                .checked_add(extent.depth)
                .is_none_or(|n| n > limit.depth)
        {
            return Err(Ir::OutOfRange.into());
        }
        Ok(range)
    }
    fn buffer_image(
        &mut self,
        buffer: &GpuBuffer<B>,
        image: &GpuImage<B>,
        region: &crate::BufferImageCopy,
        to_image: bool,
    ) -> Result<()> {
        let info = image.description();
        if info.samples != crate::SampleCount::One
            || !region.aspects.contains(crate::ImageAspects::COLOR)
        {
            return Err(crate::UnsupportedOperation::Capability(
                "multisample or depth/stencil buffer-image copy",
            )
            .into());
        }
        let range = Self::image_region(
            image,
            region.mip_level,
            region.base_array_layer,
            region.array_layer_count,
            region.image_origin,
            region.extent,
            region.aspects,
        )?;
        let caps = self.device.backend.capabilities();
        let row = u64::from(region.buffer_bytes_per_row.get());
        let rows = u64::from(region.buffer_rows_per_image.get());
        let row_bytes = u64::from(region.extent.width) * u64::from(info.format.texel_size());
        if row < row_bytes
            || rows < u64::from(region.extent.height)
            || !row.is_multiple_of(u64::from(caps.buffer_copy_row_pitch_alignment.max(1)))
            || !region
                .buffer_offset
                .is_multiple_of(caps.buffer_copy_offset_alignment.max(1))
            || !row.is_multiple_of(u64::from(info.format.texel_size()))
        {
            return Err(Ir::Mismatch.into());
        }
        let slices = u64::from(region.extent.depth) * u64::from(region.array_layer_count);
        let size = (slices - 1)
            .checked_mul(rows)
            .and_then(|n| n.checked_add(u64::from(region.extent.height) - 1))
            .and_then(|n| n.checked_mul(row))
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or(Ir::OutOfRange)?;
        Self::buffer(
            buffer,
            if to_image {
                BufferUsages::COPY_SRC
            } else {
                BufferUsages::COPY_DST
            },
            crate::BufferRange {
                offset: region.buffer_offset,
                size,
            },
        )?;
        Self::image_access(
            image,
            if to_image {
                Access::CopyDestination
            } else {
                Access::CopySource
            },
            range,
        )
    }
    /// Uploads checked image regions using an aligned staging layout.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        src: &GpuBuffer<B>,
        dst: &GpuImage<B>,
        regions: &[crate::BufferImageCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            self.buffer_image(src, dst, region, true)?;
        }
        let result = unsafe { self.raw.copy_buffer_to_image(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Copies image regions to a readback buffer.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn copy_image_to_buffer(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuBuffer<B>,
        regions: &[crate::BufferImageCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            self.buffer_image(dst, src, region, false)?;
        }
        let result = unsafe { self.raw.copy_image_to_buffer(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Copies matching formats without filtering.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn copy_image(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuImage<B>,
        regions: &[crate::ImageCopy],
    ) -> Result<()> {
        self.outside()?;
        if std::ptr::eq(src, dst)
            || src.description().format != dst.description().format
            || src.description().samples != dst.description().samples
        {
            return Err(Ir::Mismatch.into());
        }
        for r in regions {
            let a = Self::image_region(
                src,
                r.src_mip_level,
                r.src_base_array_layer,
                r.array_layer_count,
                r.src_origin,
                r.extent,
                r.aspects,
            )?;
            let b = Self::image_region(
                dst,
                r.dst_mip_level,
                r.dst_base_array_layer,
                r.array_layer_count,
                r.dst_origin,
                r.extent,
                r.aspects,
            )?;
            Self::image_access(src, Access::CopySource, a)?;
            Self::image_access(dst, Access::CopyDestination, b)?;
        }
        let result = unsafe { self.raw.copy_image(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Blits exact regions only when the backend advertises support.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn blit_image(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuImage<B>,
        regions: &[crate::ImageBlit],
        filter: crate::FilterMode,
    ) -> Result<()> {
        self.outside()?;
        let support = self
            .device
            .backend
            .format_capabilities(src.description().format);
        if !support.blit.supports(filter) {
            return Err(crate::UnsupportedOperation::ImageBlit.into());
        }
        if src.description().format != dst.description().format
            || src.description().samples != crate::SampleCount::One
            || dst.description().samples != crate::SampleCount::One
        {
            return Err(Ir::Mismatch.into());
        }
        for r in regions {
            if std::ptr::eq(src, dst) && r.src_mip_level == r.dst_mip_level {
                return Err(Ir::Mismatch.into());
            }
            let a = Self::image_region(
                src,
                r.src_mip_level,
                r.src_base_array_layer,
                r.array_layer_count,
                r.src_origin,
                r.src_extent,
                r.aspects,
            )?;
            let b = Self::image_region(
                dst,
                r.dst_mip_level,
                r.dst_base_array_layer,
                r.array_layer_count,
                r.dst_origin,
                r.dst_extent,
                r.aspects,
            )?;
            Self::image_access(src, Access::CopySource, a)?;
            Self::image_access(dst, Access::CopyDestination, b)?;
        }
        let result = unsafe { self.raw.blit_image(src.raw(), dst.raw(), regions, filter) };
        self.record(result)
    }
    /// Finishes recording. A forgotten render-pass guard is rejected at this boundary.
    pub fn finish(mut self) -> Result<RecordedCommands<B, Q>> {
        self.outside()?;
        unsafe {
            self.raw.end()?;
        }
        Ok(RecordedCommands {
            cycle: self.cycle,
            device: self.device,
            raw: self.raw,
            pool: self.pool,
            kind: PhantomData,
        })
    }
}

/// Borrowed graphics recording scope. Dropping it ends the pass; failures poison its encoder.
pub struct RenderPass<'a, B: Backend> {
    encoder: &'a mut CommandEncoder<B, Graphics>,
    colors: Vec<TextureFormat>,
    depth: Option<TextureFormat>,
    samples: crate::SampleCount,
    pipeline: Option<BoundPipeline>,
    index: Option<(u64, u64, crate::IndexFormat)>,
    viewport: bool,
    scissor: bool,
}
use crate::TextureFormat;
struct BoundPipeline {
    primitive_restart: Option<crate::IndexFormat>,
}
impl<B: Backend> CommandEncoder<B, Graphics> {
    /// Begins a graphics scope borrowing this encoder exclusively.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn begin_render_pass(
        &mut self,
        info: &crate::RenderingInfo<'_, Rhi<B>>,
    ) -> Result<RenderPass<'_, B>> {
        self.outside()?;
        if info.width == 0
            || info.height == 0
            || info.layer_count == 0
            || info.color_attachments.len()
                > self
                    .device
                    .backend
                    .capabilities()
                    .limits
                    .max_color_attachments as usize
            || (info.color_attachments.is_empty() && info.depth_attachment.is_none())
        {
            return Err(Ir::OutOfRange.into());
        }
        let mut samples = None;
        let mut views = Vec::new();
        for color in info.color_attachments {
            Self::attachment(color.view, info, &mut samples)?;
            if let Some(resolve) = color.resolve {
                let mut resolve_samples = None;
                Self::attachment(resolve, info, &mut resolve_samples)?;
                if resolve_samples != Some(crate::SampleCount::One)
                    || samples == Some(crate::SampleCount::One)
                    || resolve.info().image.format != color.view.info().image.format
                {
                    return Err(Ir::Mismatch.into());
                }
            }
            views.push(crate::ColorAttachment::<B> {
                view: color.view.raw(),
                resolve: color.resolve.map(Object::raw),
                load: color.load,
                store: color.store,
            });
        }
        let depth = if let Some(depth) = &info.depth_attachment {
            Self::attachment(depth.view, info, &mut samples)?;
            Some(crate::DepthAttachment::<B> {
                view: depth.view.raw(),
                depth_load: depth.depth_load,
                depth_store: depth.depth_store,
                stencil_load: depth.stencil_load,
                stencil_store: depth.stencil_store,
            })
        } else {
            None
        };
        let result = unsafe {
            self.raw.begin_rendering(&crate::RenderingInfo {
                label: info.label,
                width: info.width,
                height: info.height,
                layer_count: info.layer_count,
                color_attachments: &views,
                depth_attachment: depth,
            })
        };
        self.record(result)?;
        self.in_pass = true;
        Ok(RenderPass {
            encoder: self,
            colors: info
                .color_attachments
                .iter()
                .map(|a| a.view.info().image.format)
                .collect(),
            depth: info
                .depth_attachment
                .as_ref()
                .map(|a| a.view.info().image.format),
            samples: samples.unwrap_or_default(),
            pipeline: None,
            index: None,
            viewport: false,
            scissor: false,
        })
    }
    fn attachment(
        view: &GpuImageView<B>,
        info: &crate::RenderingInfo<'_, Rhi<B>>,
        samples: &mut Option<crate::SampleCount>,
    ) -> Result<()> {
        let desc = view.info().image;
        let extent = desc.mip_extent(view.info().range.base_mip_level)?;
        if view.info().range.mip_level_count != 1
            || info.width > extent.width
            || info.height > extent.height
            || info.layer_count > view.info().range.array_layer_count
        {
            return Err(Ir::Mismatch.into());
        }
        if samples.is_some_and(|s| s != desc.samples) {
            return Err(Ir::Mismatch.into());
        }
        *samples = Some(desc.samples);
        Ok(())
    }
}
impl<B: Backend> RenderPass<'_, B> {
    /// Binds a pipeline compatible with this pass's attachments.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn bind_graphics_pipeline(
        &mut self,
        pipeline: &GpuGraphicsPipeline<B>,
    ) -> Result<()> {
        if pipeline
            .info()
            .colors
            .iter()
            .map(|c| c.format)
            .ne(self.colors.iter().copied())
            || pipeline.info().depth.map(|d| d.format) != self.depth
            || pipeline.info().samples != self.samples
        {
            return Err(Ir::Mismatch.into());
        }

        let result = unsafe { self.encoder.raw.bind_graphics_pipeline(pipeline.raw()) };
        self.encoder.record(result)?;
        self.pipeline = Some(BoundPipeline {
            primitive_restart: pipeline.info().primitive_restart,
        });
        Ok(())
    }
    /// Binds immutable groups whose resource lifetimes are managed by the caller.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn bind_groups(
        &mut self,
        layout: &GpuPipelineLayout<B>,
        first_group: u32,
        groups: &[&GpuBindGroup<B>],
        dynamic_offsets: &[u64],
    ) -> Result<()> {
        let capabilities = self.encoder.device.backend.capabilities();
        let mut offsets = dynamic_offsets.iter();
        for (index, group) in groups.iter().enumerate() {
            let slot = first_group
                .checked_add(u32::try_from(index).map_err(|_| Ir::OutOfRange)?)
                .ok_or(Ir::OutOfRange)?;
            if layout.info().get(slot as usize) != Some(&group.info().layout) {
                return Err(Ir::Mismatch.into());
            }
            group
                .info()
                .validate_dynamic_offsets(slot, &mut offsets, capabilities)?;
        }
        if offsets.next().is_some() {
            return Err(Ir::Mismatch
                .with_detail("too many dynamic buffer offsets for bound groups")
                .into());
        }
        let native: Vec<_> = groups.iter().map(|g| g.raw()).collect();
        let result = unsafe {
            self.encoder
                .raw
                .bind_groups(layout.raw(), first_group, &native, dynamic_offsets)
        };
        self.encoder.record(result)
    }

    /// Binds a vertex buffer at a checked offset.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn bind_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &GpuBuffer<B>,
        offset: u64,
    ) -> Result<()> {
        if slot
            >= self
                .encoder
                .device
                .backend
                .capabilities()
                .limits
                .max_vertex_buffers
        {
            return Err(Ir::OutOfRange.into());
        }
        CommandEncoder::<B>::buffer(
            buffer,
            BufferUsages::VERTEX,
            crate::BufferRange {
                offset,
                size: u64::MAX,
            },
        )?;
        let result = unsafe {
            self.encoder
                .raw
                .bind_vertex_buffer(slot, buffer.raw(), offset)
        };
        self.encoder.record(result)?;
        Ok(())
    }
    /// Binds an index buffer aligned to its element size.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn bind_index_buffer(
        &mut self,
        buffer: &GpuBuffer<B>,
        offset: u64,
        format: crate::IndexFormat,
    ) -> Result<()> {
        let size = if format == crate::IndexFormat::Uint16 {
            2
        } else {
            4
        };
        if !offset.is_multiple_of(size) {
            return Err(Ir::Mismatch.into());
        }
        CommandEncoder::<B>::buffer(
            buffer,
            BufferUsages::INDEX,
            crate::BufferRange {
                offset,
                size: u64::MAX,
            },
        )?;
        let result = unsafe {
            self.encoder
                .raw
                .bind_index_buffer(buffer.raw(), offset, format)
        };
        self.encoder.record(result)?;
        self.index = Some((buffer.size(), offset, format));
        Ok(())
    }
    /// Sets a finite, positive viewport.
    pub fn set_viewport(&mut self, value: crate::Viewport) -> Result<()> {
        if ![
            value.x,
            value.y,
            value.width,
            value.height,
            value.min_depth,
            value.max_depth,
        ]
        .iter()
        .all(|n| n.is_finite())
            || value.width <= 0.0
            || value.height <= 0.0
            || value.min_depth < 0.0
            || value.max_depth > 1.0
            || value.min_depth > value.max_depth
        {
            return Err(Ir::OutOfRange.into());
        }
        let result = unsafe { self.encoder.raw.set_viewport(value) };
        self.encoder.record(result)?;
        self.viewport = true;
        Ok(())
    }
    /// Sets a nonempty scissor with representable coordinates.
    pub fn set_scissor(&mut self, value: crate::Rect) -> Result<()> {
        if value.width == 0
            || value.height == 0
            || value.x < 0
            || value.y < 0
            || i64::from(value.x) + i64::from(value.width) > i64::from(i32::MAX)
            || i64::from(value.y) + i64::from(value.height) > i64::from(i32::MAX)
        {
            return Err(Ir::OutOfRange.into());
        }
        let result = unsafe { self.encoder.raw.set_scissor(value) };
        self.encoder.record(result)?;
        self.scissor = true;
        Ok(())
    }
    /// Sets constant blend factors.
    pub fn set_blend_constants(&mut self, value: crate::Color) -> Result<()> {
        let result = unsafe { self.encoder.raw.set_blend_constants(value) };
        self.encoder.record(result)
    }
    /// Sets stencil comparison references.
    pub fn set_stencil_reference(&mut self, front: u32, back: u32) -> Result<()> {
        let result = unsafe { self.encoder.raw.set_stencil_reference(front, back) };
        self.encoder.record(result)
    }
    fn ready(&self) -> Result<()> {
        if self.pipeline.is_none() || !self.viewport || !self.scissor || self.encoder.failed {
            return Err(Ir::BadState.into());
        }
        Ok(())
    }

    /// Records a non-indexed draw after checking pipeline and dynamic state.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<()> {
        self.ready()?;
        let result = unsafe {
            self.encoder
                .raw
                .draw(vertex_count, instance_count, first_vertex, first_instance)
        };
        self.encoder.record(result)
    }
    /// Records an indexed draw after checking the index range.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<()> {
        self.ready()?;
        let (buffer, offset, format) = self.index.as_ref().ok_or(Ir::BadState)?;
        if self
            .pipeline
            .as_ref()
            .ok_or(Ir::BadState)?
            .primitive_restart
            .is_some_and(|required| required != *format)
        {
            return Err(Ir::Mismatch.into());
        }
        let bytes = if *format == crate::IndexFormat::Uint16 {
            2
        } else {
            4
        };
        if u64::from(first_index)
            .checked_add(u64::from(index_count))
            .and_then(|n| n.checked_mul(bytes))
            .and_then(|n| n.checked_add(*offset))
            .is_none_or(|n| n > *buffer)
        {
            return Err(Ir::OutOfRange.into());
        }
        let result = unsafe {
            self.encoder.raw.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
        self.encoder.record(result)
    }
}
impl<B: Backend> Drop for RenderPass<'_, B> {
    fn drop(&mut self) {
        let result = unsafe { self.encoder.raw.end_rendering() };
        if result.is_err() {
            self.encoder.failed = true;
        }
        self.encoder.in_pass = false;
    }
}

impl<B: Backend, Q: QueueKind> Queue<B, Q> {
    pub(super) fn new(device: Arc<super::device::Device<B>>) -> Result<Self> {
        let timeline = unsafe { device.backend.create_timeline_semaphore(0)? };
        Ok(Self {
            device,
            timeline,
            next_value: 1,
            kind: PhantomData,
        })
    }

    fn validate_submission(
        &self,
        commands: &[RecordedCommands<B, Q>],
        info: &SubmitInfo<'_, B>,
        state: &mut super::device::DeviceState<B>,
    ) -> Result<()> {
        if !info.surface_frames.is_empty() && Q::KIND != QueueType::Graphics {
            return Err(Ir::Mismatch.into());
        }
        let device = &self.device;
        let mut frame_ids = std::collections::HashSet::new();
        for frame in info.surface_frames {
            frame.require_device(&device.backend)?;
            if !frame_ids.insert(std::ptr::from_ref(*frame)) {
                return Err(Ir::BadState.into());
            }
        }
        for command in commands {
            if command.cycle != state.cycle {
                return Err(Ir::BadState
                    .with_detail("commands must be submitted in their recording cycle")
                    .into());
            }
            if !Arc::ptr_eq(&command.device, device) {
                return Err(Ir::ForeignInstance.into());
            }
        }
        Ok(())
    }
    /// Submits finished recordings once. Failure poisons the device if native submission may have started.
    /// Barriers and queue dependencies are supplied explicitly by the caller.
    ///
    /// # Safety
    /// Resources and non-owning bindings must remain valid through their last recording
    /// and GPU use. Supply correct access states and dependencies, and keep shader and
    /// vertex accesses within resource bounds. Submit once within the recording cycle.
    pub unsafe fn submit(
        &mut self,
        commands: Vec<RecordedCommands<B, Q>>,
        info: &SubmitInfo<'_, B>,
    ) -> Result<Completion<B>> {
        let device = &self.device;
        let _gate = device.gate.lock();
        let mut state = device.state.lock();
        if state.lost {
            return Err(crate::Error::DeviceLost);
        }
        self.validate_submission(&commands, info, &mut state)?;
        let native = commands.into_iter().map(|c| (c.raw, c.pool)).collect();
        let fence = unsafe { device.backend.create_fence(false)? };
        let work = Arc::new(Work {
            _backend: device.backend.clone(),
            timeline: self.timeline.clone(),
            value: self.next_value,
            pools: device.pools.clone(),
            queue: Q::KIND,
            fence,
            payload: Mutex::new(Some(Payload { commands: native })),
        });
        let payload = work.payload.lock();
        let batch = payload.as_ref().ok_or(Ir::BadState)?;
        let native_commands: Vec<_> = batch.commands.iter().map(|(command, _)| command).collect();
        let mut frames = info
            .surface_frames
            .iter()
            .map(|f| f.native.lock())
            .collect::<Vec<_>>();
        let native_frames = frames
            .iter()
            .map(|f| f.as_ref().ok_or_else(|| crate::Error::from(Ir::BadState)))
            .collect::<Result<Vec<_>>>()?;
        let waits: Vec<_> = info
            .wait_for
            .iter()
            .map(|completion| crate::TimelinePoint::<B> {
                semaphore: &completion.0.timeline,
                value: completion.0.value,
                stages: crate::PipelineStages::ALL,
            })
            .collect();
        let signals = [crate::TimelinePoint::<B> {
            semaphore: &self.timeline,
            value: self.next_value,
            stages: crate::PipelineStages::ALL,
        }];
        let result = unsafe {
            device.backend.submit(
                Q::KIND,
                &crate::Submission {
                    command_buffers: &native_commands,
                    surface_frames: &native_frames,
                    wait_timelines: &waits,
                    signal_timelines: &signals,
                    fence: Some(&work.fence),
                },
            )
        };
        drop(payload);
        if let Err(error) = result {
            state.lost = true;
            for (frame, native) in info.surface_frames.iter().zip(&mut frames) {
                frame.poison(native);
            }
            // Native failure may follow partial submission; uncertain native
            // ownership is deliberately leaked instead of being destroyed in use.
            std::mem::forget(work);
            return Err(error);
        }
        for frame in info.surface_frames {
            frame.phase.store(1, Ordering::Release);
        }
        self.next_value += 1;
        state.pending.push(work.clone());
        Ok(Completion(work))
    }
}
