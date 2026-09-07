use super::{
    Arc, Backend, Completion, GpuBindGroup, GpuBuffer, GpuGraphicsPipeline, GpuImage, GpuImageView,
    GpuPipelineLayout, GpuSurfaceFrame, Mutex, Object, Ordering, OwnedBinding, Payload, Retained,
    Rhi, Work, lock,
};
use crate::{
    BufferUsages, CommandBuffer as _, ImageSubresourceRange, InvalidResourceKind as Ir, QueueType,
    ResourceAccess as Access, Result,
};
use std::{collections::HashMap, marker::PhantomData};
mod sealed {
    pub trait Sealed {}
}
/// Supported semantic recording queue. Native scheduling initially aliases these to graphics.
pub trait QueueKind: sealed::Sealed + Send + Sync + 'static {
    /// Semantic queue role.
    const KIND: QueueType;
}
macro_rules! queues {
    ($($name:ident => $kind:ident),* $(,)?) => {$ (
        #[doc = concat!(stringify!($name), " command capability marker.")]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl QueueKind for $name { const KIND: QueueType = QueueType::$kind; }
    )*};
}
queues!(Graphics => Graphics, Compute => Compute, CopyQueue => Copy);
/// A typed queue belonging to one device.
pub struct Queue<B: Backend, Q: QueueKind = Graphics> {
    device: Rhi<B>,
    kind: PhantomData<Q>,
}
/// Explicit GPU dependencies for a submitted batch.
pub struct SubmitInfo<'a, B: Backend> {
    /// Frames touched by this batch; each frame can be associated exactly once.
    pub surface_frames: &'a [&'a GpuSurfaceFrame<B>],
    /// Previously scheduled timeline values this batch needs.
    pub wait_timelines: &'a [crate::TimelinePoint<'a, Rhi<B>>],
    /// Monotonic values published after the entire batch.
    pub signal_timelines: &'a [crate::TimelinePoint<'a, Rhi<B>>],
}
impl<B: Backend> Default for SubmitInfo<'_, B> {
    fn default() -> Self {
        Self {
            surface_frames: &[],
            wait_timelines: &[],
            signal_timelines: &[],
        }
    }
}
/// Exclusive recording scope. Graphics methods are available only through a render pass.
///
/// A copy queue cannot begin graphics rendering:
/// ```compile_fail
/// use dirk_rhi::{Backend, Rhi, CopyQueue, RenderingInfo, Result};
/// fn invalid<B: Backend>(device: &Rhi<B>, info: &RenderingInfo<'_, Rhi<B>>) -> Result<()> {
///     let mut encoder = device.create_encoder::<CopyQueue>("copy")?;
///     let _pass = encoder.begin_render_pass(info)?;
///     Ok(())
/// }
/// ```
/// An active pass exclusively borrows its encoder:
/// ```compile_fail
/// use dirk_rhi::{Backend, Rhi, Graphics, RenderingInfo, Result};
/// fn invalid<B: Backend>(device: &Rhi<B>, info: &RenderingInfo<'_, Rhi<B>>) -> Result<()> {
///     let mut encoder = device.create_encoder::<Graphics>("graphics")?;
///     let pass = encoder.begin_render_pass(info)?;
///     let _commands = encoder.finish()?;
///     drop(pass);
///     Ok(())
/// }
/// ```
pub struct CommandEncoder<B: Backend, Q: QueueKind = Graphics> {
    device: Rhi<B>,
    raw: B::CommandBuffer,
    pool: B::CommandPool,
    retained: HashMap<u64, Retained>,
    images: HashMap<u64, ImageUse<B>>,
    in_pass: bool,
    failed: bool,
    kind: PhantomData<Q>,
}
/// Finished, single-submit commands. Consumed by their compatible queue.
pub struct RecordedCommands<B: Backend, Q: QueueKind = Graphics> {
    device: Rhi<B>,
    raw: B::CommandBuffer,
    pool: B::CommandPool,
    retained: HashMap<u64, Retained>,
    images: HashMap<u64, ImageUse<B>>,
    kind: PhantomData<Q>,
}
struct ImageUse<B: Backend> {
    image: GpuImage<B>,
    first: Vec<Option<Access>>,
    last: Vec<Option<Access>>,
}
impl<B: Backend> Rhi<B> {
    /// Selects a queue capability without exposing native handles.
    #[must_use]
    pub fn queue<Q: QueueKind>(&self) -> Queue<B, Q> {
        Queue {
            device: self.clone(),
            kind: PhantomData,
        }
    }
    /// Begins a new exclusive recording. Native pools are owned by the recording through completion.
    pub fn create_encoder<Q: QueueKind>(&self, label: &str) -> Result<CommandEncoder<B, Q>> {
        self.collect_garbage()?;
        let _gate = lock(&self.0.gate)?;
        let mut pool = unsafe { self.0.backend.create_command_pool(QueueType::Graphics)? };
        let mut raw = unsafe { self.0.backend.create_command_buffer(&mut pool)? };
        unsafe {
            raw.begin(label, true)?;
        }
        Ok(CommandEncoder {
            device: self.clone(),
            raw,
            pool,
            retained: HashMap::new(),
            images: HashMap::new(),
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
    fn retain<T: Send + Sync + 'static, M: Send + Sync + 'static>(
        &mut self,
        object: &Object<B, T, M>,
    ) -> Result<()> {
        object.require_device(&self.device.0.backend)?;
        self.retained
            .entry(object.id())
            .or_insert_with(|| Retained {
                _object: object.0.clone(),
                busy: object.0.busy.clone(),
            });
        Ok(())
    }
    fn buffer(
        &mut self,
        buffer: &GpuBuffer<B>,
        usage: BufferUsages,
        range: crate::BufferRange,
    ) -> Result<crate::BufferRange> {
        if !buffer.info().usage.contains(usage) {
            return Err(Ir::Mismatch.into());
        }
        let range = range.resolve(buffer.size())?;
        self.retain(buffer)?;
        Ok(range)
    }
    fn memory_dependency(&mut self) -> Result<()> {
        let barriers = [crate::MemoryBarrier {
            src_stages: crate::PipelineStages::ALL,
            dst_stages: crate::PipelineStages::ALL,
            src_access: crate::AccessTypes::MEMORY_WRITE,
            dst_access: crate::AccessTypes::MEMORY_READ | crate::AccessTypes::MEMORY_WRITE,
        }];
        let result = unsafe {
            self.raw.barrier(&crate::DependencyInfo {
                memory_barriers: &barriers,
                buffer_barriers: &[],
                image_barriers: &[],
            })
        };
        self.record(result)
    }
    /// Establishes a semantic image access. Actual initial states are reconciled at submission.
    pub fn transition(
        &mut self,
        image: &GpuImage<B>,
        access: Access,
        range: ImageSubresourceRange,
    ) -> Result<()> {
        self.outside()?;
        self.image_access(image, access, range)
    }
    fn image_access(
        &mut self,
        image: &GpuImage<B>,
        access: Access,
        range: ImageSubresourceRange,
    ) -> Result<()> {
        image.require_live()?;
        if access == Access::Undefined {
            return Err(Ir::Mismatch
                .with_detail("Undefined is an initial state, not an executable access")
                .into());
        }
        self.retain(image)?;
        let info = image.description();
        let range = range.resolve(&info)?;
        if access.image_usage().is_empty() && access != Access::Undefined {
            return Err(Ir::Mismatch.into());
        }
        if !info.usage.contains(access.image_usage()) {
            return Err(Ir::Mismatch.into());
        }
        let count = usize::try_from(u64::from(info.mip_levels) * u64::from(info.array_layers))
            .map_err(|_| Ir::OutOfRange)?;
        let usage = self.images.entry(image.id()).or_insert_with(|| ImageUse {
            image: image.clone(),
            first: vec![None; count],
            last: vec![None; count],
        });
        let mut barriers = Vec::new();
        for layer in range.base_array_layer..range.base_array_layer + range.array_layer_count {
            for mip in range.base_mip_level..range.base_mip_level + range.mip_level_count {
                let index = (layer * info.mip_levels + mip) as usize;
                if let Some(old) = usage.last[index] {
                    if old != access || old.writes() || access.writes() {
                        if self.in_pass {
                            return Err(Ir::BadState.with_detail("resource transition required inside a render pass; declare access before beginning the pass").into());
                        }
                        barriers.push(crate::ImageBarrier::<B> {
                            image: image.raw(),
                            old_state: old,
                            new_state: access,
                            aspects: info.format.aspects(),
                            base_mip_level: mip,
                            mip_level_count: 1,
                            base_array_layer: layer,
                            array_layer_count: 1,
                            queue_transfer: None,
                        });
                    }
                } else {
                    usage.first[index] = Some(access);
                }
                usage.last[index] = Some(access);
            }
        }
        if !barriers.is_empty() {
            let result = unsafe {
                self.raw.barrier(&crate::DependencyInfo {
                    memory_barriers: &[],
                    buffer_barriers: &[],
                    image_barriers: &barriers,
                })
            };
            self.record(result)?;
        }
        Ok(())
    }
    /// Applies graph-generated semantic dependencies, checking all ranges and deriving native states.
    pub fn barrier(&mut self, info: &crate::DependencyInfo<'_, Rhi<B>>) -> Result<()> {
        self.outside()?;
        for barrier in info.image_barriers {
            if barrier.queue_transfer.is_some() {
                return Err(crate::UnsupportedOperation::Capability(
                    "exclusive cross-queue ownership in the safe scheduler",
                )
                .into());
            }
            self.transition(
                barrier.image,
                barrier.new_state,
                ImageSubresourceRange {
                    aspects: barrier.aspects,
                    base_mip_level: barrier.base_mip_level,
                    mip_level_count: barrier.mip_level_count,
                    base_array_layer: barrier.base_array_layer,
                    array_layer_count: barrier.array_layer_count,
                },
            )?;
        }
        for barrier in info.buffer_barriers {
            if barrier.queue_transfer.is_some() {
                return Err(crate::UnsupportedOperation::Capability(
                    "exclusive cross-queue ownership in the safe scheduler",
                )
                .into());
            }
            self.buffer(
                barrier.buffer,
                barrier.new_state.buffer_usage(),
                crate::BufferRange {
                    offset: barrier.offset,
                    size: barrier.size,
                },
            )?;
        }
        if !info.buffer_barriers.is_empty() || !info.memory_barriers.is_empty() {
            self.memory_dependency()?;
        }
        Ok(())
    }
    /// Copies checked, non-overlapping byte ranges.
    pub fn copy_buffer(
        &mut self,
        src: &GpuBuffer<B>,
        dst: &GpuBuffer<B>,
        regions: &[crate::BufferCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            self.buffer(
                src,
                BufferUsages::COPY_SRC,
                crate::BufferRange {
                    offset: region.src_offset,
                    size: region.size,
                },
            )?;
            self.buffer(
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
                || src.id() == dst.id()
            {
                return Err(Ir::Mismatch.into());
            }
        }
        self.memory_dependency()?;
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
        let caps = self.device.capabilities();
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
        self.buffer(
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
        self.image_access(
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
    pub fn copy_buffer_to_image(
        &mut self,
        src: &GpuBuffer<B>,
        dst: &GpuImage<B>,
        regions: &[crate::BufferImageCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            self.buffer_image(src, dst, region, true)?;
        }
        self.memory_dependency()?;
        let result = unsafe { self.raw.copy_buffer_to_image(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Copies image regions to a readback buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuBuffer<B>,
        regions: &[crate::BufferImageCopy],
    ) -> Result<()> {
        self.outside()?;
        for region in regions {
            self.buffer_image(dst, src, region, false)?;
        }
        self.memory_dependency()?;
        let result = unsafe { self.raw.copy_image_to_buffer(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Copies matching formats without filtering.
    pub fn copy_image(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuImage<B>,
        regions: &[crate::ImageCopy],
    ) -> Result<()> {
        self.outside()?;
        if src.id() == dst.id()
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
            self.image_access(src, Access::CopySource, a)?;
            self.image_access(dst, Access::CopyDestination, b)?;
        }
        let result = unsafe { self.raw.copy_image(src.raw(), dst.raw(), regions) };
        self.record(result)
    }
    /// Blits exact regions only when the backend advertises support.
    pub fn blit_image(
        &mut self,
        src: &GpuImage<B>,
        dst: &GpuImage<B>,
        regions: &[crate::ImageBlit],
        filter: crate::FilterMode,
    ) -> Result<()> {
        self.outside()?;
        let support = self.device.format_capabilities(src.description().format);
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
            if src.id() == dst.id() && r.src_mip_level == r.dst_mip_level {
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
            self.image_access(src, Access::CopySource, a)?;
            self.image_access(dst, Access::CopyDestination, b)?;
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
            device: self.device,
            raw: self.raw,
            pool: self.pool,
            retained: self.retained,
            images: self.images,
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
    pipeline: Option<GpuGraphicsPipeline<B>>,
    groups: HashMap<u32, GpuBindGroup<B>>,
    vertices: HashMap<u32, (GpuBuffer<B>, u64)>,
    index: Option<(GpuBuffer<B>, u64, crate::IndexFormat)>,
    viewport: bool,
    scissor: bool,
}
use crate::TextureFormat;
impl<B: Backend> CommandEncoder<B, Graphics> {
    /// Begins a graphics scope borrowing this encoder exclusively.
    pub fn begin_render_pass(
        &mut self,
        info: &crate::RenderingInfo<'_, Rhi<B>>,
    ) -> Result<RenderPass<'_, B>> {
        self.outside()?;
        if info.width == 0
            || info.height == 0
            || info.layer_count == 0
            || info.color_attachments.len()
                > self.device.capabilities().limits.max_color_attachments as usize
            || (info.color_attachments.is_empty() && info.depth_attachment.is_none())
        {
            return Err(Ir::OutOfRange.into());
        }
        let mut samples = None;
        let mut views = Vec::new();
        for color in info.color_attachments {
            self.attachment(color.view, info, &mut samples)?;
            self.image_access(
                &color.view.info().image,
                Access::ColorAttachment,
                color.view.info().range,
            )?;
            if let Some(resolve) = color.resolve {
                let mut resolve_samples = None;
                self.attachment(resolve, info, &mut resolve_samples)?;
                if resolve_samples != Some(crate::SampleCount::One)
                    || samples == Some(crate::SampleCount::One)
                    || resolve.info().image.description().format
                        != color.view.info().image.description().format
                {
                    return Err(Ir::Mismatch.into());
                }
                self.image_access(
                    &resolve.info().image,
                    Access::ColorAttachment,
                    resolve.info().range,
                )?;
            }
            views.push(crate::ColorAttachment::<B> {
                view: color.view.raw(),
                resolve: color.resolve.map(Object::raw),
                load: color.load,
                store: color.store,
            });
        }
        let depth = if let Some(depth) = &info.depth_attachment {
            self.attachment(depth.view, info, &mut samples)?;
            self.image_access(
                &depth.view.info().image,
                Access::DepthStencilAttachment,
                depth.view.info().range,
            )?;
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
        self.memory_dependency()?;
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
                .map(|a| a.view.info().image.description().format)
                .collect(),
            depth: info
                .depth_attachment
                .as_ref()
                .map(|a| a.view.info().image.description().format),
            samples: samples.unwrap_or_default(),
            pipeline: None,
            groups: HashMap::new(),
            vertices: HashMap::new(),
            index: None,
            viewport: false,
            scissor: false,
        })
    }
    fn attachment(
        &mut self,
        view: &GpuImageView<B>,
        info: &crate::RenderingInfo<'_, Rhi<B>>,
        samples: &mut Option<crate::SampleCount>,
    ) -> Result<()> {
        self.retain(view)?;
        let desc = view.info().image.description();
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
    /// Borrows the active native rendering command stream.
    ///
    /// # Safety
    /// External commands must obey the active pass, preserve tracked resource
    /// states, and retain their resources through submission completion. Restore
    /// any changed bindings/dynamic state before subsequent portable draws. Do
    /// not begin/end the pass or command buffer through this reference.
    pub unsafe fn native(&mut self) -> &mut B::CommandBuffer {
        &mut self.encoder.raw
    }

    /// Binds a pipeline compatible with this pass's attachments.
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GpuGraphicsPipeline<B>) -> Result<()> {
        if pipeline
            .info()
            .colors
            .iter()
            .map(|c| c.format)
            .collect::<Vec<_>>()
            != self.colors
            || pipeline.info().depth.map(|d| d.format) != self.depth
            || pipeline.info().samples != self.samples
        {
            return Err(Ir::Mismatch.into());
        }
        self.encoder.retain(pipeline)?;
        let result = unsafe { self.encoder.raw.bind_graphics_pipeline(pipeline.raw()) };
        self.encoder.record(result)?;
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }
    /// Binds validated groups, retaining their transitively referenced resources.
    pub fn bind_groups(
        &mut self,
        layout: &GpuPipelineLayout<B>,
        first_group: u32,
        groups: &[&GpuBindGroup<B>],
        dynamic_offsets: &[u64],
    ) -> Result<()> {
        self.encoder.retain(layout)?;
        let mut offsets = dynamic_offsets.iter();
        let pipeline = self.pipeline.as_ref().ok_or(Ir::BadState)?;
        if pipeline.info().layout.id() != layout.id() {
            return Err(Ir::Mismatch.into());
        }
        for (index, group) in groups.iter().enumerate() {
            let slot = first_group
                .checked_add(u32::try_from(index).map_err(|_| Ir::OutOfRange)?)
                .ok_or(Ir::OutOfRange)?;
            let expected = layout.info().get(slot as usize).ok_or(Ir::OutOfRange)?;
            if expected.info() != group.info().layout.info() {
                return Err(Ir::Mismatch.into());
            }
            self.encoder.retain(group)?;
            for ((_, resource), entry) in group.info().entries.iter().zip(expected.info()) {
                match resource {
                    OwnedBinding::Buffer(buffer, range, ty) => {
                        let (dynamic, usage) = match ty {
                            crate::BindingType::UniformBuffer { dynamic_offset } => {
                                (*dynamic_offset, BufferUsages::UNIFORM)
                            }
                            crate::BindingType::StorageBuffer {
                                read_only: true,
                                dynamic_offset,
                            } => (*dynamic_offset, BufferUsages::STORAGE),
                            _ => {
                                return Err(crate::UnsupportedOperation::Capability(
                                    "writable storage within graphics passes",
                                )
                                .into());
                            }
                        };
                        let offset = if dynamic {
                            *offsets.next().ok_or(Ir::Mismatch)?
                        } else {
                            0
                        };
                        let caps = self.encoder.device.capabilities();
                        let alignment = if usage == BufferUsages::UNIFORM {
                            caps.min_uniform_buffer_offset_alignment
                        } else {
                            caps.min_storage_buffer_offset_alignment
                        }
                        .max(1);
                        if !offset.is_multiple_of(alignment) {
                            return Err(Ir::Mismatch.into());
                        }
                        self.encoder.buffer(
                            buffer,
                            usage,
                            crate::BufferRange {
                                offset: range.offset.checked_add(offset).ok_or(Ir::OutOfRange)?,
                                size: range.size,
                            },
                        )?;
                    }
                    OwnedBinding::Sampled(view, sampler) => {
                        self.encoder.retain(view)?;
                        self.encoder.retain(sampler)?;
                        self.encoder.image_access(
                            &view.info().image,
                            Access::ShaderRead(entry.visibility),
                            view.info().range,
                        )?;
                    }
                    OwnedBinding::Storage(_) => {
                        return Err(crate::UnsupportedOperation::Capability(
                            "writable storage within graphics passes",
                        )
                        .into());
                    }
                }
            }
            self.groups.insert(slot, (*group).clone());
        }
        if offsets.next().is_some() {
            return Err(Ir::Mismatch.into());
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
    pub fn bind_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &GpuBuffer<B>,
        offset: u64,
    ) -> Result<()> {
        if slot >= self.encoder.device.capabilities().limits.max_vertex_buffers {
            return Err(Ir::OutOfRange.into());
        }
        self.encoder.buffer(
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
        self.vertices.insert(slot, (buffer.clone(), offset));
        Ok(())
    }
    /// Binds an index buffer aligned to its element size.
    pub fn bind_index_buffer(
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
        self.encoder.buffer(
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
        self.index = Some((buffer.clone(), offset, format));
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
        let pipeline = self.pipeline.as_ref().ok_or(Ir::BadState)?;
        if !self.viewport || !self.scissor || self.encoder.failed {
            return Err(Ir::BadState.into());
        }
        for (slot, layout) in pipeline.info().layout.info().iter().enumerate() {
            if !layout.info().is_empty()
                && self
                    .groups
                    .get(&(u32::try_from(slot).map_err(|_| Ir::OutOfRange)?))
                    .is_none_or(|g| g.info().layout.info() != layout.info())
            {
                return Err(Ir::Mismatch.into());
            }
        }
        for (slot, _) in pipeline.info().vertices.iter().enumerate() {
            if !self
                .vertices
                .contains_key(&(u32::try_from(slot).map_err(|_| Ir::OutOfRange)?))
            {
                return Err(Ir::BadState.into());
            }
        }
        Ok(())
    }
    /// Records a non-indexed draw after validating bindings.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<()> {
        self.ready()?;
        let pipeline = self.pipeline.as_ref().ok_or(Ir::BadState)?;
        for (slot, (stride, step, _)) in pipeline.info().vertices.iter().enumerate() {
            let (buffer, offset) = self
                .vertices
                .get(&(u32::try_from(slot).map_err(|_| Ir::OutOfRange)?))
                .ok_or(Ir::BadState)?;
            let end = if *step == crate::VertexStepMode::Vertex {
                first_vertex.checked_add(vertex_count)
            } else {
                first_instance.checked_add(instance_count)
            }
            .ok_or(Ir::OutOfRange)?;
            if u64::from(end)
                .checked_mul(*stride)
                .and_then(|n| offset.checked_add(n))
                .is_none_or(|n| n > buffer.size())
            {
                return Err(Ir::OutOfRange.into());
            }
        }
        let result = unsafe {
            self.encoder
                .raw
                .draw(vertex_count, instance_count, first_vertex, first_instance)
        };
        self.encoder.record(result)
    }
    /// Records an indexed draw. Native backends must enable robust vertex fetching.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<()> {
        self.ready()?;
        let (buffer, offset, format) = self.index.as_ref().ok_or(Ir::BadState)?;
        let bytes = if *format == crate::IndexFormat::Uint16 {
            2
        } else {
            4
        };
        if u64::from(first_index)
            .checked_add(u64::from(index_count))
            .and_then(|n| n.checked_mul(bytes))
            .and_then(|n| n.checked_add(*offset))
            .is_none_or(|n| n > buffer.size())
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
            frame.require_device(&device.0.backend)?;
            if !frame_ids.insert(frame.state.id) {
                return Err(Ir::BadState.into());
            }
        }
        let mut signals = std::collections::HashSet::new();
        for point in info.wait_timelines.iter().chain(info.signal_timelines) {
            point.semaphore.require_device(&device.0.backend)?;
        }
        for point in info.wait_timelines {
            if point.value > point.semaphore.info().load(Ordering::Acquire) {
                return Err(Ir::BadState.with_detail("wait must reference an already scheduled signal; future waits could deadlock an aliased queue").into());
            }
        }
        for point in info.signal_timelines {
            if !signals.insert(point.semaphore.id())
                || point.value <= point.semaphore.info().load(Ordering::Acquire)
            {
                return Err(Ir::BadState
                    .with_detail("timeline signals must increase monotonically")
                    .into());
            }
        }
        // Validate all acquisitions before native submission or changing shared state.
        for command in commands {
            if !Arc::ptr_eq(&command.device.0, &device.0) {
                return Err(Ir::ForeignInstance.into());
            }
            for usage in command.images.values() {
                state
                    .image_lifetimes
                    .insert(usage.image.id(), Arc::downgrade(&usage.image.0.busy));
                usage.image.require_live()?;
                if let Some(frame) = &usage.image.info().frame
                    && !frame_ids.contains(&frame.id)
                {
                    return Err(Ir::BadState.with_detail("surface image used without associating its acquisition with submission").into());
                }
            }
        }
        Ok(())
    }
    fn prepare_recording(
        &self,
        command: RecordedCommands<B, Q>,
        resulting: &mut HashMap<u64, Vec<Access>>,
    ) -> Result<Payload<B>> {
        let device = &self.device;
        let mut pool = unsafe { device.0.backend.create_command_pool(QueueType::Graphics)? };
        let mut prologue = unsafe { device.0.backend.create_command_buffer(&mut pool)? };
        unsafe {
            prologue.begin("RHI submission dependencies", true)?;
        }
        let memory = [crate::MemoryBarrier {
            src_stages: crate::PipelineStages::ALL,
            dst_stages: crate::PipelineStages::ALL,
            src_access: crate::AccessTypes::MEMORY_WRITE,
            dst_access: crate::AccessTypes::MEMORY_READ | crate::AccessTypes::MEMORY_WRITE,
        }];
        let mut barriers = Vec::new();
        for usage in command.images.values() {
            let desc = usage.image.description();
            let previous = resulting
                .entry(usage.image.id())
                .or_insert_with(|| vec![Access::Undefined; usage.first.len()]);
            for (index, first) in usage.first.iter().enumerate() {
                if let Some(first) = first {
                    let old = previous[index];
                    if old != *first || old.writes() || first.writes() {
                        barriers.push(crate::ImageBarrier::<B> {
                            image: usage.image.raw(),
                            old_state: old,
                            new_state: *first,
                            aspects: desc.format.aspects(),
                            base_mip_level: (u32::try_from(index).map_err(|_| Ir::OutOfRange)?)
                                % desc.mip_levels,
                            mip_level_count: 1,
                            base_array_layer: (u32::try_from(index).map_err(|_| Ir::OutOfRange)?)
                                / desc.mip_levels,
                            array_layer_count: 1,
                            queue_transfer: None,
                        });
                    }
                    previous[index] = usage.last[index].unwrap_or(*first);
                }
            }
        }
        unsafe {
            prologue.barrier(&crate::DependencyInfo {
                memory_barriers: &memory,
                buffer_barriers: &[],
                image_barriers: &barriers,
            })?;
            prologue.end()?;
        }
        Ok(Payload {
            commands: vec![(prologue, pool), (command.raw, command.pool)],
            retained: command.retained.into_values().collect(),
        })
    }
    fn prepare_present(
        &self,
        frames: &[&GpuSurfaceFrame<B>],
        resulting: &mut HashMap<u64, Vec<Access>>,
        native: &mut Vec<(B::CommandBuffer, B::CommandPool)>,
        retained: &mut Vec<Retained>,
    ) -> Result<()> {
        let device = &self.device;
        // A final transition owns presentation even when the graph omitted an export.
        if !frames.is_empty() {
            let mut pool = unsafe { device.0.backend.create_command_pool(QueueType::Graphics)? };
            let mut epilogue = unsafe { device.0.backend.create_command_buffer(&mut pool)? };
            unsafe {
                epilogue.begin("RHI presentation dependencies", true)?;
            }
            let mut barriers = Vec::new();
            for frame in frames {
                let image = frame.image();
                let previous = resulting
                    .entry(image.id())
                    .or_insert_with(|| vec![Access::Undefined]);
                barriers.push(crate::ImageBarrier::<B> {
                    image: image.raw(),
                    old_state: previous[0],
                    new_state: Access::Present,
                    aspects: crate::ImageAspects::COLOR,
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    queue_transfer: None,
                });
                previous[0] = Access::Present;
                retained.push(Retained {
                    _object: image.0.clone(),
                    busy: image.0.busy.clone(),
                });
            }
            unsafe {
                epilogue.barrier(&crate::DependencyInfo {
                    memory_barriers: &[],
                    buffer_barriers: &[],
                    image_barriers: &barriers,
                })?;
                epilogue.end()?;
            }
            native.push((epilogue, pool));
        }
        Ok(())
    }

    /// Submits finished recordings once. Failure poisons the device if native submission may have started.
    /// Initial image states are reconciled between recordings in submission order.
    pub fn submit(
        &self,
        commands: Vec<RecordedCommands<B, Q>>,
        info: &SubmitInfo<'_, B>,
    ) -> Result<Completion<B>> {
        let device = &self.device;
        let _gate = lock(&device.0.gate)?;
        let mut state = lock(&device.0.state)?;
        if state.lost {
            return Err(crate::Error::DeviceLost);
        }
        self.validate_submission(&commands, info, &mut state)?;
        let mut resulting = state.images.clone();
        let mut native = Vec::new();
        let mut retained = Vec::new();
        for command in commands {
            let payload = self.prepare_recording(command, &mut resulting)?;
            native.extend(payload.commands);
            retained.extend(payload.retained);
        }
        self.prepare_present(
            info.surface_frames,
            &mut resulting,
            &mut native,
            &mut retained,
        )?;
        for point in info.wait_timelines.iter().chain(info.signal_timelines) {
            retained.push(Retained {
                _object: point.semaphore.0.clone(),
                busy: point.semaphore.0.busy.clone(),
            });
        }
        let fence = unsafe { device.0.backend.create_fence(false)? };
        let work = Arc::new(Work {
            _backend: device.0.backend.clone(),
            fence,
            payload: Mutex::new(Some(Payload {
                commands: native,
                retained,
            })),
        });
        let payload = lock(&work.payload)?;
        let batch = payload.as_ref().ok_or(Ir::BadState)?;
        let native_commands: Vec<_> = batch.commands.iter().map(|(command, _)| command).collect();
        let mut frames = info
            .surface_frames
            .iter()
            .map(|f| lock(&f.native))
            .collect::<Result<Vec<_>>>()?;
        let native_frames = frames
            .iter()
            .map(|f| f.as_ref().ok_or_else(|| crate::Error::from(Ir::BadState)))
            .collect::<Result<Vec<_>>>()?;
        let waits: Vec<_> = info
            .wait_timelines
            .iter()
            .map(|p| crate::TimelinePoint::<B> {
                semaphore: p.semaphore.raw(),
                value: p.value,
                stages: p.stages,
            })
            .collect();
        let signals: Vec<_> = info
            .signal_timelines
            .iter()
            .map(|p| crate::TimelinePoint::<B> {
                semaphore: p.semaphore.raw(),
                value: p.value,
                stages: p.stages,
            })
            .collect();
        for resource in &batch.retained {
            resource.busy.fetch_add(1, Ordering::AcqRel);
        }
        let result = unsafe {
            device.0.backend.submit(
                QueueType::Graphics,
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
            frame.state.phase.store(1, Ordering::Release);
        }
        for point in info.signal_timelines {
            point.semaphore.info().store(point.value, Ordering::Release);
        }
        state.images = resulting;
        state.pending.push(work.clone());
        Ok(Completion(work))
    }
}
