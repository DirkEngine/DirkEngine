#![allow(
    clippy::missing_errors_doc,
    reason = "command recording errors are backend-dependent and use the crate Error contract"
)]

//! Command recording is currently focused on graphics and transfer work.
//!
//! Compute queue and shader types are part of the shared resource vocabulary,
//! but this layer intentionally does not yet define compute-pipeline creation,
//! binding, or dispatch. A future compute-capable command contract will add
//! those operations together rather than making the existing partial surface
//! look complete.

use std::num::NonZeroU32;

use crate::{
    AccessTypes, Api, Color, Extent3d, FilterMode, ImageAspects, ImageState, IndexFormat, LoadOp,
    Origin3d, PipelineStages, QueueTransfer, QueueType, Rect, Result, StoreOp, Viewport,
};

/// Buffer-to-buffer copy region.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferCopy {
    /// Source byte offset.
    pub src_offset: u64,
    /// Destination byte offset.
    pub dst_offset: u64,
    /// Number of bytes copied.
    pub size: u64,
}

/// Layout shared by buffer-to-image and image-to-buffer copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferImageCopy {
    /// Byte offset of the first copied texel in the buffer. This must meet
    /// [`crate::Capabilities::buffer_copy_offset_alignment`].
    pub buffer_offset: u64,
    /// Byte distance between adjacent image rows in the buffer. This must meet
    /// [`crate::Capabilities::buffer_copy_row_pitch_alignment`].
    pub buffer_bytes_per_row: NonZeroU32,
    /// Number of buffer rows between adjacent depth slices or array layers.
    pub buffer_rows_per_image: NonZeroU32,
    /// Copied image mip level.
    pub mip_level: u32,
    /// First copied image array layer.
    pub base_array_layer: u32,
    /// Copied image layer count.
    pub array_layer_count: u32,
    /// Image-space origin of the copied region.
    pub image_origin: Origin3d,
    /// Copied image extent.
    pub extent: Extent3d,
    /// Copied image aspects.
    pub aspects: ImageAspects,
}

/// Image-to-image copy region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageCopy {
    /// Source mip level.
    pub src_mip_level: u32,
    /// First source array layer.
    pub src_base_array_layer: u32,
    /// Destination mip level.
    pub dst_mip_level: u32,
    /// First destination array layer.
    pub dst_base_array_layer: u32,
    /// Number of copied array layers.
    pub array_layer_count: u32,
    /// Source texel origin.
    pub src_origin: Origin3d,
    /// Destination texel origin.
    pub dst_origin: Origin3d,
    /// Copied image extent.
    pub extent: Extent3d,
    /// Copied image aspects.
    pub aspects: ImageAspects,
}

/// Image blit region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageBlit {
    /// Source mip level.
    pub src_mip_level: u32,
    /// Destination mip level.
    pub dst_mip_level: u32,
    /// First source array layer.
    pub src_base_array_layer: u32,
    /// First destination array layer.
    pub dst_base_array_layer: u32,
    /// Number of blitted array layers.
    pub array_layer_count: u32,
    /// Source texel origin.
    pub src_origin: Origin3d,
    /// Destination texel origin.
    pub dst_origin: Origin3d,
    /// Source dimensions.
    pub src_extent: Extent3d,
    /// Destination dimensions.
    pub dst_extent: Extent3d,
    /// Blitted image aspects.
    pub aspects: ImageAspects,
}

/// Buffer transition or visibility range.
pub struct BufferBarrier<'a, B: Api> {
    /// Buffer whose accesses are ordered.
    pub buffer: &'a B::Buffer,
    /// First byte covered by the barrier.
    pub offset: u64,
    /// Number of bytes covered by the barrier; use
    /// [`BufferBarrier::REMAINING_SIZE`] to cover from `offset` through the end
    /// of the buffer.
    pub size: u64,
    /// Semantic access before the dependency.
    pub old_state: crate::ResourceAccess,
    /// Semantic access after the dependency.
    pub new_state: crate::ResourceAccess,
    /// Optional transfer of exclusive queue ownership.
    pub queue_transfer: Option<QueueTransfer>,
}

impl<B: Api> BufferBarrier<'_, B> {
    /// Api-neutral remaining-range sentinel for
    /// [`BufferBarrier::size`](BufferBarrier::size) that covers the buffer's
    /// allocation from `offset` through its end.
    ///
    /// Backends must translate this shared sentinel to their native
    /// whole-range representation (for example `VK_WHOLE_SIZE`) instead of
    /// recording the raw value as a byte count.
    pub const REMAINING_SIZE: u64 = u64::MAX;
}

/// Global memory dependency between pipeline stages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBarrier {
    /// Pipeline stages before the dependency.
    pub src_stages: PipelineStages,
    /// Pipeline stages after the dependency.
    pub dst_stages: PipelineStages,
    /// Memory accesses completed before the dependency.
    pub src_access: AccessTypes,
    /// Memory accesses made visible after the dependency.
    pub dst_access: AccessTypes,
}

/// Image transition generated by the render graph or upload path.
pub struct ImageBarrier<'a, B: Api> {
    /// Transitioned image.
    pub image: &'a B::Image,
    /// State before the transition.
    pub old_state: ImageState,
    /// State after the transition.
    pub new_state: ImageState,
    /// Selected image aspects.
    pub aspects: ImageAspects,
    /// First mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    pub mip_level_count: u32,
    /// First array layer.
    pub base_array_layer: u32,
    /// Array layer count.
    pub array_layer_count: u32,
    /// Optional transfer of exclusive queue ownership.
    pub queue_transfer: Option<QueueTransfer>,
}

/// A group of resource transitions.
pub struct DependencyInfo<'a, B: Api> {
    /// Global memory dependencies.
    pub memory_barriers: &'a [MemoryBarrier],
    /// Buffer visibility ranges.
    pub buffer_barriers: &'a [BufferBarrier<'a, B>],
    /// Image transitions.
    pub image_barriers: &'a [ImageBarrier<'a, B>],
}

/// Color attachment used by a dynamic rendering pass.
pub struct ColorAttachment<'a, B: Api> {
    /// Render target view.
    pub view: &'a B::ImageView,
    /// Optional multisample resolve target.
    pub resolve: Option<&'a B::ImageView>,
    /// Initial contents operation.
    pub load: LoadOp<Color>,
    /// Final contents operation.
    pub store: StoreOp,
}

/// Depth/stencil attachment used by a dynamic rendering pass.
pub struct DepthAttachment<'a, B: Api> {
    /// Depth target view.
    pub view: &'a B::ImageView,
    /// Initial depth operation.
    pub depth_load: LoadOp<f32>,
    /// Final depth operation.
    pub depth_store: StoreOp,
    /// Initial stencil operation.
    pub stencil_load: LoadOp<u32>,
    /// Final stencil operation.
    pub stencil_store: StoreOp,
}

/// Dynamic rendering pass description.
pub struct RenderingInfo<'a, B: Api> {
    /// Debug label.
    pub label: &'a str,
    /// Render target width.
    pub width: u32,
    /// Render target height.
    pub height: u32,
    /// Number of array layers rendered by this pass.
    pub layer_count: u32,
    /// Color attachments.
    pub color_attachments: &'a [ColorAttachment<'a, B>],
    /// Optional depth/stencil attachment.
    pub depth_attachment: Option<DepthAttachment<'a, B>>,
}

/// Timeline semaphore and target value used by a submission.
pub struct TimelinePoint<'a, B: Api> {
    /// Timeline semaphore.
    pub semaphore: &'a B::TimelineSemaphore,
    /// Counter value waited for or signaled.
    pub value: u64,
    /// Pipeline stages blocked by a wait or after which a signal occurs.
    pub stages: PipelineStages,
}

/// Native recording operations. Called only with validated state and resources.
///
/// # Safety
/// Callers must satisfy the native backend contract, retain resources, and establish
/// dependencies. Implementations must encode exactly the requested operations.
pub unsafe trait CommandBuffer<B: Api>: Send {
    /// Returns the semantic queue this command buffer records for.
    fn queue_type(&self) -> QueueType;
    /// Begins command recording.
    ///
    /// Implementations must return
    /// [`InvalidResourceKind::BadState`](crate::InvalidResourceKind::BadState)
    /// rather than resetting a command buffer whose prior submission is still
    /// executing.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn begin(&mut self, label: &str, one_time_submit: bool) -> Result<()>;
    /// Ends command recording.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn end(&mut self) -> Result<()>;
    /// Begins a rendering pass.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn begin_rendering(&mut self, info: &RenderingInfo<'_, B>) -> Result<()>;
    /// Ends the active rendering pass.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn end_rendering(&mut self) -> Result<()>;
    /// Sets the active viewport.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn set_viewport(&mut self, viewport: Viewport) -> Result<()>;
    /// Sets the active scissor rectangle.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn set_scissor(&mut self, scissor: Rect) -> Result<()>;
    /// Sets the constant blend factors applied to blended color attachments.
    ///
    /// Factors are used only while a pipeline with blending is bound.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn set_blend_constants(&mut self, color: Color) -> Result<()>;
    /// Sets the front- and back-face stencil reference values used by
    /// stencil comparisons and replace operations.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn set_stencil_reference(&mut self, front: u32, back: u32) -> Result<()>;
    /// Binds a graphics pipeline.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn bind_graphics_pipeline(&mut self, pipeline: &B::GraphicsPipeline) -> Result<()>;
    /// Binds resource groups to a pipeline layout.
    ///
    /// `dynamic_offsets` are ordered first by bind-group order and then by
    /// ascending binding number among layout entries whose buffer type enables
    /// dynamic offsets.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn bind_groups(
        &mut self,
        layout: &B::PipelineLayout,
        first_group: u32,
        groups: &[&B::BindGroup],
        dynamic_offsets: &[u64],
    ) -> Result<()>;
    /// Binds one vertex buffer.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn bind_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &B::Buffer,
        offset: u64,
    ) -> Result<()>;
    /// Binds an index buffer.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn bind_index_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<()>;
    /// Records a non-indexed draw.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<()>;
    /// Records an indexed draw.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<()>;
    /// Copies buffer regions.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn copy_buffer(
        &mut self,
        src: &B::Buffer,
        dst: &B::Buffer,
        regions: &[BufferCopy],
    ) -> Result<()>;
    /// Copies a buffer into an image.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn copy_buffer_to_image(
        &mut self,
        src: &B::Buffer,
        dst: &B::Image,
        regions: &[BufferImageCopy],
    ) -> Result<()>;
    /// Copies an image into a buffer.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn copy_image_to_buffer(
        &mut self,
        src: &B::Image,
        dst: &B::Buffer,
        regions: &[BufferImageCopy],
    ) -> Result<()>;
    /// Copies image regions without filtering.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn copy_image(
        &mut self,
        src: &B::Image,
        dst: &B::Image,
        regions: &[ImageCopy],
    ) -> Result<()>;
    /// Blits between image mip levels.
    ///
    /// # Errors
    ///
    /// Returns an error when the source and destination formats are not
    /// compatible with the backend's blit rules.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn blit_image(
        &mut self,
        src: &B::Image,
        dst: &B::Image,
        regions: &[ImageBlit],
        filter: FilterMode,
    ) -> Result<()>;
    /// Applies resource transitions.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn barrier(&mut self, dependency: &DependencyInfo<'_, B>) -> Result<()>;
}
