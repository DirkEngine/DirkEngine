use std::fmt::Debug;

use raw_window_handle::{DisplayHandle, WindowHandle};

use crate::{
    BindGroupDesc, BindGroupLayoutDesc, BufferDesc, GraphicsPipelineDesc, ImageDesc, ImageUsages,
    ImageViewDesc, NativeBuffer, NativeCommandBuffer, NativeSurfaceFrame, NativeSwapchain,
    PipelineLayoutDesc, QueueType, Result, SampleCounts, SamplerDesc, ShaderDesc,
    SurfaceCreateInfo, SwapchainDesc, TextureFormat, TimelinePoint,
};

/// Application metadata and backend policy used during device creation.
pub struct RhiCreateInfo<'a> {
    /// Engine name exposed to graphics diagnostics.
    pub engine_name: &'a str,
    /// Engine semantic version.
    pub engine_version: (u32, u32, u32),
    /// Application name exposed to graphics diagnostics.
    pub application_name: &'a str,
    /// Application semantic version.
    pub application_version: (u32, u32, u32),
    /// Enables backend validation when available.
    pub validation: bool,
    /// Display and window handles the selected device must be able to present
    /// to.
    ///
    /// Backends may inspect these handles during device selection but must not
    /// retain them after `Rhi::new` returns.
    ///
    /// Headless users may leave this unset and create a surface later, at
    /// which point presentation support can still be rejected by the backend.
    /// Vulkan requires these handles up front to enable presentation extensions.
    pub compatible_surface: Option<(DisplayHandle<'a>, WindowHandle<'a>)>,
}

/// Selected device capabilities relevant to the renderer.
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    /// Resource and pipeline limits; requests exceeding these are rejected.
    pub limits: crate::Limits,
    /// Whether nonzero depth bias clamp is supported.
    pub depth_bias_clamp: bool,
    /// Maximum supported texture anisotropy.
    pub max_sampler_anisotropy: u16,
    /// Minimum alignment for a uniform-buffer binding offset, in bytes.
    pub min_uniform_buffer_offset_alignment: u64,
    /// Minimum alignment for a storage-buffer binding offset, in bytes.
    pub min_storage_buffer_offset_alignment: u64,
    /// Required alignment of buffer offsets used for buffer/image copies.
    pub buffer_copy_offset_alignment: u64,
    /// Required alignment of buffer row pitches used for buffer/image copies.
    pub buffer_copy_row_pitch_alignment: u32,
    /// Whether a distinct compute queue is available.
    pub dedicated_compute_queue: bool,
    /// Whether a distinct copy queue is available.
    pub dedicated_copy_queue: bool,
}

/// Capabilities of one texture format on the selected device.
#[derive(Clone, Copy, Debug)]
pub struct FormatCapabilities {
    /// Whether linear texture filtering is supported.
    pub filterable: bool,
    /// Whether attachment blending is supported.
    pub blendable: bool,
    /// Exact region blit support, queried before command recording.
    pub blit: crate::BlitSupport,
    /// Image uses supported by the format.
    pub usages: ImageUsages,
}

impl FormatCapabilities {
    /// Returns whether all requested image usages are supported.
    #[must_use]
    pub const fn supports(self, usages: ImageUsages) -> bool {
        self.usages.contains(usages)
    }
}

/// CPU-waitable submission completion primitive.
///
/// # Safety
/// Implementations must preserve native object lifetimes and obey the shared
/// [`crate::Backend`] contract. Native operations are called with validated inputs.
pub unsafe trait NativeFence: Send + Sync + 'static {
    /// Waits until this fence signals or `timeout_ns` nanoseconds elapse.
    ///
    /// Returns `Ok(())` once the fence is signaled, including for a fence
    /// created signaled or one whose submission completed before the call.
    /// Returns [`Error::Timeout`](crate::Error::Timeout) when the timeout
    /// expires first; the fence may still signal afterwards.
    ///
    /// # Synchronization
    ///
    /// Implementations must serialize native host access so concurrent safe
    /// calls cannot violate backend external-synchronization rules.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn wait(&self, timeout_ns: u64) -> Result<()>;
    /// Resets this signaled fence for reuse.
    ///
    /// # Synchronization
    ///
    /// Implementations must return
    /// [`InvalidResourceKind::BadState`](crate::InvalidResourceKind::BadState)
    /// if a signaling submission has not completed, and must serialize reset
    /// against concurrent safe host operations.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn reset(&self) -> Result<()>;
}

/// Monotonically increasing GPU synchronization primitive.
///
/// # Safety
/// Implementations must preserve native object lifetimes and obey the shared
/// [`crate::Backend`] contract. Native operations are called with validated inputs.
pub unsafe trait NativeTimelineSemaphore: Clone + Send + Sync + 'static {
    /// Waits until this semaphore reaches `value` or `timeout_ns`
    /// nanoseconds elapse.
    ///
    /// Returns `Ok(())` once the semaphore's payload is at least `value`,
    /// including when it already exceeds it. Returns
    /// [`Error::Timeout`](crate::Error::Timeout) when the timeout expires
    /// first; the semaphore may still reach `value` afterwards.
    ///
    /// # Synchronization
    ///
    /// Implementations must serialize host operations where required by the
    /// native backend. Cloned handles alias the same synchronization state and
    /// therefore share that serialization.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn wait(&self, value: u64, timeout_ns: u64) -> Result<()>;
    /// Returns this semaphore's current value.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn value(&self) -> Result<u64>;
}

/// One queue submission, including presentation and timeline dependencies.
///
/// # Frame presentation protocol
///
/// Frames acquired from a swapchain are coupled to rendering through
/// `surface_frames`. For each listed frame the backend:
///
/// 1. waits, before execution begins, on the semaphore produced by
///    [`Swapchain::acquire`] for that frame;
/// 2. signals, once all recorded work completes, the semaphore consumed by
///    [`Swapchain::present`] for that frame.
///
/// A typical frame therefore acquires a frame, records into its image and
/// view, lists it in `surface_frames`, submits, and finally presents it.
/// Each frame must be submitted exactly once between acquisition and
/// presentation; see [`Swapchain`] for the full lifecycle and pacing rules.
pub struct Submission<'a, B: Api> {
    /// Recorded command buffers.
    pub command_buffers: &'a [&'a B::CommandBuffer],
    /// Acquired frames whose presentation dependencies are handled by this
    /// submission. See the type-level documentation for the protocol.
    pub surface_frames: &'a [&'a B::SurfaceFrame],
    /// Timeline values waited on before execution.
    pub wait_timelines: &'a [TimelinePoint<'a, B>],
    /// Timeline values signaled after execution.
    pub signal_timelines: &'a [TimelinePoint<'a, B>],
    /// Optional fence signaled after all submitted work completes.
    ///
    /// A timeline-only submission may leave this unset. When provided, the
    /// backend may retain submitted resources through this fence, so it must
    /// not be reused until [`Fence::wait`] succeeds.
    pub fence: Option<&'a B::Fence>,
}

/// Resource family used by borrowed portable descriptors. Each backend supplies
/// its own native types; public aliases select the active backend's safe types.
pub trait Api: Sized + Send + Sync + 'static {
    /// Buffer resource.
    type Buffer: Debug + Send + Sync + 'static;
    /// Image resource, including externally-owned surface images.
    type Image: Debug + Send + Sync + 'static;
    /// Image view resource.
    type ImageView: Debug + Send + Sync + 'static;
    /// Texture sampler resource.
    type Sampler: Debug + Send + Sync + 'static;
    /// Shader module resource.
    type Shader: Send + Sync + 'static;
    /// Bind-group layout resource.
    type BindGroupLayout: Send + Sync + 'static;
    /// Bound resource group.
    type BindGroup: Send + Sync + 'static;
    /// Pipeline layout resource.
    type PipelineLayout: Send + Sync + 'static;
    /// Graphics pipeline resource.
    type GraphicsPipeline: Send + Sync + 'static;
    /// Command allocation pool.
    type CommandPool: Send + 'static;
    /// Recorded command buffer.
    type CommandBuffer: Send + 'static;
    /// Submission completion fence.
    type Fence: Send + Sync + 'static;
    /// Timeline synchronization primitive.
    type TimelineSemaphore: Send + Sync + 'static;
    /// Presentation surface.
    /// Implementations must retain the [`crate::SurfaceTarget`] supplied at
    /// creation until the last surface handle is dropped.
    type Surface: Send + Sync + 'static;
    /// Presentation swapchain.
    type Swapchain: Send + 'static;
    /// Acquired presentation frame.
    type SurfaceFrame: Send + Sync + 'static;
}

/// Native operations underlying the safe [`crate::Rhi`].
///
/// # Safety
/// Implementations must preserve descriptor semantics and native resource ownership.
/// Native operations require valid descriptors, resources from this device, correct
/// recording state, dependencies, and external host synchronization. The safe RHI
/// establishes these obligations; native handles must not escape into safe callers.
pub unsafe trait Backend:
    Api<
        Buffer: NativeBuffer,
        CommandBuffer: NativeCommandBuffer<Self>,
        Fence: NativeFence,
        TimelineSemaphore: NativeTimelineSemaphore,
        Swapchain: NativeSwapchain<Self>,
        SurfaceFrame: NativeSurfaceFrame<Self>,
    >
{
    /// Creates a backend and selects its physical device.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn new(info: &RhiCreateInfo<'_>) -> Result<Self>;
    /// Returns selected device capabilities.
    fn capabilities(&self) -> Capabilities;
    /// Number of errors observed by native validation since device creation.
    /// Backends without a diagnostic counter return zero.
    fn validation_error_count(&self) -> usize {
        0
    }
    /// Returns the depth attachment formats supported by the selected
    /// device, ordered from most to least preferred.
    fn supported_depth_formats(&self) -> &[TextureFormat];
    /// Returns selected-device support for one texture format.
    fn format_capabilities(&self, format: TextureFormat) -> FormatCapabilities;
    /// Returns sample counts supported by `format` for all requested `usages`.
    fn supported_sample_counts(&self, format: TextureFormat, usages: ImageUsages) -> SampleCounts;
    /// Waits until all submitted device work completes.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn wait_idle(&self) -> Result<()>;
    /// Seals the current retirement bucket after its submissions.
    fn seal_garbage(&self);
    /// Reclaims the oldest sealed bucket after every associated queue has completed.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn collect_garbage(&self) -> Result<()>;

    /// Creates a buffer.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<Self::Buffer>;
    /// Creates an image.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_image(&self, desc: &ImageDesc<'_>) -> Result<Self::Image>;
    /// Creates a view of an image.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<Self::ImageView>;
    /// Creates a sampler.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<Self::Sampler>;
    /// Creates a shader module.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_shader(&self, desc: &ShaderDesc<'_>) -> Result<Self::Shader>;
    /// Creates a bind-group layout.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> Result<Self::BindGroupLayout>;
    /// Creates a bound resource group.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<Self::BindGroup>;
    /// Creates a pipeline layout.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<Self::PipelineLayout>;
    /// Creates a graphics pipeline.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<Self::GraphicsPipeline>;

    /// Creates a command pool for a semantic queue.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_command_pool(&self, queue: QueueType) -> Result<Self::CommandPool>;
    /// Allocates a command buffer from a pool.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_command_buffer(
        &self,
        pool: &mut Self::CommandPool,
    ) -> Result<Self::CommandBuffer>;
    /// Creates a fence.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence>;
    /// Creates a timeline semaphore.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<Self::TimelineSemaphore>;
    /// Submits command buffers and synchronization to a queue.
    ///
    /// Implementations must serialize access to an aliased native queue,
    /// reject command buffers recorded for a different queue, reject duplicate
    /// or stale surface frames, and keep submitted native objects alive until
    /// execution completes.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn submit(&self, queue: QueueType, submission: &Submission<'_, Self>) -> Result<()>;

    /// Creates a presentation surface.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_surface(&self, info: SurfaceCreateInfo) -> Result<Self::Surface>;
    /// Creates a presentation swapchain.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<Self::Swapchain>;
}
