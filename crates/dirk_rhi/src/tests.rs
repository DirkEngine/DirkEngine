use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};

use crate::{
    Backend, BindGroupDesc, BindGroupLayoutDesc, Buffer, BufferBarrier, BufferCopy, BufferDesc,
    BufferImageCopy, BufferUsages, Capabilities, Color, ColorSpace, CommandBuffer, DependencyInfo,
    Error, Extent3d, Fence, FilterMode, FormatCapabilities, GraphicsPipelineDesc, ImageCopy,
    ImageDesc, ImageUsages, ImageViewDesc, InvalidResourceKind, PipelineLayoutDesc, PipelineStages,
    QueueType, Rect, RenderingInfo, Result, RhiCreateInfo, SampleCount, SampleCounts, SamplerDesc,
    ShaderDesc, StencilOp, Submission, SurfaceCreateInfo, SurfaceFormat, SurfaceFrame,
    SurfaceStatus, Swapchain, SwapchainDesc, TextureFormat, TimelinePoint, TimelineSemaphore,
    UnsupportedOperation, Viewport,
};

#[derive(Clone, Debug)]
struct TestBuffer {
    data: Arc<Mutex<Vec<u8>>>,
}

impl TestBuffer {
    fn new(size: u64) -> Self {
        let size = usize::try_from(size).expect("test buffer size fits usize");
        Self {
            data: Arc::new(Mutex::new(vec![0; size])),
        }
    }

    fn checked_range(&self, offset: u64, length: usize) -> Result<std::ops::Range<usize>> {
        let start = usize::try_from(offset).map_err(|error| Error::Backend(error.into()))?;
        let end = start.checked_add(length).ok_or_else(|| {
            InvalidResourceKind::OutOfRange.with_detail("test buffer range overflowed")
        })?;
        if end
            > self
                .data
                .lock()
                .map_err(|_| Error::Backend(anyhow::anyhow!("test buffer mutex was poisoned")))?
                .len()
        {
            return Err(InvalidResourceKind::OutOfRange
                .with_detail(format!("test buffer range {start}..{end} exceeds its size"))
                .into());
        }
        Ok(start..end)
    }
}

impl Default for TestBuffer {
    fn default() -> Self {
        Self::new(0)
    }
}

unsafe impl Buffer for TestBuffer {
    fn size(&self) -> u64 {
        u64::try_from(self.data.lock().map_or(0, |data| data.len())).unwrap_or(u64::MAX)
    }

    unsafe fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let range = self.checked_range(offset, data.len())?;
        self.data
            .lock()
            .map_err(|_| Error::Backend(anyhow::anyhow!("test buffer mutex was poisoned")))?[range]
            .copy_from_slice(data);
        Ok(())
    }

    unsafe fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        let range = self.checked_range(offset, data.len())?;
        data.copy_from_slice(
            &self
                .data
                .lock()
                .map_err(|_| Error::Backend(anyhow::anyhow!("test buffer mutex was poisoned")))?
                [range],
        );
        Ok(())
    }
}

#[derive(Default)]
struct TestFence(AtomicBool);

unsafe impl Fence for TestFence {
    unsafe fn wait(&self, _timeout_ns: u64) -> Result<()> {
        if self.0.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Error::Timeout)
        }
    }

    unsafe fn reset(&self) -> Result<()> {
        self.0.store(false, Ordering::Release);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct TestTimeline(Arc<AtomicU64>);

unsafe impl TimelineSemaphore for TestTimeline {
    unsafe fn wait(&self, value: u64, _timeout_ns: u64) -> Result<()> {
        if self.0.load(Ordering::Acquire) >= value {
            Ok(())
        } else {
            Err(Error::Timeout)
        }
    }

    unsafe fn value(&self) -> Result<u64> {
        Ok(self.0.load(Ordering::Acquire))
    }
}

#[derive(Clone, Debug, Default)]
struct TestResource;

const TEST_SURFACE_FORMAT: SurfaceFormat = SurfaceFormat {
    texture: TextureFormat::Rgba8Unorm,
    color_space: ColorSpace::Srgb,
};

struct TestCommandPool(QueueType);

struct TestCommandBuffer(QueueType);

impl TestCommandBuffer {
    fn require_graphics(&self, operation: &str) -> Result<()> {
        if self.0 == QueueType::Graphics {
            Ok(())
        } else {
            Err(InvalidResourceKind::Mismatch
                .with_detail(format!(
                    "{operation} requires a graphics queue, not {:?}",
                    self.0
                ))
                .into())
        }
    }
}

#[derive(Default)]
struct TestSurfaceFrame {
    image: TestResource,
    view: TestResource,
    submitted: AtomicBool,
}

#[derive(Default)]
struct TestSwapchain;

#[derive(Default)]
struct TestBackend;

struct TestSurfaceTarget;

impl HasDisplayHandle for TestSurfaceTarget {
    fn display_handle(&self) -> std::result::Result<DisplayHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

impl HasWindowHandle for TestSurfaceTarget {
    fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

unsafe impl CommandBuffer<TestBackend> for TestCommandBuffer {
    fn queue_type(&self) -> QueueType {
        self.0
    }

    unsafe fn begin(&mut self, _label: &str, _one_time_submit: bool) -> Result<()> {
        Ok(())
    }

    unsafe fn end(&mut self) -> Result<()> {
        Ok(())
    }

    unsafe fn begin_rendering(&mut self, _info: &RenderingInfo<'_, TestBackend>) -> Result<()> {
        self.require_graphics("begin_rendering")
    }

    unsafe fn end_rendering(&mut self) -> Result<()> {
        self.require_graphics("end_rendering")
    }

    unsafe fn set_viewport(&mut self, _viewport: Viewport) -> Result<()> {
        self.require_graphics("set_viewport")
    }

    unsafe fn set_scissor(&mut self, _scissor: Rect) -> Result<()> {
        self.require_graphics("set_scissor")
    }

    unsafe fn set_blend_constants(&mut self, _color: Color) -> Result<()> {
        self.require_graphics("set_blend_constants")
    }

    unsafe fn set_stencil_reference(&mut self, _front: u32, _back: u32) -> Result<()> {
        self.require_graphics("set_stencil_reference")
    }

    unsafe fn bind_graphics_pipeline(&mut self, _pipeline: &TestResource) -> Result<()> {
        self.require_graphics("bind_graphics_pipeline")
    }

    unsafe fn bind_groups(
        &mut self,
        _layout: &TestResource,
        _first_group: u32,
        _groups: &[&TestResource],
        _dynamic_offsets: &[u64],
    ) -> Result<()> {
        self.require_graphics("bind_groups")
    }

    unsafe fn bind_vertex_buffer(
        &mut self,
        _slot: u32,
        _buffer: &TestBuffer,
        _offset: u64,
    ) -> Result<()> {
        self.require_graphics("bind_vertex_buffer")
    }

    unsafe fn bind_index_buffer(
        &mut self,
        _buffer: &TestBuffer,
        _offset: u64,
        _format: crate::IndexFormat,
    ) -> Result<()> {
        self.require_graphics("bind_index_buffer")
    }

    unsafe fn draw(
        &mut self,
        _vertex_count: u32,
        _instance_count: u32,
        _first_vertex: u32,
        _first_instance: u32,
    ) -> Result<()> {
        self.require_graphics("draw")
    }

    unsafe fn draw_indexed(
        &mut self,
        _index_count: u32,
        _instance_count: u32,
        _first_index: u32,
        _vertex_offset: i32,
        _first_instance: u32,
    ) -> Result<()> {
        self.require_graphics("draw_indexed")
    }

    unsafe fn copy_buffer(
        &mut self,
        _src: &TestBuffer,
        _dst: &TestBuffer,
        _regions: &[BufferCopy],
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn copy_buffer_to_image(
        &mut self,
        _src: &TestBuffer,
        _dst: &TestResource,
        _regions: &[BufferImageCopy],
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn copy_image_to_buffer(
        &mut self,
        _src: &TestResource,
        _dst: &TestBuffer,
        _regions: &[BufferImageCopy],
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn copy_image(
        &mut self,
        _src: &TestResource,
        _dst: &TestResource,
        _regions: &[ImageCopy],
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn blit_image(
        &mut self,
        _src: &TestResource,
        _dst: &TestResource,
        _regions: &[crate::ImageBlit],
        _filter: FilterMode,
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn barrier(&mut self, _dependency: &DependencyInfo<'_, TestBackend>) -> Result<()> {
        Ok(())
    }
}

unsafe impl SurfaceFrame<TestBackend> for TestSurfaceFrame {
    fn image(&self) -> &TestResource {
        &self.image
    }

    fn view(&self) -> &TestResource {
        &self.view
    }

    fn format(&self) -> SurfaceFormat {
        TEST_SURFACE_FORMAT
    }

    fn extent(&self) -> Extent3d {
        Extent3d::new_2d(1, 1)
    }

    fn status(&self) -> SurfaceStatus {
        SurfaceStatus::Optimal
    }
}

unsafe impl Swapchain<TestBackend> for TestSwapchain {
    fn format(&self) -> SurfaceFormat {
        TEST_SURFACE_FORMAT
    }

    fn extent(&self) -> Extent3d {
        Extent3d::new_2d(1, 1)
    }

    fn image_count(&self) -> std::num::NonZeroU32 {
        std::num::NonZeroU32::new(2).expect("test swapchain image count is nonzero")
    }

    unsafe fn acquire(&mut self, timeout_ns: u64) -> Result<TestSurfaceFrame> {
        if timeout_ns == 0 {
            Err(Error::Timeout)
        } else {
            Ok(TestSurfaceFrame::default())
        }
    }

    unsafe fn discard(&mut self, frame: TestSurfaceFrame) -> Result<()> {
        if frame.submitted.load(Ordering::Acquire) {
            Err(InvalidResourceKind::BadState
                .with_detail("submitted surface frame cannot be discarded")
                .into())
        } else {
            Ok(())
        }
    }

    unsafe fn resize(
        &mut self,
        _width: std::num::NonZeroU32,
        _height: std::num::NonZeroU32,
    ) -> Result<()> {
        Ok(())
    }

    unsafe fn present(&mut self, frame: TestSurfaceFrame) -> Result<SurfaceStatus> {
        if frame.submitted.load(Ordering::Acquire) {
            Ok(frame.status())
        } else {
            Err(InvalidResourceKind::BadState
                .with_detail("surface frame was presented before submission")
                .into())
        }
    }
}

impl crate::Api for TestBackend {
    type Buffer = TestBuffer;
    type Image = TestResource;
    type ImageView = TestResource;
    type Sampler = TestResource;
    type Shader = TestResource;
    type BindGroupLayout = TestResource;
    type BindGroup = TestResource;
    type PipelineLayout = TestResource;
    type GraphicsPipeline = TestResource;
    type CommandPool = TestCommandPool;
    type CommandBuffer = TestCommandBuffer;
    type Fence = TestFence;
    type TimelineSemaphore = TestTimeline;
    type Surface = TestResource;
    type Swapchain = TestSwapchain;
    type SurfaceFrame = TestSurfaceFrame;
}

unsafe impl Backend for TestBackend {
    unsafe fn new(_info: &RhiCreateInfo<'_>) -> Result<Self> {
        Ok(Self)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            limits: crate::Limits::default(),
            depth_bias_clamp: false,
            max_sampler_anisotropy: 1,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 16,
            buffer_copy_offset_alignment: 512,
            buffer_copy_row_pitch_alignment: 256,
            dedicated_compute_queue: false,
            dedicated_copy_queue: false,
        }
    }

    fn supported_depth_formats(&self) -> &[TextureFormat] {
        &[TextureFormat::Depth32Float, TextureFormat::Depth16Unorm]
    }

    fn format_capabilities(&self, _format: TextureFormat) -> FormatCapabilities {
        FormatCapabilities {
            filterable: true,
            blendable: true,
            blit: crate::BlitSupport::Linear,
            usages: ImageUsages::ALL,
        }
    }

    fn supported_sample_counts(
        &self,
        _format: TextureFormat,
        _usages: ImageUsages,
    ) -> SampleCounts {
        SampleCounts::ALL
    }

    unsafe fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    unsafe fn collect_garbage(&self) -> Result<()> {
        Ok(())
    }

    unsafe fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<TestBuffer> {
        if desc.size == 0 {
            return Err(InvalidResourceKind::Empty
                .with_detail("buffer size must be nonzero")
                .into());
        }
        Ok(TestBuffer::new(desc.size))
    }

    unsafe fn create_image(&self, _desc: &ImageDesc<'_>) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_image_view(&self, _desc: &ImageViewDesc<'_, Self>) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_sampler(&self, _desc: &SamplerDesc<'_>) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_shader(&self, _desc: &ShaderDesc<'_>) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_bind_group_layout(
        &self,
        _desc: &BindGroupLayoutDesc<'_>,
    ) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_bind_group(&self, _desc: &BindGroupDesc<'_, Self>) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_pipeline_layout(
        &self,
        _desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_graphics_pipeline(
        &self,
        _desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_command_pool(&self, queue: QueueType) -> Result<TestCommandPool> {
        Ok(TestCommandPool(queue))
    }

    unsafe fn create_command_buffer(
        &self,
        pool: &mut TestCommandPool,
    ) -> Result<TestCommandBuffer> {
        Ok(TestCommandBuffer(pool.0))
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<TestFence> {
        Ok(TestFence(AtomicBool::new(signaled)))
    }

    unsafe fn create_timeline_semaphore(&self, initial_value: u64) -> Result<TestTimeline> {
        Ok(TestTimeline(Arc::new(AtomicU64::new(initial_value))))
    }

    unsafe fn submit(&self, queue: QueueType, submission: &Submission<'_, Self>) -> Result<()> {
        if submission
            .command_buffers
            .iter()
            .any(|command| command.queue_type() != queue)
        {
            return Err(InvalidResourceKind::Mismatch
                .with_detail("command buffer queue does not match submission queue")
                .into());
        }
        for frame in submission.surface_frames {
            frame
                .submitted
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| {
                    InvalidResourceKind::BadState
                        .with_detail("surface frame was submitted more than once")
                })?;
        }
        if let Some(fence) = submission.fence {
            fence.0.store(true, Ordering::Release);
        }
        for point in submission.signal_timelines {
            point.semaphore.0.store(point.value, Ordering::Release);
        }
        Ok(())
    }

    unsafe fn create_surface(&self, _info: SurfaceCreateInfo) -> Result<TestResource> {
        Ok(TestResource)
    }

    unsafe fn create_swapchain(&self, _desc: &SwapchainDesc<'_, Self>) -> Result<TestSwapchain> {
        Ok(TestSwapchain)
    }
}

#[test]
fn usage_flags_compose_without_backend_values() {
    const DEFAULT: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::VERTEX);
    let usage = BufferUsages::COPY_DST | BufferUsages::VERTEX;

    assert_eq!(usage, DEFAULT);
    assert!(usage.contains(BufferUsages::COPY_DST));
    assert!(usage.contains(BufferUsages::VERTEX));
    assert!(!usage.contains(BufferUsages::UNIFORM));
    assert!(BufferUsages::NONE.is_empty());
    assert!(BufferUsages::from_bits(usage.bits()).is_some());
    assert!(BufferUsages::from_bits(u32::MAX).is_none());
}

#[test]
fn image_usage_flags_preserve_all_requested_roles() {
    let usage = ImageUsages::SAMPLED | ImageUsages::COLOR_ATTACHMENT | ImageUsages::COPY_SRC;

    assert!(usage.contains(ImageUsages::SAMPLED | ImageUsages::COPY_SRC));
    assert!(usage.contains(ImageUsages::COLOR_ATTACHMENT));
}

#[test]
fn semantic_types_do_not_encode_backend_constants() {
    assert_eq!(Extent3d::new_2d(1920, 1080).depth, 1);
    assert_eq!(SampleCount::Four as u8, 4);
    assert_eq!(BufferBarrier::<TestBackend>::REMAINING_SIZE, u64::MAX);
    assert_eq!(
        PipelineStages::ALL,
        PipelineStages::INDIRECT
            | PipelineStages::VERTEX_INPUT
            | PipelineStages::VERTEX_SHADER
            | PipelineStages::EARLY_DEPTH_STENCIL
            | PipelineStages::FRAGMENT_SHADER
            | PipelineStages::LATE_DEPTH_STENCIL
            | PipelineStages::COLOR_OUTPUT
            | PipelineStages::COMPUTE_SHADER
            | PipelineStages::COPY
            | PipelineStages::HOST
    );
    assert!(SampleCount::Four < SampleCount::Eight);
    assert_eq!(TextureFormat::Rgba16Float.texel_size(), 8);
    assert!(
        SampleCounts::ONE
            .union(SampleCounts::FOUR)
            .supports(SampleCount::Four)
    );
    assert!(!SampleCounts::ONE.supports(SampleCount::Two));
    assert!(
        TestBackend
            .format_capabilities(TextureFormat::Rgba8Unorm)
            .supports(ImageUsages::SAMPLED | ImageUsages::COPY_DST)
    );
}

#[test]
fn surface_create_info_keeps_its_target_alive() {
    let target = Arc::new(TestSurfaceTarget);
    let weak = Arc::downgrade(&target);
    let info = SurfaceCreateInfo::new(target.clone());

    drop(target);
    assert!(weak.upgrade().is_some());
    assert!(matches!(
        info.window_handle(),
        Err(HandleError::Unavailable)
    ));
    drop(info);
    assert!(weak.upgrade().is_none());
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "constructors must reproduce these constants exactly"
)]
fn semantic_helpers_build_expected_values() {
    let viewport = Viewport::dimensions(0.0, 0.0, 1280.0, 720.0);
    assert_eq!(viewport.min_depth, 0.0);
    assert_eq!(viewport.max_depth, 1.0);

    let scissor = Rect::new(4, 8, 640, 360);
    assert_eq!((scissor.x, scissor.y), (4, 8));

    assert_eq!(Color::TRANSPARENT.a, 0.0);
    assert_eq!(Color::BLACK, Color::new(0.0, 0.0, 0.0, 1.0));
    assert_eq!(
        Color::WHITE,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    );
}

#[test]
fn resources_own_their_stateful_operations() -> Result<()> {
    // SAFETY: these tests use the in-memory native mock with owned test inputs.
    unsafe {
        let buffer = TestBuffer::new(12);
        assert_eq!(buffer.size(), 12);
        buffer.write(8, &[1, 2, 3, 4])?;
        assert!(matches!(
            buffer.write(9, &[1, 2, 3, 4]),
            Err(Error::InvalidResource(error))
                if error.kind() == InvalidResourceKind::OutOfRange
        ));
        let mut bytes = [0; 4];
        buffer.read(8, &mut bytes)?;
        assert_eq!(bytes, [1, 2, 3, 4]);

        let fence = TestFence::default();
        assert!(matches!(fence.wait(0), Err(Error::Timeout)));
        fence.0.store(true, Ordering::Release);
        fence.wait(u64::MAX)?;
        assert!(fence.0.load(Ordering::Acquire));
        fence.reset()?;
        assert!(!fence.0.load(Ordering::Acquire));
        assert!(matches!(fence.wait(u64::MAX), Err(Error::Timeout)));

        let timeline = TestTimeline::default();
        assert!(matches!(timeline.wait(42, 0), Err(Error::Timeout)));
        timeline.0.store(42, Ordering::Release);
        timeline.wait(42, u64::MAX)?;
        assert_eq!(timeline.value()?, 42);
        assert!(matches!(timeline.wait(43, u64::MAX), Err(Error::Timeout)));
        Ok(())
    }
}

#[test]
fn backend_errors_keep_anyhow_context() {
    let source = anyhow::anyhow!("native allocation failed").context("creating buffer");
    let error = Error::from(source);

    assert!(matches!(error, Error::Backend(_)));
    assert_eq!(error.to_string(), "graphics backend error: creating buffer");

    let Error::Backend(inner) = &error else {
        unreachable!("classified as a backend error above");
    };
    let causes: Vec<_> = inner.chain().map(ToString::to_string).collect();
    assert_eq!(causes, ["creating buffer", "native allocation failed"]);
}

#[test]
fn typed_errors_describe_recoverable_conditions() {
    assert_eq!(
        Error::from(UnsupportedOperation::ImageBlit).to_string(),
        "unsupported RHI operation: image blits are not supported by this backend"
    );
    let language_error = Error::from(UnsupportedOperation::ShaderSource(
        crate::ShaderLanguage::Msl,
    ));
    assert_eq!(
        language_error.to_string(),
        "unsupported RHI operation: Metal Shading Language shader source is not supported by this backend"
    );
    assert_eq!(
        Error::from(InvalidResourceKind::ForeignInstance.with_detail("buffer came from backend B"))
            .to_string(),
        "invalid RHI request: resource belongs to a different RHI instance: buffer came from backend B"
    );
}

#[test]
fn shader_sources_report_their_language() {
    assert_eq!(
        crate::ShaderSource::SpirV(&[0; 4]).language(),
        crate::ShaderLanguage::SpirV
    );
    assert_eq!(
        crate::ShaderSource::Msl("void main() {}").language(),
        crate::ShaderLanguage::Msl
    );
}

#[test]
#[allow(clippy::float_cmp, reason = "default bias must be exactly zero")]
fn stencil_state_defaults_to_keep_operations() {
    let face = crate::StencilFaceState {
        compare: crate::CompareOp::Always,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::default(),
        pass_op: StencilOp::Replace,
    };
    assert_eq!(face.depth_fail_op, StencilOp::Keep);
    assert_eq!(crate::DepthBiasState::default().constant_factor, 0.0);
}

#[test]
fn backend_contract_accepts_borrowed_descriptors_and_submission() -> Result<()> {
    // SAFETY: these tests use the in-memory native mock with owned test inputs.
    unsafe {
        let backend = TestBackend::new(&RhiCreateInfo {
            engine_name: "test",
            engine_version: (0, 1, 0),
            application_name: "test",
            application_version: (0, 1, 0),
            validation: false,
            compatible_surface: None,
        })?;
        let mut pool = backend.create_command_pool(QueueType::Graphics)?;
        let command_buffer = backend.create_command_buffer(&mut pool)?;
        let command_buffers = [&command_buffer];
        let surface_frames: &[&TestSurfaceFrame] = &[];
        let wait_timelines: &[TimelinePoint<'_, TestBackend>] = &[];
        let signal_timelines: &[TimelinePoint<'_, TestBackend>] = &[];
        let submission = Submission {
            command_buffers: &command_buffers,
            surface_frames,
            wait_timelines,
            signal_timelines,
            fence: None,
        };

        backend.submit(QueueType::Graphics, &submission)
    }
}

#[test]
fn command_buffers_report_incompatible_queue_commands() -> Result<()> {
    // SAFETY: the in-memory mock deliberately supports validation of invalid calls.
    unsafe {
        let backend = TestBackend;
        let mut pool = backend.create_command_pool(QueueType::Copy)?;
        let mut command = backend.create_command_buffer(&mut pool)?;

        let error = command
            .draw(3, 1, 0, 0)
            .expect_err("draws require a graphics command buffer");
        assert!(matches!(
            error,
            Error::InvalidResource(error) if error.kind() == InvalidResourceKind::Mismatch
        ));

        let src = TestBuffer::new(1);
        let dst = TestBuffer::new(1);
        command.copy_buffer(
            &src,
            &dst,
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 1,
            }],
        )?;
        Ok(())
    }
}

#[test]
fn surface_frames_reject_invalid_lifecycle_transitions() -> Result<()> {
    // SAFETY: the in-memory mock deliberately supports validation of invalid calls.
    unsafe {
        let backend = TestBackend;
        let mut swapchain = TestSwapchain;

        let unsubmitted = swapchain.acquire(u64::MAX)?;
        assert!(matches!(
            swapchain.present(unsubmitted),
            Err(Error::InvalidResource(error))
                if error.kind() == InvalidResourceKind::BadState
        ));

        let submitted = swapchain.acquire(u64::MAX)?;
        {
            let frames = [&submitted];
            let submission = Submission {
                command_buffers: &[],
                surface_frames: &frames,
                wait_timelines: &[],
                signal_timelines: &[],
                fence: None,
            };
            backend.submit(QueueType::Graphics, &submission)?;
            assert!(matches!(
                backend.submit(QueueType::Graphics, &submission),
                Err(Error::InvalidResource(error))
                    if error.kind() == InvalidResourceKind::BadState
            ));
        }
        assert!(matches!(
            swapchain.discard(submitted),
            Err(Error::InvalidResource(error))
                if error.kind() == InvalidResourceKind::BadState
        ));

        assert!(matches!(swapchain.acquire(0), Err(Error::Timeout)));
        assert_eq!(swapchain.image_count().get(), 2);
        Ok(())
    }
}

fn safe_device() -> Result<crate::Rhi<TestBackend>> {
    crate::Rhi::new(&RhiCreateInfo {
        engine_name: "test",
        engine_version: (0, 1, 0),
        application_name: "test",
        application_version: (0, 1, 0),
        validation: false,
        compatible_surface: None,
    })
}

#[test]
fn submitted_buffers_reject_host_access_until_completion() -> Result<()> {
    let device = safe_device()?;
    let src = device.create_buffer(&BufferDesc {
        label: "upload",
        size: 16,
        usage: BufferUsages::COPY_SRC,
        memory: crate::MemoryDomain::Upload,
    })?;
    let dst = device.create_buffer(&BufferDesc {
        label: "destination",
        size: 16,
        usage: BufferUsages::COPY_DST,
        memory: crate::MemoryDomain::Device,
    })?;
    src.write(0, &[1; 16])?;
    let alias = src.clone();
    let mut encoder = device.create_encoder::<crate::CopyQueue>("copy")?;
    encoder.copy_buffer(
        &src,
        &dst,
        &[BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: 16,
        }],
    )?;
    let completion = device
        .queue::<crate::CopyQueue>()
        .submit(vec![encoder.finish()?], &crate::SubmitInfo::default())?;
    assert!(
        matches!(alias.write(0,&[2;16]),Err(Error::InvalidResource(error)) if error.kind()==InvalidResourceKind::BadState)
    );
    drop(src);
    drop(dst);
    completion.wait(u64::MAX)?;
    alias.write(0, &[3; 16])?;
    Ok(())
}

#[test]
fn dropping_completion_keeps_submission_owned_by_device() -> Result<()> {
    let device = safe_device()?;
    let src = device.create_buffer(&BufferDesc {
        label: "upload",
        size: 16,
        usage: BufferUsages::COPY_SRC,
        memory: crate::MemoryDomain::Upload,
    })?;
    let dst = device.create_buffer(&BufferDesc {
        label: "destination",
        size: 16,
        usage: BufferUsages::COPY_DST,
        memory: crate::MemoryDomain::Device,
    })?;
    let mut encoder = device.create_encoder::<crate::Graphics>("copy")?;
    encoder.copy_buffer(
        &src,
        &dst,
        &[BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: 16,
        }],
    )?;
    drop(
        device
            .queue::<crate::Graphics>()
            .submit(vec![encoder.finish()?], &crate::SubmitInfo::default())?,
    );
    assert!(src.write(0, &[0; 16]).is_err());
    device.collect_garbage()?;
    src.write(0, &[0; 16])?;
    Ok(())
}

#[test]
fn safe_recording_rejects_foreign_resources_and_invalid_ranges() -> Result<()> {
    let a = safe_device()?;
    let b = safe_device()?;
    let src = a.create_buffer(&BufferDesc {
        label: "source",
        size: 16,
        usage: BufferUsages::COPY_SRC,
        memory: crate::MemoryDomain::Upload,
    })?;
    let dst = b.create_buffer(&BufferDesc {
        label: "foreign",
        size: 16,
        usage: BufferUsages::COPY_DST,
        memory: crate::MemoryDomain::Device,
    })?;
    let mut encoder = a.create_encoder::<crate::CopyQueue>("copy")?;
    assert!(
        matches!(encoder.copy_buffer(&src,&dst,&[BufferCopy{src_offset:0,dst_offset:0,size:16}]),Err(Error::InvalidResource(error)) if error.kind()==InvalidResourceKind::ForeignInstance)
    );
    assert!(src.write(15, &[1; 2]).is_err());
    Ok(())
}

#[test]
fn forgotten_render_pass_cannot_be_finished_or_submitted() -> Result<()> {
    let device = safe_device()?;
    let image = device.create_image(&ImageDesc {
        label: "target",
        dimension: crate::ImageDimension::TwoD,
        extent: Extent3d::new_2d(4, 4),
        format: TextureFormat::Rgba8Unorm,
        usage: ImageUsages::COLOR_ATTACHMENT,
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::One,
    })?;
    let view = device.view(&image)?;
    let mut encoder = device.create_encoder::<crate::Graphics>("pass")?;
    let attachments = [crate::ColorAttachment {
        view: &view,
        resolve: None,
        load: crate::LoadOp::Clear(Color::BLACK),
        store: crate::StoreOp::Store,
    }];
    let pass = encoder.begin_render_pass(&RenderingInfo {
        label: "pass",
        width: 4,
        height: 4,
        layer_count: 1,
        color_attachments: &attachments,
        depth_attachment: None,
    })?;
    std::mem::forget(pass);
    assert!(encoder.finish().is_err());
    Ok(())
}

#[test]
fn padded_upload_rows_preserve_pixels_and_zero_padding() -> Result<()> {
    let caps = safe_device()?.capabilities();
    let layout = crate::UploadLayout::new(3, 2, TextureFormat::Rgba8Unorm, caps)?;
    let pixels: Vec<u8> = (0..24).collect();
    let padded = layout.pack(&pixels)?;
    assert_eq!(padded.len(), 512);
    assert_eq!(&padded[..12], &pixels[..12]);
    assert_eq!(&padded[256..268], &pixels[12..]);
    assert!(
        padded[12..256]
            .iter()
            .chain(padded[268..].iter())
            .all(|byte| *byte == 0)
    );
    assert!(layout.pack(&pixels[..23]).is_err());
    Ok(())
}

#[test]
fn shader_binding_maps_agree_for_sparse_stage_specific_layouts() -> Result<()> {
    use crate::{BindGroupLayoutEntry as E, BindingType as T, ShaderStage, ShaderStages as S};
    let vertex_only = E {
        binding: 7,
        ty: T::UniformBuffer {
            dynamic_offset: false,
        },
        visibility: S::VERTEX,
    };
    let fragment_only = E {
        binding: 13,
        ty: T::SampledImage,
        visibility: S::FRAGMENT,
    };
    let shared = E {
        binding: 42,
        ty: T::StorageBuffer {
            read_only: true,
            dynamic_offset: false,
        },
        visibility: S::VERTEX | S::FRAGMENT,
    };
    let merged = [vertex_only, fragment_only, shared];
    let fragment = [fragment_only, shared];
    let pipeline = crate::BindingMap::new(&[&[], &merged], ShaderStage::Fragment)?;
    let shader = crate::BindingMap::new(&[&[], &fragment], ShaderStage::Fragment)?;
    assert_eq!(pipeline.get(1, 13), shader.get(1, 13));
    assert_eq!(pipeline.get(1, 42), shader.get(1, 42));
    assert_eq!(pipeline.get(1, 7), None);
    assert_eq!(pipeline.get(1, 42).and_then(|s| s.buffer), Some(0));
    Ok(())
}

#[test]
fn primitive_restart_requires_strip_topology() -> Result<()> {
    let device = safe_device()?;
    let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: "empty",
        bind_group_layouts: &[],
    })?;
    // The test backend does not execute shader code.
    let vertex = unsafe {
        device.create_shader(&ShaderDesc {
            label: "vertex",
            stage: crate::ShaderStage::Vertex,
            entry: "main",
            source: crate::ShaderSource::SpirV(&[]),
        })?
    };
    let mut desc = GraphicsPipelineDesc {
        label: "restart",
        layout: &layout,
        vertex: &vertex,
        fragment: None,
        vertex_buffers: &[],
        raster: crate::RasterState::default(),
        color_targets: &[],
        depth: None,
        depth_bias: crate::DepthBiasState::default(),
        primitive_restart: Some(crate::IndexFormat::Uint16),
        alpha_to_coverage: false,
        samples: SampleCount::One,
    };
    assert!(device.create_graphics_pipeline(&desc).is_err());
    desc.raster.topology = crate::PrimitiveTopology::LineList;
    assert!(device.create_graphics_pipeline(&desc).is_err());
    desc.raster.topology = crate::PrimitiveTopology::TriangleStrip;
    let pipeline = device.create_graphics_pipeline(&desc)?;
    assert_eq!(pipeline.info().primitive_restart, desc.primitive_restart);
    Ok(())
}
