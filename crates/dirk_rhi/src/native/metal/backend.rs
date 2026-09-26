use std::sync::Arc;

use crate::{
    Backend, BindGroupDesc, BindGroupLayoutDesc, BufferDesc, Capabilities, FormatCapabilities,
    GraphicsPipelineDesc, ImageDesc, ImageUsages, ImageViewDesc, PipelineLayoutDesc, QueueType,
    Result, RhiCreateInfo, SampleCount, SampleCounts, SamplerDesc, ShaderDesc, Submission,
    SurfaceCreateInfo, SwapchainDesc, TextureFormat,
};
use metal::{CommandQueue, Device, MTLCommandBufferStatus};

use super::{
    command::{MetalCommandBuffer, MetalCommandPool},
    presentation::{MetalSurface, MetalSurfaceFrame, MetalSwapchain},
    resource::{
        MetalBindGroup, MetalBindGroupLayout, MetalBuffer, MetalFence, MetalGraphicsPipeline,
        MetalImage, MetalImageView, MetalPipelineLayout, MetalSampler, MetalShader,
        MetalTimelineSemaphore, require_context,
    },
};

pub(crate) enum Garbage {
    Buffer(metal::Buffer),
    Texture(metal::Texture),
    Sampler(metal::SamplerState),
    Shader(metal::Function),
    Pipeline(metal::RenderPipelineState),
    Depth(metal::DepthStencilState),
    Surface(metal::MetalLayer, Arc<dyn crate::SurfaceTarget>),
}
impl Garbage {
    fn destroy(self) {
        match self {
            Self::Buffer(v) => drop(v),
            Self::Texture(v) => drop(v),
            Self::Sampler(v) => drop(v),
            Self::Shader(v) => drop(v),
            Self::Pipeline(v) => drop(v),
            Self::Depth(v) => drop(v),
            Self::Surface(layer, target) => {
                drop(layer);
                drop(target);
            }
        }
    }
}
pub(crate) struct Context {
    pub device: Device,
    queue: CommandQueue,
    garbage: parking_lot::Mutex<crate::retirement::RetirementQueue<Garbage>>,
}

impl Context {
    pub(crate) fn retire(&self, value: Garbage) {
        self.garbage.lock().push(value);
    }
    fn drain(&self) {
        for value in self.garbage.lock().drain() {
            value.destroy();
        }
    }

    pub(crate) fn queue(&self, _queue: QueueType) -> &CommandQueue {
        &self.queue
    }
}

/// Native Metal implementation of the [`Backend`] contract.
pub struct MetalBackend {
    context: Arc<Context>,
    capabilities: Capabilities,
}

impl crate::Api for MetalBackend {
    type Buffer = MetalBuffer;
    type Image = MetalImage;
    type ImageView = MetalImageView;
    type Sampler = MetalSampler;
    type Shader = MetalShader;
    type BindGroupLayout = MetalBindGroupLayout;
    type BindGroup = MetalBindGroup;
    type PipelineLayout = MetalPipelineLayout;
    type GraphicsPipeline = MetalGraphicsPipeline;
    type CommandPool = MetalCommandPool;
    type CommandBuffer = MetalCommandBuffer;
    type Fence = MetalFence;
    type TimelineSemaphore = MetalTimelineSemaphore;
    type Surface = MetalSurface;
    type Swapchain = MetalSwapchain;
    type SurfaceFrame = MetalSurfaceFrame;
}

// SAFETY: native resources own their parent objects; the shared RHI validates calls and retains GPU use.
unsafe impl Backend for MetalBackend {
    unsafe fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        metal::objc::rc::autoreleasepool(|| {
            let device = Device::system_default().ok_or(crate::Error::NoDevice)?;
            let queue = device.new_command_queue();
            queue.set_label(&format!("{} graphics queue", info.application_name));
            let max_buffer_size = device.max_buffer_length();
            let context = Arc::new(Context {
                device,
                queue,
                garbage: parking_lot::Mutex::new(crate::retirement::RetirementQueue::default()),
            });
            Ok(Self {
                context,
                capabilities: Capabilities {
                    limits: crate::Limits {
                        max_buffer_size,
                        max_uniform_buffer_binding_size: max_buffer_size,
                        max_storage_buffer_binding_size: max_buffer_size,
                        ..crate::Limits::default()
                    },
                    depth_bias_clamp: true,
                    max_sampler_anisotropy: 16,
                    min_uniform_buffer_offset_alignment: 256,
                    min_storage_buffer_offset_alignment: 16,
                    buffer_copy_offset_alignment: 4,
                    buffer_copy_row_pitch_alignment: 256,
                    dedicated_compute_queue: false,
                    dedicated_copy_queue: false,
                },
            })
        })
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn supported_depth_formats(&self) -> &'static [crate::TextureFormat] {
        &[
            TextureFormat::Depth32Float,
            TextureFormat::Depth16Unorm,
            TextureFormat::Depth32FloatStencil8,
        ]
    }

    fn format_capabilities(&self, format: TextureFormat) -> FormatCapabilities {
        // Conservative common support, with 32-bit filtering queried explicitly.
        // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
        let (depth, storage, filterable) = match format {
            TextureFormat::Rgba8Unorm
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::Rgba16Float => (false, true, true),
            TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8Srgb
            | TextureFormat::Rgba8Srgb
            | TextureFormat::R11G11B10Float => (false, false, true),
            TextureFormat::R32Float | TextureFormat::Rg32Float | TextureFormat::Rgba32Float => (
                false,
                true,
                self.context.device.supports_32bit_float_filtering(),
            ),
            TextureFormat::Depth16Unorm => (true, false, true),
            TextureFormat::Depth32Float | TextureFormat::Depth32FloatStencil8 => (
                true,
                false,
                self.context.device.supports_32bit_float_filtering(),
            ),
            _ => {
                return FormatCapabilities {
                    usages: ImageUsages::NONE,
                    filterable: false,
                    blendable: false,
                    blit: crate::BlitSupport::None,
                };
            }
        };
        let mut usages = ImageUsages::COPY_SRC | ImageUsages::COPY_DST | ImageUsages::SAMPLED;
        usages |= if depth {
            ImageUsages::DEPTH_STENCIL_ATTACHMENT
        } else {
            ImageUsages::COLOR_ATTACHMENT
        };
        if storage {
            usages |= ImageUsages::STORAGE;
        }
        FormatCapabilities {
            usages,
            filterable,
            blendable: !depth,
            blit: crate::BlitSupport::None,
        }
    }

    fn supported_sample_counts(&self, format: TextureFormat, usages: ImageUsages) -> SampleCounts {
        if !self.format_capabilities(format).usages.contains(usages) {
            return SampleCounts::default();
        }
        if matches!(
            format,
            TextureFormat::R32Float | TextureFormat::Rg32Float | TextureFormat::Rgba32Float
        ) && !self.context.device.supports_32bit_MSAA()
        {
            return SampleCounts::ONE;
        }
        let mut counts = SampleCounts::ONE;
        for (count, flag) in [
            (SampleCount::Two, SampleCounts::TWO),
            (SampleCount::Four, SampleCounts::FOUR),
            (SampleCount::Eight, SampleCounts::EIGHT),
        ] {
            if self
                .context
                .device
                .supports_texture_sample_count(super::convert::samples(count))
            {
                counts |= flag;
            }
        }
        counts
    }

    unsafe fn wait_idle(&self) -> Result<()> {
        metal::objc::rc::autoreleasepool(|| {
            let command = self.context.queue.new_command_buffer();
            command.commit();
            command.wait_until_completed();
            command_result(command)?;
            Ok(())
        })
    }

    fn seal_garbage(&self) {
        self.context.garbage.lock().seal();
    }

    unsafe fn collect_garbage(&self) -> Result<()> {
        for value in self.context.garbage.lock().collect() {
            value.destroy();
        }
        Ok(())
    }

    unsafe fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<MetalBuffer> {
        metal::objc::rc::autoreleasepool(|| MetalBuffer::create(&self.context, desc))
    }

    unsafe fn create_image(&self, desc: &ImageDesc<'_>) -> Result<MetalImage> {
        metal::objc::rc::autoreleasepool(|| MetalImage::create(&self.context, desc))
    }

    unsafe fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<MetalImageView> {
        metal::objc::rc::autoreleasepool(|| MetalImageView::create(&self.context, desc))
    }

    unsafe fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<MetalSampler> {
        metal::objc::rc::autoreleasepool(|| Ok(MetalSampler::create(&self.context, desc)))
    }

    unsafe fn create_shader(&self, desc: &ShaderDesc<'_>) -> Result<MetalShader> {
        metal::objc::rc::autoreleasepool(|| MetalShader::create(&self.context, desc))
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> Result<MetalBindGroupLayout> {
        MetalBindGroupLayout::create(&self.context, desc)
    }

    unsafe fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<MetalBindGroup> {
        MetalBindGroup::create(&self.context, desc)
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<MetalPipelineLayout> {
        MetalPipelineLayout::create(&self.context, desc)
    }

    unsafe fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<MetalGraphicsPipeline> {
        metal::objc::rc::autoreleasepool(|| MetalGraphicsPipeline::create(&self.context, desc))
    }

    unsafe fn create_command_pool(&self, queue: QueueType) -> Result<MetalCommandPool> {
        Ok(MetalCommandPool {
            context: self.context.clone(),
            queue,
        })
    }

    unsafe fn create_command_buffer(
        &self,
        pool: &mut MetalCommandPool,
    ) -> Result<MetalCommandBuffer> {
        require_context(&self.context, &pool.context)?;
        Ok(MetalCommandBuffer::create(pool))
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<MetalFence> {
        Ok(MetalFence::create(&self.context, signaled))
    }

    unsafe fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<MetalTimelineSemaphore> {
        let event = self.context.device.new_shared_event();
        event.set_signaled_value(initial_value);
        Ok(MetalTimelineSemaphore {
            context: self.context.clone(),
            event,
        })
    }

    unsafe fn submit(&self, queue: QueueType, submission: &Submission<'_, Self>) -> Result<()> {
        metal::objc::rc::autoreleasepool(|| {
            if let Some(fence) = submission.fence {
                require_context(&self.context, &fence.context)?;
            }
            let mut commands = submission
                .command_buffers
                .iter()
                .map(|command| {
                    require_context(&self.context, &command.context)?;
                    if command.queue != queue {
                        return Err(crate::InvalidResourceKind::Mismatch.into());
                    }
                    command.command_for_submit()
                })
                .collect::<Result<Vec<_>>>()?;
            if commands.is_empty() {
                commands.push(self.context.queue(queue).new_command_buffer().to_owned());
            }
            for point in submission
                .wait_timelines
                .iter()
                .chain(submission.signal_timelines)
            {
                require_context(&self.context, &point.semaphore.context)?;
            }
            for frame in submission.surface_frames {
                require_context(&self.context, &frame.context)?;
            }
            // Waits must precede recorded work, not be appended after its encoders.
            if !submission.wait_timelines.is_empty() {
                let prelude = self.context.queue(queue).new_command_buffer().to_owned();
                for point in submission.wait_timelines {
                    prelude.encode_wait_for_event(&point.semaphore.event, point.value);
                }
                commands.insert(0, prelude);
            }
            let last = &commands[commands.len() - 1];
            for frame in submission.surface_frames {
                frame.mark_submitted()?;
                last.present_drawable(&frame.drawable);
            }
            for point in submission.signal_timelines {
                last.encode_signal_event(&point.semaphore.event, point.value);
            }
            if let Some(fence) = submission.fence {
                fence.track(&commands);
            }
            for command in commands {
                command.commit();
            }
            Ok(())
        })
    }

    unsafe fn create_surface(&self, info: SurfaceCreateInfo) -> Result<MetalSurface> {
        metal::objc::rc::autoreleasepool(|| MetalSurface::create(&self.context, &info))
    }

    unsafe fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<MetalSwapchain> {
        metal::objc::rc::autoreleasepool(|| MetalSwapchain::create(&self.context, desc))
    }
}

pub(crate) fn command_result(command: &metal::CommandBufferRef) -> Result<()> {
    match command.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(crate::Error::DeviceLost),
        status => Err(crate::Error::Backend(anyhow::anyhow!(
            "Metal command buffer completed with unexpected status {status:?}"
        ))),
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        metal::objc::rc::autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            command.commit();
            command.wait_until_completed();
            self.drain();
        });
    }
}
