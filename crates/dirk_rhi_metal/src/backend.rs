use std::sync::Arc;

use dirk_rhi::{
    Backend, BindGroupDesc, BindGroupLayoutDesc, BufferDesc, Capabilities, GraphicsPipelineDesc,
    ImageDesc, ImageViewDesc, PipelineLayoutDesc, QueueType, Result, RhiCreateInfo, SamplerDesc,
    ShaderDesc, Submission, SurfaceCreateInfo, SwapchainDesc,
};
use metal::{CommandQueue, Device, MTLCommandBufferStatus};

use crate::{
    command::{MetalCommandBuffer, MetalCommandPool},
    presentation::{MetalSurface, MetalSurfaceFrame, MetalSwapchain},
    resource::{
        MetalBindGroup, MetalBindGroupLayout, MetalBuffer, MetalFence, MetalGraphicsPipeline,
        MetalImage, MetalImageView, MetalPipelineLayout, MetalSampler, MetalShader,
        MetalTimelineSemaphore, require_context,
    },
};

pub(crate) struct Context {
    pub device: Device,
    queue: CommandQueue,
}

impl Context {
    pub(crate) fn queue(&self, _queue: QueueType) -> &CommandQueue {
        &self.queue
    }
}

/// Native Metal implementation of the [`Backend`] contract.
pub struct MetalBackend {
    context: Arc<Context>,
    capabilities: Capabilities,
}

impl Backend for MetalBackend {
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

    fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        let device = Device::system_default().ok_or(dirk_rhi::Error::NoDevice)?;
        let max_samples = [
            dirk_rhi::SampleCount::Eight,
            dirk_rhi::SampleCount::Four,
            dirk_rhi::SampleCount::Two,
        ]
        .into_iter()
        .find(|samples| device.supports_texture_sample_count(crate::convert::samples(*samples)))
        .unwrap_or(dirk_rhi::SampleCount::One);
        let queue = device.new_command_queue();
        queue.set_label(&format!("{} graphics queue", info.application_name));
        let context = Arc::new(Context { device, queue });
        Ok(Self {
            context,
            capabilities: Capabilities {
                max_samples,
                max_sampler_anisotropy: 16,
                min_uniform_buffer_offset_alignment: 256,
                min_storage_buffer_offset_alignment: 16,
                dedicated_compute_queue: false,
                dedicated_copy_queue: false,
            },
        })
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    fn supported_depth_formats(&self) -> &'static [dirk_rhi::TextureFormat] {
        // Apple GPUs prefer combined float depth/stencil; Intel Macs also
        // accept packed 24-bit formats, which map to the float pair here.
        &[dirk_rhi::TextureFormat::Depth32Float]
    }

    fn wait_idle(&self) -> Result<()> {
        let command = self.context.queue.new_command_buffer();
        command.commit();
        command.wait_until_completed();
        command_result(command)
    }

    fn collect_garbage(&self) -> Result<()> {
        // Metal resources are reference counted and command buffers retain
        // objects used by their encoders until GPU completion.
        Ok(())
    }

    fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<MetalBuffer> {
        MetalBuffer::create(&self.context, desc)
    }

    fn create_image(&self, desc: &ImageDesc<'_>) -> Result<MetalImage> {
        MetalImage::create(&self.context, desc)
    }

    fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<MetalImageView> {
        MetalImageView::create(&self.context, desc)
    }

    fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<MetalSampler> {
        Ok(MetalSampler::create(&self.context, desc))
    }

    fn create_shader(&self, desc: &ShaderDesc<'_>) -> Result<MetalShader> {
        MetalShader::create(&self.context, desc)
    }

    fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> Result<MetalBindGroupLayout> {
        MetalBindGroupLayout::create(&self.context, desc)
    }

    fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<MetalBindGroup> {
        MetalBindGroup::create(&self.context, desc)
    }

    fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<MetalPipelineLayout> {
        MetalPipelineLayout::create(&self.context, desc)
    }

    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<MetalGraphicsPipeline> {
        MetalGraphicsPipeline::create(&self.context, desc)
    }

    fn create_command_pool(&self, queue: QueueType) -> Result<MetalCommandPool> {
        Ok(MetalCommandPool {
            context: self.context.clone(),
            queue,
        })
    }

    fn create_command_buffer(&self, pool: &mut MetalCommandPool) -> Result<MetalCommandBuffer> {
        require_context(&self.context, &pool.context)?;
        Ok(MetalCommandBuffer::create(pool))
    }

    fn create_fence(&self, signaled: bool) -> Result<MetalFence> {
        Ok(MetalFence::create(&self.context, signaled))
    }

    fn create_timeline_semaphore(&self, initial_value: u64) -> Result<MetalTimelineSemaphore> {
        let event = self.context.device.new_shared_event();
        event.set_signaled_value(initial_value);
        Ok(MetalTimelineSemaphore {
            context: self.context.clone(),
            event,
        })
    }

    fn submit(&self, queue: QueueType, submission: &Submission<'_, Self>) -> Result<()> {
        if let Some(fence) = submission.fence {
            require_context(&self.context, &fence.context)?;
        }
        let mut commands = submission
            .command_buffers
            .iter()
            .map(|command| {
                require_context(&self.context, &command.context)?;
                if command.queue != queue {
                    return Err(dirk_rhi::InvalidResource::Mismatch.into());
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
        let first = &commands[0];
        for point in submission.wait_timelines {
            first.encode_wait_for_event(&point.semaphore.event, point.value);
        }
        let last = &commands[commands.len() - 1];
        for frame in submission.surface_frames {
            last.present_drawable(&frame.drawable);
            frame.mark_submitted();
        }
        for point in submission.signal_timelines {
            last.encode_signal_event(&point.semaphore.event, point.value);
        }
        if let Some(fence) = submission.fence {
            last.encode_signal_event(&fence.event, fence.value());
        }
        for command in commands {
            command.commit();
        }
        Ok(())
    }

    fn create_surface(&self, info: SurfaceCreateInfo) -> Result<MetalSurface> {
        MetalSurface::create(&self.context, info)
    }

    fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<MetalSwapchain> {
        MetalSwapchain::create(&self.context, desc)
    }
}

fn command_result(command: &metal::CommandBufferRef) -> Result<()> {
    match command.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        MTLCommandBufferStatus::Error => Err(dirk_rhi::Error::DeviceLost),
        status => Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
            "Metal command buffer completed with unexpected status {status:?}"
        ))),
    }
}
