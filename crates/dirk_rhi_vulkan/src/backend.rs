use std::sync::Arc;

use ash::vk;
use dirk_rhi::{
    Backend, BindGroupDesc, BindGroupLayoutDesc, BufferDesc, Capabilities, CommandBuffer,
    FormatCapabilities, GraphicsPipelineDesc, ImageDesc, ImageUsages, ImageViewDesc,
    InvalidResourceKind as Ir, PipelineLayoutDesc, QueueType, Result, RhiCreateInfo, SampleCounts,
    SamplerDesc, ShaderDesc, Submission, SurfaceCreateInfo, SwapchainDesc, TextureFormat,
};

use crate::{
    command::{VulkanCommandBuffer, VulkanCommandPool},
    convert,
    device::Context,
    presentation::{VulkanSurface, VulkanSurfaceFrame, VulkanSwapchain},
    resource::{
        VulkanBindGroup, VulkanBindGroupLayout, VulkanBuffer, VulkanFence, VulkanGraphicsPipeline,
        VulkanImage, VulkanImageView, VulkanPipelineLayout, VulkanSampler, VulkanShader,
        VulkanTimelineSemaphore,
    },
    vk_error,
};

/// Vulkan 1.3 implementation of the [`Backend`] contract.
pub struct VulkanBackend {
    pub(crate) context: Arc<Context>,
}

impl VulkanBackend {
    fn require_context(&self, context: &Arc<Context>) -> Result<()> {
        if Arc::ptr_eq(&self.context, context) {
            Ok(())
        } else {
            Err(Ir::ForeignInstance
                .with_detail("resource belongs to another Vulkan device")
                .into())
        }
    }

    fn mark_surface_frames<'a>(
        frames: &'a [&'a VulkanSurfaceFrame],
    ) -> Result<Vec<&'a VulkanSurfaceFrame>> {
        let mut marked = Vec::with_capacity(frames.len());
        for frame in frames {
            if let Err(error) = frame.mark_submitted() {
                for frame in marked {
                    VulkanSurfaceFrame::unmark_submitted(frame);
                }
                return Err(error);
            }
            marked.push(*frame);
        }
        Ok(marked)
    }

    fn validate_submission<'a>(
        &self,
        queue: QueueType,
        submission: &'a Submission<'a, Self>,
    ) -> Result<Vec<&'a VulkanSurfaceFrame>> {
        if let Some(fence) = submission.fence {
            self.require_context(fence.context())?;
        }
        for command in submission.command_buffers {
            self.require_context(command.context())?;
        }
        for frame in submission.surface_frames {
            self.require_context(frame.context())?;
        }
        for point in submission
            .wait_timelines
            .iter()
            .chain(submission.signal_timelines)
        {
            self.require_context(point.semaphore.context())?;
        }
        if submission
            .command_buffers
            .iter()
            .any(|command| command.queue_type() != queue)
        {
            return Err(Ir::Mismatch
                .with_detail("command buffer was allocated for a different queue")
                .into());
        }
        Self::mark_surface_frames(submission.surface_frames)
    }

    /// Returns the loaded Vulkan entry.
    #[must_use]
    pub fn entry(&self) -> &ash::Entry {
        &self.context.entry
    }

    /// Returns the Vulkan instance.
    #[must_use]
    pub fn instance(&self) -> &ash::Instance {
        &self.context.instance
    }

    /// Returns the selected physical device.
    #[must_use]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.context.physical_device
    }

    /// Returns the logical Vulkan device.
    #[must_use]
    pub fn device(&self) -> &ash::Device {
        &self.context.device
    }

    /// Returns the native queue used for a semantic RHI queue.
    #[must_use]
    pub fn queue(&self, queue: QueueType) -> vk::Queue {
        self.context.queue(queue)
    }

    /// Returns the native queue-family index for a semantic RHI queue.
    #[must_use]
    pub fn queue_family(&self, queue: QueueType) -> u32 {
        self.context.queue_family(queue)
    }
}

impl Backend for VulkanBackend {
    type Buffer = VulkanBuffer;
    type Image = VulkanImage;
    type ImageView = VulkanImageView;
    type Sampler = VulkanSampler;
    type Shader = VulkanShader;
    type BindGroupLayout = VulkanBindGroupLayout;
    type BindGroup = VulkanBindGroup;
    type PipelineLayout = VulkanPipelineLayout;
    type GraphicsPipeline = VulkanGraphicsPipeline;
    type CommandPool = VulkanCommandPool;
    type CommandBuffer = VulkanCommandBuffer;
    type Fence = VulkanFence;
    type TimelineSemaphore = VulkanTimelineSemaphore;
    type Surface = VulkanSurface;
    type Swapchain = VulkanSwapchain;
    type SurfaceFrame = VulkanSurfaceFrame;

    fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        Ok(Self {
            context: Context::new(info)?,
        })
    }

    fn capabilities(&self) -> Capabilities {
        self.context.capabilities
    }

    fn supported_depth_formats(&self) -> &[TextureFormat] {
        self.context.supported_depth_formats
    }

    fn format_capabilities(&self, format: TextureFormat) -> FormatCapabilities {
        let properties = unsafe {
            self.context.instance.get_physical_device_format_properties(
                self.context.physical_device,
                convert::format(format),
            )
        };
        let features = properties.optimal_tiling_features;
        let mut usages = ImageUsages::COPY_SRC | ImageUsages::COPY_DST;
        if features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usages |= ImageUsages::SAMPLED;
        }
        if features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usages |= ImageUsages::STORAGE;
        }
        if features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usages |= ImageUsages::COLOR_ATTACHMENT;
        }
        if features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usages |= ImageUsages::DEPTH_STENCIL_ATTACHMENT;
        }
        FormatCapabilities { usages }
    }

    fn supported_sample_counts(&self, format: TextureFormat, usages: ImageUsages) -> SampleCounts {
        unsafe {
            self.context
                .instance
                .get_physical_device_image_format_properties(
                    self.context.physical_device,
                    convert::format(format),
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    convert::image_usage(usages),
                    vk::ImageCreateFlags::empty(),
                )
        }
        .map_or(SampleCounts::default(), |properties| {
            convert::rhi_sample_counts(properties.sample_counts)
        })
    }

    fn wait_idle(&self) -> Result<()> {
        unsafe { self.context.device.device_wait_idle() }.map_err(vk_error)?;
        self.context.collect_all_garbage();
        Ok(())
    }

    fn collect_garbage(&self) -> Result<()> {
        self.context.collect_garbage();
        Ok(())
    }

    fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<VulkanBuffer> {
        VulkanBuffer::create(&self.context, desc)
    }

    fn create_image(&self, desc: &ImageDesc<'_>) -> Result<VulkanImage> {
        VulkanImage::create(&self.context, desc)
    }

    fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<VulkanImageView> {
        if !Arc::ptr_eq(&self.context, desc.image.context()) {
            return Err(Ir::ForeignInstance
                .with_detail("image belongs to another Vulkan device")
                .into());
        }
        VulkanImageView::create(&self.context, desc)
    }

    fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<VulkanSampler> {
        VulkanSampler::create(&self.context, desc)
    }

    fn create_shader(&self, desc: &ShaderDesc<'_>) -> Result<VulkanShader> {
        VulkanShader::create(&self.context, desc)
    }

    fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDesc<'_>,
    ) -> Result<VulkanBindGroupLayout> {
        VulkanBindGroupLayout::create(&self.context, desc)
    }

    fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<VulkanBindGroup> {
        VulkanBindGroup::create(&self.context, desc)
    }

    fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDesc<'_, Self>,
    ) -> Result<VulkanPipelineLayout> {
        VulkanPipelineLayout::create(&self.context, desc)
    }

    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<VulkanGraphicsPipeline> {
        VulkanGraphicsPipeline::create(&self.context, desc)
    }

    fn create_command_pool(&self, queue: QueueType) -> Result<VulkanCommandPool> {
        VulkanCommandPool::create(&self.context, queue)
    }

    fn create_command_buffer(&self, pool: &mut VulkanCommandPool) -> Result<VulkanCommandBuffer> {
        self.require_context(pool.context())?;
        VulkanCommandBuffer::create(pool)
    }

    fn create_fence(&self, signaled: bool) -> Result<VulkanFence> {
        VulkanFence::create(&self.context, signaled)
    }

    fn create_timeline_semaphore(&self, initial_value: u64) -> Result<VulkanTimelineSemaphore> {
        VulkanTimelineSemaphore::create(&self.context, initial_value)
    }

    fn submit(&self, queue: QueueType, submission: &Submission<'_, Self>) -> Result<()> {
        let marked_frames = self.validate_submission(queue, submission)?;
        let fence = submission.fence;
        let mut waits =
            Vec::with_capacity(submission.surface_frames.len() + submission.wait_timelines.len());
        waits.extend(submission.surface_frames.iter().map(|frame| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.image_available())
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        }));
        waits.extend(submission.wait_timelines.iter().map(|point| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(point.semaphore.raw())
                .value(point.value)
                .stage_mask(convert::pipeline_stages(point.stages))
        }));
        let mut signals =
            Vec::with_capacity(submission.surface_frames.len() + submission.signal_timelines.len());
        signals.extend(submission.surface_frames.iter().map(|frame| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.render_finished())
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        }));
        signals.extend(submission.signal_timelines.iter().map(|point| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(point.semaphore.raw())
                .value(point.value)
                .stage_mask(convert::pipeline_stages(point.stages))
        }));
        let commands = submission
            .command_buffers
            .iter()
            .map(|command| vk::CommandBufferSubmitInfo::default().command_buffer(command.raw()))
            .collect::<Vec<_>>();
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        if let Err(error) = unsafe {
            self.context
                .device
                .queue_submit2(
                    self.context.queue(queue),
                    std::slice::from_ref(&submit),
                    fence.map_or_else(vk::Fence::null, super::resource::VulkanFence::raw),
                )
                .map_err(vk_error)
        } {
            for frame in marked_frames {
                frame.unmark_submitted();
            }
            return Err(error);
        }
        let retained = submission
            .command_buffers
            .iter()
            .flat_map(|command| command.retained())
            .chain(
                submission
                    .surface_frames
                    .iter()
                    .map(|frame| frame.generation.retain()),
            )
            .chain(
                submission
                    .wait_timelines
                    .iter()
                    .map(|point| point.semaphore.retain()),
            )
            .chain(
                submission
                    .signal_timelines
                    .iter()
                    .map(|point| point.semaphore.retain()),
            );
        if let Some(fence) = fence {
            fence.retain_resources(retained);
        }
        Ok(())
    }

    fn create_surface(&self, info: SurfaceCreateInfo) -> Result<VulkanSurface> {
        VulkanSurface::create(&self.context, &info)
    }

    fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<VulkanSwapchain> {
        VulkanSwapchain::create(&self.context, desc)
    }
}
