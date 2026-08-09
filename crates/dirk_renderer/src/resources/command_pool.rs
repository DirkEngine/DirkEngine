use std::{marker::PhantomData, ops::Deref, sync::Arc};

use ash::vk;
use dirk_rhi::{Backend as _, CommandBuffer as _, Fence as _, QueueType, Submission};
use dirk_rhi_vulkan::{VulkanCommandBuffer, VulkanCommandPool};

use crate::{Result, resources::ActiveRhi};

#[derive(Debug)]
pub struct Graphics;
#[derive(Debug)]
pub struct Transfer;

/// Queue marker implemented by command-pool kinds used by the renderer.
pub trait Pool {
    /// Semantic RHI queue used by this pool.
    const QUEUE: QueueType;
}

impl Pool for Graphics {
    const QUEUE: QueueType = QueueType::Graphics;
}

impl Pool for Transfer {
    const QUEUE: QueueType = QueueType::Copy;
}

/// Typed renderer wrapper around an RHI command pool.
pub struct CommandPool<Type: Pool> {
    rhi: Arc<ActiveRhi>,
    inner: VulkanCommandPool,
    pool_type: PhantomData<Type>,
}

impl<Type: Pool> CommandPool<Type> {
    /// Creates a resettable command pool for this marker's queue.
    pub fn build(rhi: &Arc<ActiveRhi>) -> Result<Self> {
        Ok(Self {
            inner: rhi.create_command_pool(Type::QUEUE)?,
            rhi: rhi.clone(),
            pool_type: PhantomData,
        })
    }

    #[cfg_attr(not(feature = "editor"), allow(unused))]
    pub fn raw(&self) -> vk::CommandPool {
        self.inner.raw()
    }

    pub fn allocate_buffer(&self) -> Result<CommandBuffer> {
        let inner = self.rhi.create_command_buffer(&self.inner)?;
        Ok(CommandBuffer {
            device: self.rhi.device().clone(),
            raw: inner.raw(),
            inner,
            rhi: self.rhi.clone(),
        })
    }

    pub fn begin_single_time(&self) -> Result<CommandBuffer> {
        let mut command = self.allocate_buffer()?;
        command.inner.begin("immediate renderer command", true)?;
        Ok(command)
    }
}

/// Renderer command buffer backed by the active RHI.
pub struct CommandBuffer {
    device: ash::Device,
    raw: vk::CommandBuffer,
    inner: VulkanCommandBuffer,
    rhi: Arc<ActiveRhi>,
}

impl CommandBuffer {
    pub fn begin_command_buffer(
        &mut self,
        begin_info: &vk::CommandBufferBeginInfo<'_>,
    ) -> Result<()> {
        self.inner.begin(
            "renderer command buffer",
            begin_info
                .flags
                .contains(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        Ok(())
    }

    pub fn end_command_buffer(&mut self) -> Result<()> {
        self.inner.end()?;
        Ok(())
    }

    pub(crate) fn rhi_mut(&mut self) -> &mut VulkanCommandBuffer {
        &mut self.inner
    }

    pub(crate) fn rhi(&self) -> &VulkanCommandBuffer {
        &self.inner
    }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.inner.raw(), bind_point, pipeline);
        }
    }

    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device
                .cmd_set_viewport(self.inner.raw(), first_viewport, viewports);
        }
    }

    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device
                .cmd_set_scissor(self.inner.raw(), first_scissor, scissors);
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.inner.raw(),
                bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    pub fn bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.inner.raw(), first_binding, buffers, offsets);
        }
    }

    pub fn bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.inner.raw(), buffer, offset, index_type);
        }
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.inner.raw(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    /// Ends, submits, and waits for a short-lived command buffer.
    pub fn end_and_submit(&mut self) -> Result<()> {
        self.inner.end()?;
        let fence = self.rhi.create_fence(false)?;
        self.rhi.submit(
            self.inner.queue(),
            &Submission {
                command_buffers: &[&self.inner],
                surface_frames: &[],
                wait_timelines: &[],
                signal_timelines: &[],
                fence: &fence,
            },
        )?;
        fence.wait(u64::MAX)?;
        Ok(())
    }
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        // Vulkan handles are plain values; the backend wrapper owns the
        // command allocation for at least as long as this renderer wrapper.
        &self.raw
    }
}
