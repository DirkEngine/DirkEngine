use std::{marker::PhantomData, ops::Deref};

use ash::{Device, prelude::VkResult, vk};

use crate::{
    Result,
    physical_device::QueueFamilyIndices,
    resources::{
        queues::{QueueType, Queues},
        sync::Fence,
    },
};

#[derive(Debug)]
pub struct Graphics;
#[derive(Debug)]
pub struct Transfer;

/// Wrapper for [`vk::CommandPool`].
pub struct CommandPool<Type: Pool> {
    device: Device,
    /// The command pool
    pool: vk::CommandPool,
    /// The type of queue commands will be submitted to
    queue_type: QueueType,
    pool_type: PhantomData<Type>,
}

pub trait Pool {
    fn get_index(families: &QueueFamilyIndices) -> u32;
    fn get_queue_type() -> QueueType;
}

impl Pool for Transfer {
    fn get_index(families: &QueueFamilyIndices) -> u32 {
        families.transfer
    }
    fn get_queue_type() -> QueueType {
        QueueType::Transfer
    }
}

impl Pool for Graphics {
    fn get_index(families: &QueueFamilyIndices) -> u32 {
        families.graphics
    }
    fn get_queue_type() -> QueueType {
        QueueType::Graphics
    }
}

impl<Type: Pool> CommandPool<Type> {
    /// Will build a command pool with the specified settings.
    pub fn build(
        device: &Device,
        families: &QueueFamilyIndices,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<Self> {
        let index = Type::get_index(families);
        let queue_type = Type::get_queue_type();

        let info = vk::CommandPoolCreateInfo::default()
            .flags(flags)
            .queue_family_index(index);

        let pool = unsafe { device.create_command_pool(&info, None)? };

        Ok(Self {
            device: device.clone(),
            pool,
            queue_type,
            pool_type: PhantomData,
        })
    }
    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }
    }
    #[cfg_attr(not(feature = "editor"), allow(unused))]
    pub fn raw(&self) -> vk::CommandPool {
        self.pool
    }
    pub fn allocate_buffer(&self) -> Result<CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buff = unsafe { self.device.allocate_command_buffers(&allocate_info)?[0] };

        Ok(CommandBuffer {
            device: self.device.clone(),
            buff,
            pool: self.pool,
            queue_type: self.queue_type,
        })
    }

    pub fn begin_single_time(&self) -> Result<CommandBuffer> {
        let buff = self.allocate_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        buff.begin_command_buffer(&begin_info)?;
        Ok(buff)
    }
}

/// Wrapper for [`vk::CommandBuffer`].
pub struct CommandBuffer {
    device: Device,
    /// The buffer
    buff: vk::CommandBuffer,
    /// The pool this command buffer was allocated from.
    pool: vk::CommandPool,
    /// The type of queue to submit to
    queue_type: QueueType,
}

impl CommandBuffer {
    pub fn begin_command_buffer(
        &self,
        begin_info: &vk::CommandBufferBeginInfo<'_>,
    ) -> VkResult<()> {
        unsafe { self.device.begin_command_buffer(self.buff, begin_info) }
    }
    pub fn end_command_buffer(&self) -> VkResult<()> {
        unsafe { self.device.end_command_buffer(self.buff) }
    }
    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.buff, bind_point, pipeline);
        }
    }
    pub fn begin_rendering(&self, rendering_info: &vk::RenderingInfo<'_>) {
        unsafe {
            self.device.cmd_begin_rendering(self.buff, rendering_info);
        }
    }
    pub fn end_rendering(&self) {
        unsafe {
            self.device.cmd_end_rendering(self.buff);
        }
    }
    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device
                .cmd_set_viewport(self.buff, first_viewport, viewports);
        }
    }
    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device
                .cmd_set_scissor(self.buff, first_scissor, scissors);
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
                self.buff,
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
                .cmd_bind_vertex_buffers(self.buff, first_binding, buffers, offsets);
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
                .cmd_bind_index_buffer(self.buff, buffer, offset, index_type);
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
                self.buff,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
    pub fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.buff,
                src_buffer,
                dst_image,
                dst_image_layout,
                regions,
            );
        }
    }
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buff,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }
    pub fn blit_image(
        &self,
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.device.cmd_blit_image(
                self.buff,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                regions,
                filter,
            );
        }
    }
    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn copy_image(
        &self,
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_image(
                self.buff,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                regions,
            );
        }
    }
    pub fn copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        unsafe {
            self.device
                .cmd_copy_buffer(self.buff, src_buffer, dst_buffer, regions);
        }
    }
    pub fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfo<'_>) {
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.buff, dependency_info);
        };
    }

    pub fn submit(
        &self,
        queues: &Queues,
        submit_info: vk::SubmitInfo,
        fence: Option<&Fence>,
    ) -> VkResult<()> {
        queues.submit(self.queue_type, std::slice::from_ref(&submit_info), fence)
    }
    pub fn end_and_submit(&self, queues: &Queues) -> VkResult<()> {
        let info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.buff));
        unsafe {
            self.device.end_command_buffer(self.buff)?;
        };
        let fence = Fence::unsignaled(&self.device)?;

        let submit_result = self.submit(queues, info, Some(&fence));
        if submit_result.is_ok() {
            fence.wait(u64::MAX)
        } else {
            submit_result
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(self.pool, std::slice::from_ref(&self.buff));
        }
    }
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buff
    }
}
