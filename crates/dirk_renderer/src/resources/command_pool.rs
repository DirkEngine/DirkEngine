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
#[derive(Debug)]
#[expect(
    dead_code,
    reason = "No compute pool exists yet; the marker completes the set of queue kinds."
)]
pub struct Compute;

/// Wrapper for [`vk::CommandPool`].
///
/// Only the renderer thread allocates from, frees into or destroys a pool,
/// which provides the external synchronization Vulkan requires.
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

impl Pool for Compute {
    fn get_index(families: &QueueFamilyIndices) -> u32 {
        families.compute
    }
    fn get_queue_type() -> QueueType {
        QueueType::Compute
    }
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

        // SAFETY: `info` outlives the call and names a queue family the device
        // was created with.
        let pool = unsafe { device.create_command_pool(&info, None)? };

        Ok(Self {
            device: device.clone(),
            pool,
            queue_type,
            pool_type: PhantomData,
        })
    }
    /// Destroys the pool and frees every buffer allocated from it.
    ///
    /// Called exactly once, from the owner's `Drop`, after the device idled
    /// and the owner's [`CommandBuffer`]s were dropped.
    pub fn destroy(&self) {
        // SAFETY: the pool was created from `self.device`. Callers uphold the
        // contract above, so none of its buffers is pending execution.
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }
    }
    #[cfg_attr(
        not(feature = "editor"),
        expect(
            dead_code,
            reason = "Only the editor's egui renderer records its own commands."
        )
    )]
    pub fn raw(&self) -> vk::CommandPool {
        self.pool
    }
    pub fn allocate_buffer(&self) -> Result<CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `self.pool` is live and `allocate_info` outlives the call. On
        // success Vulkan returns exactly `command_buffer_count` (1) buffers.
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
///
/// Invariant: `buff` was allocated from `pool` on `device` and stays valid
/// until `Drop` frees it, and only the renderer thread records it. The `cmd_*`
/// wrappers forward their arguments unchecked, so callers in this crate record
/// only between `begin_command_buffer` and `end_command_buffer` and pass live
/// handles from the same device that outlive the submission.
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
        // SAFETY: see the `CommandBuffer` invariant. Callers begin a buffer
        // once, right after `allocate_buffer`, so it is in the initial state.
        unsafe { self.device.begin_command_buffer(self.buff, begin_info) }
    }
    pub fn end_command_buffer(&self) -> VkResult<()> {
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe { self.device.end_command_buffer(self.buff) }
    }
    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe {
            self.device
                .cmd_bind_pipeline(self.buff, bind_point, pipeline);
        }
    }
    pub fn begin_rendering(&self, rendering_info: &vk::RenderingInfo<'_>) {
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe {
            self.device.cmd_begin_rendering(self.buff, rendering_info);
        }
    }
    pub fn end_rendering(&self) {
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe {
            self.device.cmd_end_rendering(self.buff);
        }
    }
    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe {
            self.device
                .cmd_set_viewport(self.buff, first_viewport, viewports);
        }
    }
    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
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
    #[cfg_attr(
        feature = "editor",
        expect(
            dead_code,
            reason = "The editor shows the viewport in egui instead of copying it to the swapchain."
        )
    )]
    pub fn copy_image(
        &self,
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant.
        unsafe {
            self.device
                .cmd_copy_buffer(self.buff, src_buffer, dst_buffer, regions);
        }
    }
    pub fn pipeline_barrier2(&self, dependency_info: &vk::DependencyInfo<'_>) {
        // SAFETY: see the `CommandBuffer` invariant.
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
        // SAFETY: see the `CommandBuffer` invariant; `begin_single_time` began it.
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
        // SAFETY: `buff` came from `pool` on this device and is freed once.
        // Its owners drop it only after the fence of its submission signalled
        // (or before it was ever submitted), so it is not pending execution.
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
