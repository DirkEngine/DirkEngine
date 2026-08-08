use std::sync::Arc;

use ash::vk;
use dirk_rhi::{
    BufferCopy, BufferImageCopy, ColorAttachment, CommandBuffer, DependencyInfo, DepthAttachment,
    FilterMode, ImageBlit, ImageCopy, IndexFormat, LoadOp, QueueType, Rect, RenderingInfo, Result,
    Viewport,
};

use crate::{
    VulkanBackend, VulkanBindGroup, VulkanBuffer, VulkanGraphicsPipeline, VulkanImage,
    VulkanImageView, VulkanPipelineLayout, convert,
    device::{Context, Garbage, Retained},
    vk_error,
};

#[derive(Clone)]
/// Vulkan command allocator associated with one RHI queue.
pub struct VulkanCommandPool(Arc<CommandPoolInner>);

struct CommandPoolInner {
    context: Arc<Context>,
    raw: vk::CommandPool,
    queue: QueueType,
}

impl VulkanCommandPool {
    pub(crate) fn create(context: &Arc<Context>, queue: QueueType) -> Result<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(context.queue_family(queue));
        let raw =
            unsafe { context.device.create_command_pool(&create_info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(CommandPoolInner {
            context: context.clone(),
            raw,
            queue,
        })))
    }

    #[must_use]
    /// Returns the native command-pool handle.
    pub fn raw(&self) -> vk::CommandPool {
        self.0.raw
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for CommandPoolInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::CommandPool(self.raw));
    }
}

/// Primary Vulkan command buffer implementing neutral RHI recording.
pub struct VulkanCommandBuffer {
    pool: Arc<CommandPoolInner>,
    raw: vk::CommandBuffer,
    retained: Vec<Retained>,
}

impl VulkanCommandBuffer {
    pub(crate) fn create(pool: &VulkanCommandPool) -> Result<Self> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.0.raw)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let raw = unsafe {
            pool.0
                .context
                .device
                .allocate_command_buffers(&allocate_info)
        }
        .map_err(vk_error)?[0];
        Ok(Self {
            pool: pool.0.clone(),
            raw,
            retained: Vec::new(),
        })
    }

    #[must_use]
    /// Returns the native command-buffer handle.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.raw
    }

    #[must_use]
    /// Returns the semantic queue this command buffer can be submitted to.
    pub fn queue(&self) -> QueueType {
        self.pool.queue
    }

    pub(crate) fn retained(&self) -> impl Iterator<Item = Retained> + '_ {
        let pool: Retained = self.pool.clone();
        std::iter::once(pool).chain(self.retained.iter().cloned())
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.pool.context
    }

    fn assert_context(&self, context: &Arc<Context>) {
        assert!(
            Arc::ptr_eq(&self.pool.context, context),
            "Vulkan command resource belongs to a different RHI instance"
        );
    }
}

impl CommandBuffer<VulkanBackend> for VulkanCommandBuffer {
    fn begin(&mut self, _label: &str, one_time_submit: bool) -> Result<()> {
        self.retained.clear();
        unsafe {
            self.pool
                .context
                .device
                .reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())
                .map_err(vk_error)?;
            let flags = if one_time_submit {
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            } else {
                vk::CommandBufferUsageFlags::empty()
            };
            self.pool
                .context
                .device
                .begin_command_buffer(
                    self.raw,
                    &vk::CommandBufferBeginInfo::default().flags(flags),
                )
                .map_err(vk_error)
        }
    }

    fn end(&mut self) -> Result<()> {
        unsafe {
            self.pool
                .context
                .device
                .end_command_buffer(self.raw)
                .map_err(vk_error)
        }
    }

    fn begin_rendering(&mut self, info: &RenderingInfo<'_, VulkanBackend>) -> Result<()> {
        for attachment in info.color_attachments {
            self.assert_context(attachment.view.context());
            if let Some(resolve) = attachment.resolve {
                self.assert_context(resolve.context());
            }
        }
        if let Some(attachment) = &info.depth_attachment {
            self.assert_context(attachment.view.context());
        }
        self.retained.extend(
            info.color_attachments
                .iter()
                .flat_map(|attachment| {
                    [
                        Some(attachment.view.retain()),
                        attachment.resolve.map(VulkanImageView::retain),
                    ]
                })
                .flatten(),
        );
        if let Some(attachment) = &info.depth_attachment {
            self.retained.push(attachment.view.retain());
        }
        let color_attachments = info
            .color_attachments
            .iter()
            .map(|attachment| color_attachment(attachment))
            .collect::<Vec<_>>();
        let depth_attachment = info.depth_attachment.as_ref().map(depth_attachment);
        let stencil_attachment = info
            .depth_attachment
            .as_ref()
            .filter(|attachment| {
                attachment
                    .view
                    .aspects()
                    .contains(dirk_rhi::ImageAspects::STENCIL)
            })
            .map(stencil_attachment);
        let mut rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: info.extent.width,
                    height: info.extent.height,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments);
        if let Some(depth) = depth_attachment.as_ref() {
            rendering = rendering.depth_attachment(depth);
        }
        if let Some(stencil) = stencil_attachment.as_ref() {
            rendering = rendering.stencil_attachment(stencil);
        }
        unsafe {
            self.pool
                .context
                .device
                .cmd_begin_rendering(self.raw, &rendering);
        }
        Ok(())
    }

    fn end_rendering(&mut self) -> Result<()> {
        unsafe {
            self.pool.context.device.cmd_end_rendering(self.raw);
        }
        Ok(())
    }

    fn set_viewport(&mut self, viewport: Viewport) {
        let viewport = vk::Viewport {
            x: viewport.x,
            y: viewport.y,
            width: viewport.width,
            height: viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        };
        unsafe {
            self.pool
                .context
                .device
                .cmd_set_viewport(self.raw, 0, std::slice::from_ref(&viewport));
        }
    }

    fn set_scissor(&mut self, scissor: Rect) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: scissor.x,
                y: scissor.y,
            },
            extent: vk::Extent2D {
                width: scissor.width,
                height: scissor.height,
            },
        };
        unsafe {
            self.pool
                .context
                .device
                .cmd_set_scissor(self.raw, 0, std::slice::from_ref(&scissor));
        }
    }

    fn bind_graphics_pipeline(&mut self, pipeline: &VulkanGraphicsPipeline) {
        self.assert_context(pipeline.context());
        self.retained.push(pipeline.retain());
        unsafe {
            self.pool.context.device.cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            );
        }
    }

    fn bind_groups(
        &mut self,
        layout: &VulkanPipelineLayout,
        first_group: u32,
        groups: &[&VulkanBindGroup],
    ) {
        self.assert_context(layout.context());
        for group in groups {
            self.assert_context(group.context());
        }
        self.retained.push(layout.retain());
        self.retained
            .extend(groups.iter().map(|group| group.retain()));
        let groups = groups.iter().map(|group| group.raw()).collect::<Vec<_>>();
        unsafe {
            self.pool.context.device.cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                layout.raw(),
                first_group,
                &groups,
                &[],
            );
        }
    }

    fn bind_vertex_buffer(&mut self, slot: u32, buffer: &VulkanBuffer, offset: u64) {
        self.assert_context(buffer.context());
        self.retained.push(buffer.retain());
        unsafe {
            self.pool.context.device.cmd_bind_vertex_buffers(
                self.raw,
                slot,
                std::slice::from_ref(&buffer.raw()),
                std::slice::from_ref(&offset),
            );
        }
    }

    fn bind_index_buffer(&mut self, buffer: &VulkanBuffer, offset: u64, format: IndexFormat) {
        self.assert_context(buffer.context());
        self.retained.push(buffer.retain());
        unsafe {
            self.pool.context.device.cmd_bind_index_buffer(
                self.raw,
                buffer.raw(),
                offset,
                convert::index(format),
            );
        }
    }

    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.pool.context.device.cmd_draw_indexed(
                self.raw,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    fn copy_buffer(&mut self, src: &VulkanBuffer, dst: &VulkanBuffer, regions: &[BufferCopy]) {
        self.assert_context(src.context());
        self.assert_context(dst.context());
        self.retained.extend([src.retain(), dst.retain()]);
        let regions = regions
            .iter()
            .map(|region| vk::BufferCopy {
                src_offset: region.src_offset,
                dst_offset: region.dst_offset,
                size: region.size,
            })
            .collect::<Vec<_>>();
        unsafe {
            self.pool
                .context
                .device
                .cmd_copy_buffer(self.raw, src.raw(), dst.raw(), &regions);
        }
    }

    fn copy_buffer_to_image(
        &mut self,
        src: &VulkanBuffer,
        dst: &VulkanImage,
        regions: &[BufferImageCopy],
    ) {
        self.assert_context(src.context());
        self.assert_context(dst.context());
        self.retained.extend([src.retain(), dst.retain()]);
        let regions = regions
            .iter()
            .map(|region| {
                vk::BufferImageCopy::default()
                    .buffer_offset(region.buffer_offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: convert::aspects(region.aspects),
                        mip_level: region.mip_level,
                        base_array_layer: region.base_array_layer,
                        layer_count: region.array_layer_count,
                    })
                    .image_extent(vk::Extent3D {
                        width: region.extent.width,
                        height: region.extent.height,
                        depth: region.extent.depth,
                    })
            })
            .collect::<Vec<_>>();
        unsafe {
            self.pool.context.device.cmd_copy_buffer_to_image(
                self.raw,
                src.raw(),
                dst.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    fn copy_image(&mut self, src: &VulkanImage, dst: &VulkanImage, regions: &[ImageCopy]) {
        self.assert_context(src.context());
        self.assert_context(dst.context());
        self.retained.extend([src.retain(), dst.retain()]);
        let regions = regions
            .iter()
            .map(|region| vk::ImageCopy {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: convert::aspects(region.aspects),
                    mip_level: region.src_mip_level,
                    base_array_layer: region.src_base_array_layer,
                    layer_count: region.array_layer_count,
                },
                src_offset: vk::Offset3D {
                    x: i32::try_from(region.src_origin.x).unwrap_or(i32::MAX),
                    y: i32::try_from(region.src_origin.y).unwrap_or(i32::MAX),
                    z: i32::try_from(region.src_origin.z).unwrap_or(i32::MAX),
                },
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: convert::aspects(region.aspects),
                    mip_level: region.dst_mip_level,
                    base_array_layer: region.dst_base_array_layer,
                    layer_count: region.array_layer_count,
                },
                dst_offset: vk::Offset3D {
                    x: i32::try_from(region.dst_origin.x).unwrap_or(i32::MAX),
                    y: i32::try_from(region.dst_origin.y).unwrap_or(i32::MAX),
                    z: i32::try_from(region.dst_origin.z).unwrap_or(i32::MAX),
                },
                extent: vk::Extent3D {
                    width: region.extent.width,
                    height: region.extent.height,
                    depth: region.extent.depth,
                },
            })
            .collect::<Vec<_>>();
        unsafe {
            self.pool.context.device.cmd_copy_image(
                self.raw,
                src.raw(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    fn blit_image(
        &mut self,
        src: &VulkanImage,
        dst: &VulkanImage,
        regions: &[ImageBlit],
        filter: FilterMode,
    ) -> Result<()> {
        self.assert_context(src.context());
        self.assert_context(dst.context());
        self.retained.extend([src.retain(), dst.retain()]);
        let regions = regions
            .iter()
            .map(|region| vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: region.src_mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [vk::Offset3D::default(), extent_offset(region.src_extent)],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: region.dst_mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [vk::Offset3D::default(), extent_offset(region.dst_extent)],
            })
            .collect::<Vec<_>>();
        unsafe {
            self.pool.context.device.cmd_blit_image(
                self.raw,
                src.raw(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
                convert::filter(filter),
            );
        }
        Ok(())
    }

    fn barrier(&mut self, dependency: &DependencyInfo<'_, VulkanBackend>) {
        for barrier in dependency.image_barriers {
            self.assert_context(barrier.image.context());
        }
        self.retained.extend(
            dependency
                .image_barriers
                .iter()
                .map(|barrier| barrier.image.retain()),
        );
        let barriers = dependency
            .image_barriers
            .iter()
            .map(|barrier| {
                let (src_stage, src_access, old_layout) = convert::image_state(barrier.old_state);
                let (dst_stage, dst_access, new_layout) = convert::image_state(barrier.new_state);
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    .old_layout(old_layout)
                    .new_layout(new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(barrier.image.raw())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: convert::aspects(barrier.aspects),
                        base_mip_level: barrier.base_mip_level,
                        level_count: barrier.mip_level_count,
                        base_array_layer: barrier.base_array_layer,
                        layer_count: barrier.array_layer_count,
                    })
            })
            .collect::<Vec<_>>();
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe {
            self.pool
                .context
                .device
                .cmd_pipeline_barrier2(self.raw, &dependency);
        }
    }
}

fn color_attachment(
    attachment: &ColorAttachment<'_, VulkanBackend>,
) -> vk::RenderingAttachmentInfo<'static> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(attachment.view.raw())
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(load_op(&attachment.load))
        .store_op(convert::store(attachment.store));
    if let LoadOp::Clear(color) = attachment.load {
        info = info.clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [color.r, color.g, color.b, color.a],
            },
        });
    }
    if let Some(resolve) = attachment.resolve {
        info = info
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(resolve.raw())
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
    info
}

fn depth_attachment(
    attachment: &DepthAttachment<'_, VulkanBackend>,
) -> vk::RenderingAttachmentInfo<'static> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(attachment.view.raw())
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .load_op(load_op(&attachment.depth_load))
        .store_op(convert::store(attachment.depth_store));
    if let LoadOp::Clear(depth) = attachment.depth_load {
        info = info.clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth,
                stencil: stencil_clear(attachment.stencil_load),
            },
        });
    }
    info
}

fn stencil_attachment(
    attachment: &DepthAttachment<'_, VulkanBackend>,
) -> vk::RenderingAttachmentInfo<'static> {
    let mut info = vk::RenderingAttachmentInfo::default()
        .image_view(attachment.view.raw())
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .load_op(load_op(&attachment.stencil_load))
        .store_op(convert::store(attachment.stencil_store));
    if let LoadOp::Clear(stencil) = attachment.stencil_load {
        info = info.clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: depth_clear(attachment.depth_load),
                stencil,
            },
        });
    }
    info
}

fn load_op<T>(value: &LoadOp<T>) -> vk::AttachmentLoadOp {
    match value {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

fn depth_clear(value: LoadOp<f32>) -> f32 {
    if let LoadOp::Clear(value) = value {
        value
    } else {
        1.0
    }
}

fn stencil_clear(value: LoadOp<u32>) -> u32 {
    if let LoadOp::Clear(value) = value {
        value
    } else {
        0
    }
}

fn extent_offset(extent: dirk_rhi::Extent3d) -> vk::Offset3D {
    vk::Offset3D {
        x: i32::try_from(extent.width).unwrap_or(i32::MAX),
        y: i32::try_from(extent.height).unwrap_or(i32::MAX),
        z: i32::try_from(extent.depth).unwrap_or(i32::MAX),
    }
}
