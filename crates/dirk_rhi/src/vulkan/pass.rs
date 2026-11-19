use std::ops::Range;

use ash::vk;

use super::{Buffer, CommandBuffer, Device, GraphicsPipeline, pipeline::PUSH_STAGES};
use crate::{
    DrawArguments, DrawIndexedArguments, Image, IndexFormat, LoadOp, Rect, RenderPassDesc, StoreOp,
    Subresource, Viewport, image::ViewOf,
};

// Render pass recording. Inputs were validated by the render pass, and every
// resource is from `device` and outlives the frame's work thanks to deferred
// destruction.
impl CommandBuffer {
    /// Begins rendering to the attachments of `desc`, which cover `extent`,
    /// with viewport and scissor covering it too.
    pub(crate) fn begin_rendering(
        self,
        device: &Device,
        desc: &RenderPassDesc<'_>,
        (width, height): (u32, u32),
    ) {
        let colors: Vec<_> = desc
            .colors
            .iter()
            .map(|color| {
                let (load, clear) = color.load.vk();
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(color.image.attachment_view(color.subresource))
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(load)
                    .store_op(color.store.vk())
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: clear.unwrap_or_default(),
                        },
                    });
                match color.resolve {
                    Some(resolve) => attachment
                        .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(resolve.attachment_view(Subresource::default()))
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    None => attachment,
                }
            })
            .collect();
        let depth = desc.depth.map(|depth| {
            let (load, clear) = depth.load.vk();
            vk::RenderingAttachmentInfo::default()
                .image_view(depth.image.attachment_view(depth.subresource))
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(load)
                .store_op(depth.store.vk())
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear.unwrap_or_default(),
                        stencil: 0,
                    },
                })
        });
        let stencil = desc
            .depth
            .filter(|depth| depth.image.info().format.has_stencil())
            .and(depth);
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors);
        if let Some(depth) = &depth {
            info = info.depth_attachment(depth);
        }
        if let Some(stencil) = &stencil {
            info = info.stencil_attachment(stencil);
        }
        device.begin_label(self.raw, desc.label);
        device.bind_heap(self.raw, vk::PipelineBindPoint::GRAPHICS);
        // SAFETY: see the recording note above.
        unsafe { device.raw.cmd_begin_rendering(self.raw, &info) };
        #[expect(
            clippy::cast_precision_loss,
            reason = "image extents are far below f32's exact integer range"
        )]
        let size = (width as f32, height as f32);
        self.set_viewport(
            device,
            Viewport {
                x: 0.0,
                y: 0.0,
                width: size.0,
                height: size.1,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        );
        self.set_scissor(
            device,
            Rect {
                x: 0,
                y: 0,
                width,
                height,
            },
        );
    }

    /// Ends the render pass begun last.
    pub(crate) fn end_rendering(self, device: &Device) {
        // SAFETY: see the recording note above.
        unsafe { device.raw.cmd_end_rendering(self.raw) };
        device.end_label(self.raw);
    }

    pub(crate) fn bind_pipeline(self, device: &Device, pipeline: &GraphicsPipeline) {
        // SAFETY: see the recording note above.
        unsafe {
            device
                .raw
                .cmd_bind_pipeline(self.raw, vk::PipelineBindPoint::GRAPHICS, pipeline.raw);
        }
    }

    /// Pushes `data` to every stage; nothing when it is empty.
    pub(crate) fn push(self, device: &Device, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        // SAFETY: see the recording note above; push data is at most the
        // layout's range.
        unsafe {
            device.raw.cmd_push_constants(
                self.raw,
                device.heap.pipeline_layout,
                PUSH_STAGES,
                0,
                data,
            );
        }
    }

    pub(crate) fn set_viewport(self, device: &Device, viewport: Viewport) {
        let viewport = vk::Viewport {
            x: viewport.x,
            y: viewport.y,
            width: viewport.width,
            height: viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        };
        // SAFETY: see the recording note above.
        unsafe { device.raw.cmd_set_viewport(self.raw, 0, &[viewport]) };
    }

    pub(crate) fn set_scissor(self, device: &Device, scissor: Rect) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: scissor.x.cast_signed(),
                y: scissor.y.cast_signed(),
            },
            extent: vk::Extent2D {
                width: scissor.width,
                height: scissor.height,
            },
        };
        // SAFETY: see the recording note above.
        unsafe { device.raw.cmd_set_scissor(self.raw, 0, &[scissor]) };
    }

    pub(crate) fn bind_index_buffer(
        self,
        device: &Device,
        buffer: &Buffer,
        offset: u64,
        format: IndexFormat,
    ) {
        let format = match format {
            IndexFormat::U16 => vk::IndexType::UINT16,
            IndexFormat::U32 => vk::IndexType::UINT32,
        };
        // SAFETY: see the recording note above.
        unsafe {
            device
                .raw
                .cmd_bind_index_buffer(self.raw, buffer.raw, offset, format);
        }
    }

    pub(crate) fn draw(self, device: &Device, vertices: Range<u32>, instances: Range<u32>) {
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_draw(
                self.raw,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    pub(crate) fn draw_indexed(
        self,
        device: &Device,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_draw_indexed(
                self.raw,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
        }
    }

    pub(crate) fn draw_indirect(self, device: &Device, buffer: &Buffer, offset: u64, count: u32) {
        // SAFETY: see the recording note above; the device supports
        // multi-draw indirect and the arguments fit the buffer.
        unsafe {
            device.raw.cmd_draw_indirect(
                self.raw,
                buffer.raw,
                offset,
                count,
                stride::<DrawArguments>(),
            );
        }
    }

    pub(crate) fn draw_indexed_indirect(
        self,
        device: &Device,
        buffer: &Buffer,
        offset: u64,
        count: u32,
    ) {
        // SAFETY: see the recording note above; the device supports
        // multi-draw indirect, the arguments fit the buffer, and the caller
        // keeps the indices they read inside the index buffer.
        unsafe {
            device.raw.cmd_draw_indexed_indirect(
                self.raw,
                buffer.raw,
                offset,
                count,
                stride::<DrawIndexedArguments>(),
            );
        }
    }
}

/// The stride of indirect arguments `T`, packed back to back.
#[expect(
    clippy::cast_possible_truncation,
    reason = "arguments are 20 bytes at most"
)]
const fn stride<T>() -> u32 {
    size_of::<T>() as u32
}

impl Image {
    /// The view of `subresource` alone, which the render pass checked the
    /// image has.
    fn attachment_view(&self, subresource: Subresource) -> vk::ImageView {
        self.raw
            .attachment_view(self.view_of(subresource).unwrap_or(ViewOf::Full))
    }
}

impl<T: Copy> LoadOp<T> {
    /// The equivalent Vulkan load op and the clear value, if it clears.
    fn vk(self) -> (vk::AttachmentLoadOp, Option<T>) {
        match self {
            Self::Load => (vk::AttachmentLoadOp::LOAD, None),
            Self::Clear(value) => (vk::AttachmentLoadOp::CLEAR, Some(value)),
            Self::Discard => (vk::AttachmentLoadOp::DONT_CARE, None),
        }
    }
}

impl StoreOp {
    /// The equivalent Vulkan store op.
    const fn vk(self) -> vk::AttachmentStoreOp {
        match self {
            Self::Store => vk::AttachmentStoreOp::STORE,
            Self::Discard => vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}
