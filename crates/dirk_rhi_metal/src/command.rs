use std::sync::Arc;

use dirk_rhi::{
    BufferCopy, BufferImageCopy, CommandBuffer as RhiCommandBuffer, DependencyInfo, FilterMode,
    ImageBlit, ImageCopy, IndexFormat, Origin3d, QueueType, Rect, RenderingInfo, Result,
    ShaderStages, Viewport,
};
use metal::foreign_types::ForeignType;
use metal::{
    MTLBlitOption, MTLClearColor, MTLOrigin, MTLScissorRect, MTLSize, MTLViewport,
    RenderPassDescriptor,
};
use parking_lot::Mutex;

use crate::{
    MetalBackend,
    backend::Context,
    convert,
    resource::{
        MetalBindGroup, MetalBuffer, MetalGraphicsPipeline, MetalImage, MetalPipelineLayout,
        OwnedBinding, VERTEX_BUFFER_BASE, binding_visibility, require_context,
    },
};

/// Lightweight Metal command allocator for one semantic queue.
pub struct MetalCommandPool {
    pub(crate) context: Arc<Context>,
    pub(crate) queue: QueueType,
}

struct IndexBinding {
    buffer: MetalBuffer,
    offset: u64,
    format: IndexFormat,
}

struct CommandState {
    command: Option<metal::CommandBuffer>,
    render: Option<metal::RenderCommandEncoder>,
    pipeline: Option<MetalGraphicsPipeline>,
    index: Option<IndexBinding>,
    ended: bool,
    submitted: bool,
}

/// Metal command buffer implementing the backend-neutral recording contract.
pub struct MetalCommandBuffer {
    pub(crate) context: Arc<Context>,
    pub(crate) queue: QueueType,
    state: Mutex<CommandState>,
}

impl MetalCommandBuffer {
    pub(crate) fn create(pool: &MetalCommandPool) -> Self {
        Self {
            context: pool.context.clone(),
            queue: pool.queue,
            state: Mutex::new(CommandState {
                command: None,
                render: None,
                pipeline: None,
                index: None,
                ended: false,
                submitted: false,
            }),
        }
    }

    pub(crate) fn command_for_submit(&self) -> Result<metal::CommandBuffer> {
        let mut state = self.state.lock();
        if !state.ended || state.submitted {
            return Err(dirk_rhi::Error::InvalidResource(
                "Metal command buffer must be ended exactly once before submission",
            ));
        }
        state.submitted = true;
        state
            .command
            .clone()
            .ok_or(dirk_rhi::Error::InvalidResource(
                "Metal command buffer has not begun recording",
            ))
    }

    fn with_blit(&mut self, encode: impl FnOnce(&metal::BlitCommandEncoderRef)) {
        let state = self.state.get_mut();
        if state.render.is_some() {
            return;
        }
        if let Some(command) = &state.command {
            let encoder = command.new_blit_command_encoder();
            encode(encoder);
            encoder.end_encoding();
        }
    }
}

impl RhiCommandBuffer<MetalBackend> for MetalCommandBuffer {
    fn begin(&mut self, label: &str, _one_time_submit: bool) -> Result<()> {
        let state = self.state.get_mut();
        if state.command.is_some() && !state.submitted {
            return Err(dirk_rhi::Error::InvalidResource(
                "Metal command buffer is already recording",
            ));
        }
        let command = self
            .context
            .queue(self.queue)
            .new_command_buffer()
            .to_owned();
        command.set_label(label);
        *state = CommandState {
            command: Some(command),
            render: None,
            pipeline: None,
            index: None,
            ended: false,
            submitted: false,
        };
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        let state = self.state.get_mut();
        if state.render.is_some() {
            return Err(dirk_rhi::Error::InvalidResource(
                "end_rendering must be called before ending a command buffer",
            ));
        }
        if state.command.is_none() || state.ended {
            return Err(dirk_rhi::Error::InvalidResource(
                "Metal command buffer is not recording",
            ));
        }
        state.ended = true;
        Ok(())
    }

    fn begin_rendering(&mut self, info: &RenderingInfo<'_, MetalBackend>) -> Result<()> {
        let state = self.state.get_mut();
        if state.render.is_some() {
            return Err(dirk_rhi::Error::InvalidResource(
                "nested rendering passes are not supported",
            ));
        }
        let command = state
            .command
            .as_ref()
            .ok_or(dirk_rhi::Error::InvalidResource(
                "command buffer has not begun recording",
            ))?;
        let descriptor = RenderPassDescriptor::new();
        for (index, attachment) in info.color_attachments.iter().enumerate() {
            require_context(&self.context, &attachment.view.context)?;
            let metal_attachment =
                descriptor
                    .color_attachments()
                    .object_at(u64::try_from(index).map_err(|_| {
                        dirk_rhi::Error::InvalidResource("too many color attachments")
                    })?)
                    .ok_or(dirk_rhi::Error::InvalidResource(
                        "too many color attachments",
                    ))?;
            metal_attachment.set_texture(Some(&attachment.view.raw));
            metal_attachment.set_load_action(convert::load(&attachment.load));
            if let dirk_rhi::LoadOp::Clear(color) = attachment.load {
                metal_attachment.set_clear_color(MTLClearColor::new(
                    f64::from(color.r),
                    f64::from(color.g),
                    f64::from(color.b),
                    f64::from(color.a),
                ));
            }
            if let Some(resolve) = attachment.resolve {
                require_context(&self.context, &resolve.context)?;
                metal_attachment.set_resolve_texture(Some(&resolve.raw));
            }
            metal_attachment.set_store_action(convert::store(
                attachment.store,
                attachment.resolve.is_some(),
            ));
        }
        if let Some(attachment) = &info.depth_attachment {
            require_context(&self.context, &attachment.view.context)?;
            if attachment
                .view
                .aspects
                .contains(dirk_rhi::ImageAspects::DEPTH)
            {
                let depth =
                    descriptor
                        .depth_attachment()
                        .ok_or(dirk_rhi::Error::InvalidResource(
                            "Metal depth attachment is unavailable",
                        ))?;
                depth.set_texture(Some(&attachment.view.raw));
                depth.set_load_action(convert::load(&attachment.depth_load));
                depth.set_store_action(convert::store(attachment.depth_store, false));
                if let dirk_rhi::LoadOp::Clear(value) = attachment.depth_load {
                    depth.set_clear_depth(f64::from(value));
                }
            }
            if attachment
                .view
                .aspects
                .contains(dirk_rhi::ImageAspects::STENCIL)
            {
                let stencil =
                    descriptor
                        .stencil_attachment()
                        .ok_or(dirk_rhi::Error::InvalidResource(
                            "Metal stencil attachment is unavailable",
                        ))?;
                stencil.set_texture(Some(&attachment.view.raw));
                stencil.set_load_action(convert::load(&attachment.stencil_load));
                stencil.set_store_action(convert::store(attachment.stencil_store, false));
                if let dirk_rhi::LoadOp::Clear(value) = attachment.stencil_load {
                    stencil.set_clear_stencil(value);
                }
            }
        }
        let encoder = command.new_render_command_encoder(descriptor).to_owned();
        encoder.set_label(info.label);
        state.render = Some(encoder);
        Ok(())
    }

    fn end_rendering(&mut self) -> Result<()> {
        let encoder =
            self.state
                .get_mut()
                .render
                .take()
                .ok_or(dirk_rhi::Error::InvalidResource(
                    "no Metal rendering pass is active",
                ))?;
        encoder.end_encoding();
        Ok(())
    }

    fn set_viewport(&mut self, viewport: Viewport) {
        if let Some(encoder) = &self.state.get_mut().render {
            encoder.set_viewport(MTLViewport {
                originX: f64::from(viewport.x),
                originY: f64::from(viewport.y),
                width: f64::from(viewport.width),
                height: f64::from(viewport.height),
                znear: f64::from(viewport.min_depth),
                zfar: f64::from(viewport.max_depth),
            });
        }
    }

    fn set_scissor(&mut self, scissor: Rect) {
        if let Some(encoder) = &self.state.get_mut().render {
            encoder.set_scissor_rect(MTLScissorRect {
                x: u64::try_from(scissor.x.max(0)).unwrap_or(0),
                y: u64::try_from(scissor.y.max(0)).unwrap_or(0),
                width: u64::from(scissor.width),
                height: u64::from(scissor.height),
            });
        }
    }

    fn bind_graphics_pipeline(&mut self, pipeline: &MetalGraphicsPipeline) {
        if require_context(&self.context, &pipeline.context).is_err() {
            return;
        }
        let state = self.state.get_mut();
        if let Some(encoder) = &state.render {
            encoder.set_render_pipeline_state(&pipeline.raw);
            encoder.set_front_facing_winding(pipeline.winding);
            encoder.set_cull_mode(pipeline.cull);
            if let Some(depth) = &pipeline.depth {
                encoder.set_depth_stencil_state(depth);
            }
            state.pipeline = Some(pipeline.clone());
        }
    }

    fn bind_groups(
        &mut self,
        layout: &MetalPipelineLayout,
        first_group: u32,
        groups: &[&MetalBindGroup],
    ) {
        if require_context(&self.context, &layout.context).is_err() {
            return;
        }
        let Some(encoder) = &self.state.get_mut().render else {
            return;
        };
        for (relative, group) in groups.iter().enumerate() {
            let Ok(relative) = u32::try_from(relative) else {
                return;
            };
            let Some(group_index) = first_group.checked_add(relative) else {
                return;
            };
            let Some(offsets) = layout.offsets.get(group_index as usize) else {
                return;
            };
            let Some(expected_layout) = layout.layouts.get(group_index as usize) else {
                return;
            };
            if !Arc::ptr_eq(&group.layout.context, &expected_layout.context)
                || group.layout.entries.as_ref() != expected_layout.entries.as_ref()
            {
                return;
            }
            for (binding, resource) in group.entries.iter() {
                let visibility = binding_visibility(&group.layout, *binding);
                let buffer_index = offsets.buffers + u64::from(*binding);
                let texture_index = offsets.textures + u64::from(*binding);
                let sampler_index = offsets.samplers + u64::from(*binding);
                match resource {
                    OwnedBinding::Buffer { buffer, offset } => {
                        if visibility.contains(ShaderStages::VERTEX) {
                            encoder.set_vertex_buffer(buffer_index, Some(&buffer.raw), *offset);
                        }
                        if visibility.contains(ShaderStages::FRAGMENT) {
                            encoder.set_fragment_buffer(buffer_index, Some(&buffer.raw), *offset);
                        }
                    }
                    OwnedBinding::SampledImage { view, sampler } => {
                        bind_texture(encoder, visibility, texture_index, &view.raw);
                        if visibility.contains(ShaderStages::VERTEX) {
                            encoder.set_vertex_sampler_state(sampler_index, Some(&sampler.raw));
                        }
                        if visibility.contains(ShaderStages::FRAGMENT) {
                            encoder.set_fragment_sampler_state(sampler_index, Some(&sampler.raw));
                        }
                    }
                    OwnedBinding::StorageImage(view) => {
                        bind_texture(encoder, visibility, texture_index, &view.raw);
                    }
                }
            }
        }
    }

    fn bind_vertex_buffer(&mut self, slot: u32, buffer: &MetalBuffer, offset: u64) {
        if let Some(encoder) = &self.state.get_mut().render {
            encoder.set_vertex_buffer(
                VERTEX_BUFFER_BASE + u64::from(slot),
                Some(&buffer.raw),
                offset,
            );
        }
    }

    fn bind_index_buffer(&mut self, buffer: &MetalBuffer, offset: u64, format: IndexFormat) {
        self.state.get_mut().index = Some(IndexBinding {
            buffer: buffer.clone(),
            offset,
            format,
        });
    }

    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let state = self.state.get_mut();
        let (Some(encoder), Some(pipeline), Some(index)) =
            (&state.render, &state.pipeline, &state.index)
        else {
            return;
        };
        let index_size = match index.format {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        };
        encoder.draw_indexed_primitives_instanced_base_instance(
            pipeline.topology,
            u64::from(index_count),
            convert::index(index.format),
            &index.buffer.raw,
            index.offset + u64::from(first_index) * index_size,
            u64::from(instance_count),
            i64::from(vertex_offset),
            u64::from(first_instance),
        );
    }

    fn copy_buffer(&mut self, src: &MetalBuffer, dst: &MetalBuffer, regions: &[BufferCopy]) {
        self.with_blit(|encoder| {
            for region in regions {
                encoder.copy_from_buffer(
                    &src.raw,
                    region.src_offset,
                    &dst.raw,
                    region.dst_offset,
                    region.size,
                );
            }
        });
    }

    fn copy_buffer_to_image(
        &mut self,
        src: &MetalBuffer,
        dst: &MetalImage,
        regions: &[BufferImageCopy],
    ) {
        self.with_blit(|encoder| {
            for region in regions {
                let bytes_per_row =
                    u64::from(region.extent.width) * convert::bytes_per_pixel(dst.format);
                let bytes_per_image = bytes_per_row * u64::from(region.extent.height);
                for layer in 0..region.array_layer_count {
                    encoder.copy_from_buffer_to_texture(
                        &src.raw,
                        region.buffer_offset + bytes_per_image * u64::from(layer),
                        bytes_per_row,
                        bytes_per_image,
                        size(region.extent),
                        &dst.raw,
                        u64::from(region.base_array_layer + layer),
                        u64::from(region.mip_level),
                        MTLOrigin::default(),
                        MTLBlitOption::empty(),
                    );
                }
            }
        });
    }

    fn copy_image(&mut self, src: &MetalImage, dst: &MetalImage, regions: &[ImageCopy]) {
        self.with_blit(|encoder| {
            for region in regions {
                for layer in 0..region.array_layer_count {
                    encoder.copy_from_texture(
                        &src.raw,
                        u64::from(region.src_base_array_layer + layer),
                        u64::from(region.src_mip_level),
                        origin(region.src_origin),
                        size(region.extent),
                        &dst.raw,
                        u64::from(region.dst_base_array_layer + layer),
                        u64::from(region.dst_mip_level),
                        origin(region.dst_origin),
                    );
                }
            }
        });
    }

    fn blit_image(
        &mut self,
        src: &MetalImage,
        dst: &MetalImage,
        regions: &[ImageBlit],
        filter: FilterMode,
    ) -> Result<()> {
        if regions.is_empty() {
            return Ok(());
        }
        let same_texture = src.raw.as_ptr() == dst.raw.as_ptr();
        if same_texture {
            if filter != FilterMode::Linear
                || regions
                    .iter()
                    .any(|region| region.src_mip_level.checked_add(1) != Some(region.dst_mip_level))
            {
                return Err(dirk_rhi::Error::Unsupported(
                    "Metal same-image blits only support linear mipmap generation",
                ));
            }
            self.with_blit(|encoder| {
                encoder.generate_mipmaps(&src.raw);
            });
            return Ok(());
        }
        if regions
            .iter()
            .any(|region| region.src_extent != region.dst_extent)
        {
            return Err(dirk_rhi::Error::Unsupported(
                "Metal does not natively support scaled cross-image blits",
            ));
        }
        self.with_blit(|encoder| {
            for region in regions {
                encoder.copy_from_texture(
                    &src.raw,
                    0,
                    u64::from(region.src_mip_level),
                    MTLOrigin::default(),
                    size(region.src_extent),
                    &dst.raw,
                    0,
                    u64::from(region.dst_mip_level),
                    MTLOrigin::default(),
                );
            }
        });
        Ok(())
    }

    fn barrier(&mut self, _dependency: &DependencyInfo<'_, MetalBackend>) {
        // Resources use Metal's tracked hazard mode, so encoder boundaries are
        // sufficient for render-graph transitions and visibility.
    }
}

fn bind_texture(
    encoder: &metal::RenderCommandEncoderRef,
    visibility: ShaderStages,
    index: u64,
    texture: &metal::TextureRef,
) {
    if visibility.contains(ShaderStages::VERTEX) {
        encoder.set_vertex_texture(index, Some(texture));
    }
    if visibility.contains(ShaderStages::FRAGMENT) {
        encoder.set_fragment_texture(index, Some(texture));
    }
}

fn origin(value: Origin3d) -> MTLOrigin {
    MTLOrigin {
        x: u64::from(value.x),
        y: u64::from(value.y),
        z: u64::from(value.z),
    }
}

fn size(value: dirk_rhi::Extent3d) -> MTLSize {
    MTLSize {
        width: u64::from(value.width),
        height: u64::from(value.height),
        depth: u64::from(value.depth),
    }
}
