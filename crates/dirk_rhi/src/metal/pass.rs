use std::{ops::Range, ptr::NonNull};

use dirk_shaders::bindless::heap::metal::{HEAP_BUFFER, PUSH_BUFFER};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLIndexType, MTLLoadAction,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLScissorRect, MTLStoreAction, MTLViewport,
};

use super::{Buffer, CommandBuffer, Device, GraphicsPipeline, usize};
use crate::{
    DrawArguments, DrawIndexedArguments, IndexFormat, LoadOp, MAX_PUSH_DATA_SIZE, Rect,
    RenderPassDesc, StoreOp, Viewport,
};

// Render pass recording. Inputs were validated by the render pass, and every
// resource is from `device` and outlives the frame's work thanks to deferred
// destruction.
impl CommandBuffer {
    /// Begins rendering to the attachments of `desc`, which cover `extent`,
    /// with viewport and scissor covering it too.
    pub(crate) fn begin_rendering(
        &mut self,
        device: &Device,
        desc: &RenderPassDesc<'_>,
        (width, height): (u32, u32),
    ) {
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (index, color) in desc.colors.iter().enumerate() {
            // SAFETY: the index is below the color attachment limit.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(index) };
            attachment.setTexture(Some(&color.image.raw.raw));
            let (load, clear) = color.load.mtl();
            attachment.setLoadAction(load);
            let [red, green, blue, alpha] = clear.unwrap_or_default().map(f64::from);
            attachment.setClearColor(MTLClearColor {
                red,
                green,
                blue,
                alpha,
            });
            attachment.setStoreAction(match (color.resolve, color.store) {
                (None, store) => store.mtl(),
                (Some(_), StoreOp::Store) => MTLStoreAction::StoreAndMultisampleResolve,
                (Some(_), StoreOp::Discard) => MTLStoreAction::MultisampleResolve,
            });
            attachment.setResolveTexture(color.resolve.map(|resolve| &*resolve.raw.raw));
        }
        if let Some(depth) = desc.depth {
            let (load, clear) = depth.load.mtl();
            let attachment = pass.depthAttachment();
            attachment.setTexture(Some(&depth.image.raw.raw));
            attachment.setLoadAction(load);
            attachment.setStoreAction(depth.store.mtl());
            attachment.setClearDepth(f64::from(clear.unwrap_or_default()));
            if depth.image.info().format.has_stencil() {
                let stencil = pass.stencilAttachment();
                stencil.setTexture(Some(&depth.image.raw.raw));
                stencil.setLoadAction(load);
                stencil.setStoreAction(depth.store.mtl());
                stencil.setClearStencil(0);
            }
        }
        let encoder =
            self.begin_render(|commands| commands.renderCommandEncoderWithDescriptor(&pass));
        encoder.setLabel(Some(&NSString::from_str(desc.label)));
        // SAFETY: the heap outlives the work of every frame.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&device.heap.arguments),
                0,
                HEAP_BUFFER as usize,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&device.heap.arguments),
                0,
                HEAP_BUFFER as usize,
            );
        }
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
    pub(crate) fn end_rendering(&mut self, _device: &Device) {
        self.end_encoding();
        self.indices = None;
    }

    pub(crate) fn bind_pipeline(&mut self, _device: &Device, pipeline: &GraphicsPipeline) {
        let encoder = self.render();
        encoder.setRenderPipelineState(&pipeline.raw);
        if let Some(depth) = &pipeline.depth {
            encoder.setDepthStencilState(Some(depth));
        }
        encoder.setCullMode(pipeline.cull_mode);
        encoder.setFrontFacingWinding(pipeline.winding);
        let bias = pipeline.bias;
        encoder.setDepthBias_slopeScale_clamp(bias.constant, bias.slope, bias.clamp);
        self.primitive = pipeline.primitive;
    }

    /// Pushes `data` to every stage; nothing when it is empty.
    pub(crate) fn push(&mut self, _device: &Device, data: &[u8]) {
        let Some(push) = Push::new(data) else { return };
        let encoder = self.render();
        // SAFETY: `push` holds the bytes read during the calls.
        unsafe {
            encoder.setVertexBytes_length_atIndex(push.bytes(), push.length, PUSH_BUFFER as usize);
            encoder.setFragmentBytes_length_atIndex(
                push.bytes(),
                push.length,
                PUSH_BUFFER as usize,
            );
        }
    }

    pub(crate) fn set_viewport(&mut self, _device: &Device, viewport: Viewport) {
        self.render().setViewport(MTLViewport {
            originX: f64::from(viewport.x),
            originY: f64::from(viewport.y),
            width: f64::from(viewport.width),
            height: f64::from(viewport.height),
            znear: f64::from(viewport.min_depth),
            zfar: f64::from(viewport.max_depth),
        });
    }

    pub(crate) fn set_scissor(&mut self, _device: &Device, scissor: Rect) {
        self.render().setScissorRect(MTLScissorRect {
            x: scissor.x as usize,
            y: scissor.y as usize,
            width: scissor.width as usize,
            height: scissor.height as usize,
        });
    }

    /// Binds indices for the following indexed draws, which Metal passes
    /// with each draw.
    pub(crate) fn bind_index_buffer(
        &mut self,
        _device: &Device,
        buffer: &Buffer,
        offset: u64,
        format: IndexFormat,
    ) {
        let format = match format {
            IndexFormat::U16 => MTLIndexType::UInt16,
            IndexFormat::U32 => MTLIndexType::UInt32,
        };
        self.indices = Some((buffer.raw.clone(), offset, format));
    }

    pub(crate) fn draw(&mut self, device: &Device, vertices: Range<u32>, instances: Range<u32>) {
        self.make_resident(device);
        // SAFETY: see the recording note above.
        unsafe {
            self.render()
                .drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    self.primitive,
                    vertices.start as usize,
                    vertices.len(),
                    instances.len(),
                    instances.start as usize,
                );
        }
    }

    pub(crate) fn draw_indexed(
        &mut self,
        device: &Device,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        self.make_resident(device);
        let (buffer, offset, format) = self
            .indices
            .as_ref()
            .expect("indexed draws follow binding an index buffer");
        let size = if *format == MTLIndexType::UInt16 {
            2
        } else {
            4
        };
        // SAFETY: see the recording note above.
        unsafe {
            self.render()
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    self.primitive,
                    indices.len(),
                    *format,
                    buffer,
                    usize(offset + u64::from(indices.start) * size),
                    instances.len(),
                    base_vertex as isize,
                    instances.start as usize,
                );
        }
    }

    /// Makes `count` draws, one Metal draw each, since Metal has no
    /// multi-draw indirect.
    pub(crate) fn draw_indirect(
        &mut self,
        device: &Device,
        buffer: &Buffer,
        offset: u64,
        count: u32,
    ) {
        self.make_resident(device);
        for draw in 0..u64::from(count) {
            let offset = offset + draw * size_of::<DrawArguments>() as u64;
            // SAFETY: see the recording note above; the arguments fit the
            // buffer.
            unsafe {
                self.render()
                    .drawPrimitives_indirectBuffer_indirectBufferOffset(
                        self.primitive,
                        &buffer.raw,
                        usize(offset),
                    );
            }
        }
    }

    /// Makes `count` indexed draws, one Metal draw each.
    pub(crate) fn draw_indexed_indirect(
        &mut self,
        device: &Device,
        buffer: &Buffer,
        offset: u64,
        count: u32,
    ) {
        self.make_resident(device);
        let (indices, index_offset, format) = self
            .indices
            .clone()
            .expect("indexed draws follow binding an index buffer");
        for draw in 0..u64::from(count) {
            let offset = offset + draw * size_of::<DrawIndexedArguments>() as u64;
            // SAFETY: see the recording note above; the arguments fit the
            // buffer, and the caller keeps the indices they read inside the
            // index buffer.
            unsafe {
                self.render()
                    .drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        self.primitive,
                        format,
                        &indices,
                        usize(index_offset),
                        &buffer.raw,
                        usize(offset),
                    );
            }
        }
    }
}

/// Push data as Metal sets it: Metal checks that the bytes cover the
/// shader's push struct, whose size MSL rounds up to its alignment, so they
/// are padded to sixteen.
pub(super) struct Push {
    padded: [u8; MAX_PUSH_DATA_SIZE as usize],
    /// Bytes set, a multiple of sixteen.
    pub(super) length: usize,
}

impl Push {
    /// `data` padded, or `None` when empty.
    pub(super) fn new(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let mut padded = [0; MAX_PUSH_DATA_SIZE as usize];
        padded[..data.len()].copy_from_slice(data);
        Some(Self {
            padded,
            length: data.len().next_multiple_of(16),
        })
    }

    /// The padded bytes, for `setBytes` calls.
    pub(super) fn bytes(&self) -> NonNull<std::ffi::c_void> {
        NonNull::from(&self.padded).cast()
    }
}

impl<T: Copy> LoadOp<T> {
    /// The equivalent Metal load action and the clear value, if it clears.
    fn mtl(self) -> (MTLLoadAction, Option<T>) {
        match self {
            Self::Load => (MTLLoadAction::Load, None),
            Self::Clear(value) => (MTLLoadAction::Clear, Some(value)),
            Self::Discard => (MTLLoadAction::DontCare, None),
        }
    }
}

impl StoreOp {
    /// The equivalent Metal store action.
    const fn mtl(self) -> MTLStoreAction {
        match self {
            Self::Store => MTLStoreAction::Store,
            Self::Discard => MTLStoreAction::DontCare,
        }
    }
}
