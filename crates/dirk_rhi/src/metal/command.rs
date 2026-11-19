use std::sync::Arc;

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor,
    MTLCommandBufferErrorOption, MTLCommandEncoder, MTLCommandQueue, MTLIndexType, MTLOrigin,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLSize,
};

use super::{Buffer, Check, Device, Image, Object, usize};
use crate::{
    BufferBarrier, BufferCopy, BufferImageCopy, Format, ImageCopy, ImageRegion, QueueType, Result,
    command::Transition,
};

/// Creates command buffers for one queue and frame slot. Metal command
/// buffers are created per submission, so there is nothing to recycle.
pub(crate) struct CommandPool {
    device: Arc<Device>,
    queue: QueueType,
}

/// The encoder a command buffer records with. Metal records copies and
/// render passes with separate encoders, one at a time.
enum Encoder {
    None,
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    Render(Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>),
}

/// A command buffer being recorded.
pub(crate) struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Encoder,
    /// Primitives the bound pipeline draws.
    pub(super) primitive: MTLPrimitiveType,
    /// The bound index buffer, its offset and type.
    pub(super) indices: Option<(Object<dyn MTLBuffer>, u64, MTLIndexType)>,
}

impl CommandPool {
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Vulkan, where this can fail"
    )]
    pub(crate) fn new(device: &Arc<Device>, queue: QueueType) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            queue,
        })
    }

    /// Nothing to reset: each frame creates its command buffers.
    #[expect(
        clippy::unnecessary_wraps,
        clippy::unused_self,
        reason = "the signature is shared with Vulkan, where this can fail"
    )]
    pub(crate) fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    /// Begins recording a command buffer labelled `label`. With validation,
    /// failures report the encoder that failed.
    pub(crate) fn begin(&mut self, label: &str) -> Result<CommandBuffer> {
        let desc = MTLCommandBufferDescriptor::new();
        if self.device.validation {
            desc.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        }
        let raw = autoreleasepool(|_| {
            self.device.queues[self.queue as usize]
                .raw
                .commandBufferWithDescriptor(&desc)
        })
        .check("commandBufferWithDescriptor")?;
        raw.setLabel(Some(&NSString::from_str(label)));
        Ok(CommandBuffer {
            raw,
            encoder: Encoder::None,
            primitive: MTLPrimitiveType::Triangle,
            indices: None,
        })
    }
}

impl CommandBuffer {
    /// The command buffer, to commit once encoding ended.
    pub(super) fn raw(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }

    /// Ends the open encoder, if any.
    pub(super) fn end_encoding(&mut self) {
        match std::mem::replace(&mut self.encoder, Encoder::None) {
            Encoder::None => {}
            Encoder::Blit(encoder) => encoder.endEncoding(),
            Encoder::Render(encoder) => encoder.endEncoding(),
        }
    }

    /// The open blit encoder, beginning one if needed.
    pub(super) fn blit(&mut self) -> &ProtocolObject<dyn MTLBlitCommandEncoder> {
        if !matches!(self.encoder, Encoder::Blit(_)) {
            self.end_encoding();
            let encoder = autoreleasepool(|_| self.raw.blitCommandEncoder())
                .expect("Metal creates encoders for command buffers without an open one");
            self.encoder = Encoder::Blit(encoder);
        }
        match &self.encoder {
            Encoder::Blit(encoder) => encoder,
            _ => unreachable!("a blit encoder was just begun"),
        }
    }

    /// Begins a render pass with `encoder`, ending the open encoder.
    pub(super) fn begin_render(
        &mut self,
        begin: impl FnOnce(
            &ProtocolObject<dyn MTLCommandBuffer>,
        ) -> Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>,
    ) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        self.end_encoding();
        let encoder = autoreleasepool(|_| begin(&self.raw))
            .expect("Metal creates render encoders for validated passes");
        self.encoder = Encoder::Render(encoder);
        self.render()
    }

    /// The open render encoder.
    pub(super) fn render(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        match &self.encoder {
            Encoder::Render(encoder) => encoder,
            _ => unreachable!("render pass commands are recorded inside a render pass"),
        }
    }
}

impl Drop for CommandBuffer {
    /// Metal requires encoders to end, even in discarded command buffers.
    fn drop(&mut self) {
        self.end_encoding();
    }
}

// Recording. Inputs were validated by the encoder, and every resource is from
// `device` and outlives the frame's work thanks to deferred destruction.
impl CommandBuffer {
    pub(crate) fn copy_buffer(
        &mut self,
        _device: &Device,
        src: &Buffer,
        dst: &Buffer,
        regions: &[BufferCopy],
    ) {
        let blit = self.blit();
        for region in regions {
            // SAFETY: see the recording note above.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src.raw,
                    usize(region.src_offset),
                    &dst.raw,
                    usize(region.dst_offset),
                    usize(region.size),
                );
            }
        }
    }

    pub(crate) fn copy_buffer_to_image(
        &mut self,
        _device: &Device,
        src: &Buffer,
        dst: &Image,
        _format: Format,
        regions: &[BufferImageCopy],
    ) {
        let blit = self.blit();
        for region in regions {
            for (slice, offset) in region.slices() {
                // SAFETY: see the recording note above.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src.raw,
                        offset,
                        region.bytes_per_row as usize,
                        region.bytes_per_image(),
                        region.image.mtl_size(),
                        &dst.raw,
                        slice,
                        region.image.mip_level as usize,
                        region.image.mtl_origin(),
                    );
                }
            }
        }
    }

    pub(crate) fn copy_image_to_buffer(
        &mut self,
        _device: &Device,
        src: &Image,
        dst: &Buffer,
        _format: Format,
        regions: &[BufferImageCopy],
    ) {
        let blit = self.blit();
        for region in regions {
            for (slice, offset) in region.slices() {
                // SAFETY: see the recording note above.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        &src.raw,
                        slice,
                        region.image.mip_level as usize,
                        region.image.mtl_origin(),
                        region.image.mtl_size(),
                        &dst.raw,
                        offset,
                        region.bytes_per_row as usize,
                        region.bytes_per_image(),
                    );
                }
            }
        }
    }

    pub(crate) fn copy_image(
        &mut self,
        _device: &Device,
        src: &Image,
        dst: &Image,
        _format: Format,
        regions: &[ImageCopy],
    ) {
        let blit = self.blit();
        for region in regions {
            for layer in 0..region.src.layer_count {
                // SAFETY: see the recording note above.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src.raw,
                        (region.src.base_layer + layer) as usize,
                        region.src.mip_level as usize,
                        region.src.mtl_origin(),
                        region.src.mtl_size(),
                        &dst.raw,
                        (region.dst.base_layer + layer) as usize,
                        region.dst.mip_level as usize,
                        region.dst.mtl_origin(),
                    );
                }
            }
        }
    }

    /// Generates every mip of `image` from mip 0. Metal orders the levels
    /// and tracks the hazards around them, and rejects images of one mip,
    /// which have none to generate.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Vulkan, whose blits can fail"
    )]
    pub(crate) fn generate_mips(&mut self, _device: &Device, image: &crate::Image) -> Result<()> {
        if image.info().mip_levels > 1 {
            self.blit().generateMipmapsForTexture(&image.raw.raw);
        }
        Ok(())
    }

    /// Records nothing: Metal tracks every hazard, and its queues need no
    /// ownership transfers; see the backend's documentation.
    #[expect(
        clippy::unused_self,
        reason = "Vulkan records the barrier into the command buffer"
    )]
    pub(crate) fn barrier(
        &mut self,
        _device: &Device,
        _queue: QueueType,
        _buffers: &[BufferBarrier<'_>],
        _images: &[Transition],
    ) {
    }
}

impl BufferImageCopy {
    /// Each copied slice of the image, as a Metal slice index, and the
    /// offset of its texels in the buffer. Layers are copied one by one;
    /// a 3D region is one slice of `depth` images.
    fn slices(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let image = self.bytes_per_image();
        let (base, offset) = (self.image.base_layer as usize, usize(self.buffer_offset));
        (0..self.image.layer_count as usize)
            .map(move |layer| (base + layer, offset + layer * image))
    }

    /// Bytes from one layer or depth slice to the next.
    fn bytes_per_image(&self) -> usize {
        self.bytes_per_row as usize * self.rows_per_image as usize
    }
}

impl ImageRegion {
    /// The region's first texel.
    pub(super) const fn mtl_origin(&self) -> MTLOrigin {
        MTLOrigin {
            x: self.origin.x as usize,
            y: self.origin.y as usize,
            z: self.origin.z as usize,
        }
    }

    /// The region's size.
    pub(super) const fn mtl_size(&self) -> MTLSize {
        MTLSize {
            width: self.extent.width as usize,
            height: self.extent.height as usize,
            depth: self.extent.depth as usize,
        }
    }
}
