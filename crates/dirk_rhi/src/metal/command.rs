//! Command recording, and how encoders are ordered.
//!
//! Resources are untracked (see `allocator`), so Metal orders nothing
//! between encoders; fences carry out the RHI's barriers instead. Every
//! encoder waits, when it begins, for the fences of the encoders before
//! the last barrier, and updates a fence of its own when it ends. A barrier
//! ends the open encoder, and the encoders begun since the barrier before
//! become the ones later encoders wait for. Since each of those waited for
//! the encoders before it, every encoder after a barrier runs after every
//! encoder before it, as with a Vulkan barrier; encoders between two
//! barriers overlap freely.
//!
//! Command buffers continue the chain through their queue's join fence:
//! their first encoders wait for it, and submitting one ends it with an
//! encoder that waits for its last encoders and updates the join fence.
//! Metal evaluates fences in submission order, so each command buffer runs
//! after everything submitted to its queue before it.

use std::sync::Arc;

use dirk_shaders::bindless::heap::metal::HEAP_BUFFER;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor,
    MTLCommandBufferErrorOption, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLFence, MTLHeap, MTLIndexType, MTLOrigin, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderStages, MTLSize,
};
use parking_lot::Mutex;

use super::{
    Buffer, Check, Device, Image, Object,
    compute::ComputePipelineId,
    heap::{Declared, Heap, Usage},
    usize,
};
use crate::{
    BufferBarrier, BufferCopy, BufferImageCopy, Format, ImageCopy, ImageRegion, QueueType, Result,
    command::Transition,
};

/// A fence an encoder updates when it ends.
pub(super) type Fence = Object<dyn MTLFence>;

/// The stages render encoders wait at and update fences after: all of them.
const RENDER_STAGES: MTLRenderStages =
    MTLRenderStages(MTLRenderStages::Vertex.0 | MTLRenderStages::Fragment.0);

/// The fences of one queue and frame slot's command buffers, one per
/// encoder, reused once the slot comes round again: by then no submitted
/// work waits for them.
struct Fences {
    device: Object<dyn MTLDevice>,
    /// Fences no command buffer of the slot's current frame used.
    free: Vec<Fence>,
    /// Fences the slot's current frame used.
    used: Vec<Fence>,
}

impl Fences {
    /// A fence for a new encoder.
    #[expect(
        clippy::expect_used,
        reason = "Metal fails to create fences only when out of memory, as Rust's allocations do."
    )]
    fn take(&mut self) -> Fence {
        let fence = self
            .free
            .pop()
            .unwrap_or_else(|| self.device.newFence().expect("Metal creates fences").into());
        self.used.push(fence.clone());
        fence
    }
}

/// Creates command buffers for one queue and frame slot, and lends them
/// fences. Metal command buffers are created per submission, so only the
/// fences are recycled.
pub(crate) struct CommandPool {
    device: Arc<Device>,
    queue: QueueType,
    /// Shared with the pool's command buffers, which take fences from it.
    fences: Arc<Mutex<Fences>>,
}

/// The encoder a command buffer records with. Metal records copies, render
/// passes and dispatches with separate encoders, one at a time.
pub(super) enum Encoder {
    None,
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    Render(Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>),
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
}

impl Encoder {
    /// Makes the encoder's commands wait for `fence`.
    fn wait(&self, fence: &Fence) {
        match self {
            Self::None => {}
            Self::Blit(encoder) => encoder.waitForFence(fence),
            Self::Render(encoder) => encoder.waitForFence_beforeStages(fence, RENDER_STAGES),
            Self::Compute(encoder) => encoder.waitForFence(fence),
        }
    }

    /// Updates `fence` once the encoder's commands complete, and ends it.
    fn end(self, fence: &Fence) {
        match self {
            Self::None => {}
            Self::Blit(encoder) => {
                encoder.updateFence(fence);
                encoder.endEncoding();
            }
            Self::Render(encoder) => {
                encoder.updateFence_afterStages(fence, RENDER_STAGES);
                encoder.endEncoding();
            }
            Self::Compute(encoder) => {
                encoder.updateFence(fence);
                encoder.endEncoding();
            }
        }
    }
}

/// How a command buffer orders its encoders; see the module's
/// documentation.
struct Order {
    fences: Arc<Mutex<Fences>>,
    /// The fences encoders begun from now on wait for: those of the
    /// encoders begun before the last barrier but after the one before it,
    /// or the queue's join fence before the first barrier ending an
    /// encoder.
    waits: Vec<Fence>,
    /// The fences of the encoders begun since the last barrier; the open
    /// encoder updates the last one.
    since: Vec<Fence>,
}

/// What render and compute encoders make resident: every heap, and the
/// resources the bindless heap declares, as of their generations.
#[derive(Default)]
struct Residency {
    /// The generations of the heaps and of the declared resources; zeros,
    /// which neither has, until fetched.
    generations: [u64; 2],
    heaps: Vec<Object<dyn MTLHeap>>,
    /// Alive until the frame closes, which command buffers cannot outlive.
    declared: Declared,
    /// The usage up to which the open encoder made these resident, heaps
    /// included; `None` if it made nothing resident.
    resident: Option<Usage>,
}

/// A command buffer being recorded.
pub(crate) struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Encoder,
    // Boxed, so encoders, which are moved around, stay small.
    order: Box<Order>,
    residency: Box<Residency>,
    /// Primitives the bound pipeline draws.
    pub(super) primitive: MTLPrimitiveType,
    /// The usage the bound graphics pipeline needs resources declared up
    /// to.
    pub(super) usage: Usage,
    /// The bound index buffer, its offset and type.
    pub(super) indices: Option<(Object<dyn MTLBuffer>, u64, MTLIndexType)>,
    /// The compute pipeline bound to the open compute encoder.
    pub(super) compute_pipeline: Option<ComputePipelineId>,
}

impl CommandPool {
    #[expect(
        clippy::unnecessary_wraps,
        reason = "The signature is shared with Vulkan, where this can fail."
    )]
    pub(crate) fn new(device: &Arc<Device>, queue: QueueType) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            queue,
            fences: Arc::new(Mutex::new(Fences {
                device: device.raw.clone(),
                free: Vec::new(),
                used: Vec::new(),
            })),
        })
    }

    /// Recycles the fences of the slot's previous frame, whose work
    /// completed.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "The signature is shared with Vulkan, where this can fail."
    )]
    pub(crate) fn reset(&mut self) -> Result<()> {
        let fences = &mut *self.fences.lock();
        fences.free.append(&mut fences.used);
        Ok(())
    }

    /// Begins recording a command buffer labelled `label`. With validation,
    /// failures report the encoder that failed.
    pub(crate) fn begin(&mut self, label: &str) -> Result<CommandBuffer> {
        let desc = MTLCommandBufferDescriptor::new();
        if self.device.validation {
            desc.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        }
        let queue = &self.device.queues[self.queue as usize];
        let raw = autoreleasepool(|_| queue.raw.commandBufferWithDescriptor(&desc))
            .check("commandBufferWithDescriptor")?;
        raw.setLabel(Some(&NSString::from_str(label)));
        Ok(CommandBuffer {
            raw,
            encoder: Encoder::None,
            order: Box::new(Order {
                fences: self.fences.clone(),
                waits: vec![queue.join.clone()],
                since: Vec::new(),
            }),
            residency: Box::default(),
            primitive: MTLPrimitiveType::Triangle,
            usage: Usage::Read,
            indices: None,
            compute_pipeline: None,
        })
    }
}

impl CommandBuffer {
    /// Ends the open encoder and begins the one `begin` creates: it waits
    /// for the encoders before the last barrier, and updates a fence of its
    /// own when it ends.
    pub(super) fn open(
        &mut self,
        begin: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>) -> Encoder,
    ) {
        self.end_encoding();
        let encoder = autoreleasepool(|_| begin(&self.raw));
        for fence in &self.order.waits {
            encoder.wait(fence);
        }
        self.order.since.push(self.order.fences.lock().take());
        self.residency.resident = None;
        self.encoder = encoder;
    }

    /// Ends the open encoder, if any, updating its fence.
    pub(super) fn end_encoding(&mut self) {
        let encoder = std::mem::replace(&mut self.encoder, Encoder::None);
        if let Some(fence) = self.order.since.last() {
            encoder.end(fence);
        }
        self.compute_pipeline = None;
    }

    /// Orders every encoder begun from now on after every encoder begun so
    /// far, as a barrier: ends the open encoder, and makes later ones wait
    /// for the encoders begun since the last barrier. Those waited for the
    /// ones before, so nothing is lost when there are none.
    pub(super) fn sync(&mut self) {
        self.end_encoding();
        if !self.order.since.is_empty() {
            self.order.waits = std::mem::take(&mut self.order.since);
        }
    }

    /// Ends recording for submission to the queue whose join fence is
    /// `join`: an encoder waiting for every encoder recorded updates it.
    pub(super) fn finish(&mut self, device: &Device, join: &Fence) -> Result<()> {
        self.sync();
        device.join(&self.raw, &self.order.waits, join)
    }

    /// The command buffer, to commit once recording finished.
    pub(super) fn raw(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }

    /// The open blit encoder, beginning one if needed.
    #[expect(
        clippy::expect_used,
        reason = "Metal creates encoders for uncommitted command buffers without an open one."
    )]
    pub(super) fn blit(&mut self) -> &ProtocolObject<dyn MTLBlitCommandEncoder> {
        if !matches!(self.encoder, Encoder::Blit(_)) {
            self.open(|commands| {
                Encoder::Blit(
                    commands
                        .blitCommandEncoder()
                        .expect("Metal creates encoders for command buffers without an open one"),
                )
            });
        }
        match &self.encoder {
            Encoder::Blit(encoder) => encoder,
            // The branch above begins a blit encoder unless one is open; the
            // enum holding it cannot say so.
            _ => unreachable!("a blit encoder was just begun"),
        }
    }

    /// The open compute encoder, beginning one, which binds the heap, if
    /// needed, with every heap resident and resources declared up to
    /// `usage`.
    #[expect(
        clippy::expect_used,
        reason = "Metal creates encoders for uncommitted command buffers without an open one."
    )]
    pub(super) fn compute(
        &mut self,
        device: &Device,
        usage: Usage,
    ) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        if !matches!(self.encoder, Encoder::Compute(_)) {
            self.open(|commands| {
                let encoder = commands
                    .computeCommandEncoder()
                    .expect("Metal creates encoders for command buffers without an open one");
                // SAFETY: the heap outlives the work of every frame.
                unsafe {
                    encoder.setBuffer_offset_atIndex(
                        Some(&device.heap.arguments),
                        0,
                        HEAP_BUFFER as usize,
                    );
                }
                Encoder::Compute(encoder)
            });
        }
        self.make_resident(device, usage);
        match &self.encoder {
            Encoder::Compute(encoder) => encoder,
            // As in `blit`: the branch above begins a compute encoder unless
            // one is open, and making resources resident keeps it open.
            _ => unreachable!("a compute encoder was just begun"),
        }
    }

    /// Begins a render pass with `encoder`, ending the open encoder.
    #[expect(
        clippy::expect_used,
        reason = "Metal creates render encoders for passes the RHI validated."
    )]
    pub(super) fn begin_render(
        &mut self,
        begin: impl FnOnce(
            &ProtocolObject<dyn MTLCommandBuffer>,
        ) -> Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>,
    ) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        self.open(|commands| {
            Encoder::Render(
                begin(commands).expect("Metal creates render encoders for validated passes"),
            )
        });
        self.render()
    }

    /// The open render encoder.
    pub(super) fn render(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        match &self.encoder {
            Encoder::Render(encoder) => encoder,
            // Render commands follow `begin_render`: those of a pass go
            // through its `RenderPass`, which borrows the encoder until the
            // pass ends, so no other command ends the render encoder first.
            _ => unreachable!("render pass commands are recorded inside a render pass"),
        }
    }

    /// Makes every heap, and the resources the bindless heap declares up to
    /// `usage`, resident for the open render or compute encoder, unless it
    /// did since they last changed: before every draw and dispatch, since
    /// shaders reach resources created at any time through the bindless
    /// heap. Costs two atomic loads when nothing changed.
    pub(super) fn make_resident(&mut self, device: &Device, usage: Usage) {
        let generations = [device.allocator.generation(), device.heap.generation()];
        if self.residency.generations != generations {
            let (heaps, (declared_generation, declared)) =
                (device.allocator.heaps(), device.heap.declared());
            *self.residency = Residency {
                generations: [heaps.0, declared_generation],
                heaps: heaps.1,
                declared,
                resident: None,
            };
        }
        let resident = self.residency.resident;
        if resident >= Some(usage) {
            return;
        }
        let Residency {
            heaps, declared, ..
        } = &*self.residency;
        // The heaps, unless resident already, and the usages not declared
        // yet.
        let heaps = if resident.is_none() { &heaps[..] } else { &[] };
        let usages = Usage::ALL
            .into_iter()
            .filter(|declare| Some(*declare) > resident && *declare <= usage);
        match &self.encoder {
            Encoder::Render(encoder) => {
                for heap in heaps {
                    encoder.useHeap_stages(heap, RENDER_STAGES);
                }
                Heap::declare(declared, usages, |resources, count, usage| {
                    // SAFETY: `declared` holds live resources; see
                    // `Residency`.
                    unsafe {
                        encoder.useResources_count_usage_stages(
                            resources,
                            count,
                            usage,
                            RENDER_STAGES,
                        );
                    }
                });
            }
            Encoder::Compute(encoder) => {
                for heap in heaps {
                    encoder.useHeap(heap);
                }
                Heap::declare(declared, usages, |resources, count, usage| {
                    // SAFETY: as above.
                    unsafe { encoder.useResources_count_usage(resources, count, usage) };
                });
            }
            // No other encoder reads the bindless heap, so needs nothing
            // resident.
            Encoder::None | Encoder::Blit(_) => return,
        }
        self.residency.resident = Some(usage);
    }
}

impl Drop for CommandBuffer {
    /// Metal requires encoders to end, even in discarded command buffers.
    fn drop(&mut self) {
        self.end_encoding();
    }
}

impl Device {
    /// Encodes into `commands` an encoder waiting for `waits` and updating
    /// `join`, a queue's join fence, which the queue's later command
    /// buffers wait for. It fills the scratch buffer, since Metal may skip
    /// encoders without commands.
    pub(super) fn join(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        waits: &[Fence],
        join: &Fence,
    ) -> Result<()> {
        let encoder =
            autoreleasepool(|_| commands.blitCommandEncoder()).check("blitCommandEncoder")?;
        encoder.setLabel(Some(&NSString::from_str("join")));
        for fence in waits {
            encoder.waitForFence(fence);
        }
        encoder.fillBuffer_range_value(&self.scratch, NSRange::new(0, self.scratch.length()), 0);
        encoder.updateFence(join);
        encoder.endEncoding();
        Ok(())
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

    /// Generates every mip of `image` from mip 0, in an encoder of its own
    /// between barriers, as the image's moves from rest and back would be.
    /// Metal orders the levels, and rejects images of one mip, which have
    /// none to generate.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "The signature is shared with Vulkan, whose blits can fail."
    )]
    pub(crate) fn generate_mips(&mut self, _device: &Device, image: &crate::Image) -> Result<()> {
        if image.info().mip_levels > 1 {
            self.sync();
            self.blit().generateMipmapsForTexture(&image.raw.raw);
            self.sync();
        }
        Ok(())
    }

    /// Orders the commands after the barrier after those before it, with
    /// fences; see the module's documentation. Metal textures have no
    /// layouts and its queues need no ownership transfers, so the accesses
    /// themselves do not matter.
    pub(crate) fn barrier(
        &mut self,
        _device: &Device,
        _queue: QueueType,
        _buffers: &[BufferBarrier<'_>],
        _images: &[Transition],
    ) {
        self.sync();
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
