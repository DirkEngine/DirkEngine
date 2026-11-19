use std::{ops::Range, sync::Arc};

use ash::vk;

use super::{Buffer, Check, Device, Image, presentation::PRESENT_STAGES};
use crate::{
    BufferAccess, BufferBarrier, BufferCopy, BufferImageCopy, Filter, Format, ImageAccess,
    ImageBarrier, ImageCopy, ImageRange, ImageRegion, QueueTransfer, QueueType, Result,
    ShaderStages,
};

/// Command buffers for one queue and frame slot. Reset as a whole when the
/// slot is reused, after its previous frame's work completed.
pub(crate) struct CommandPool {
    device: Arc<Device>,
    raw: vk::CommandPool,
    /// Every command buffer allocated so far, reused after a reset.
    buffers: Vec<vk::CommandBuffer>,
    /// How many of `buffers` the current frame has begun.
    used: usize,
}

/// A command buffer in the recording state.
#[derive(Clone, Copy)]
pub(crate) struct CommandBuffer {
    pub(crate) raw: vk::CommandBuffer,
}

impl CommandPool {
    /// Creates a pool for `queue`'s family.
    pub(crate) fn new(device: &Arc<Device>, queue: QueueType) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.family(queue));
        // SAFETY: `info` is valid for the device.
        let raw =
            unsafe { device.raw.create_command_pool(&info, None) }.check("vkCreateCommandPool")?;
        Ok(Self {
            device: device.clone(),
            raw,
            buffers: Vec::new(),
            used: 0,
        })
    }

    /// Resets every command buffer for reuse.
    ///
    /// The work of every command buffer begun since the last reset must have
    /// completed, which beginning the slot's next frame guarantees.
    pub(crate) fn reset(&mut self) -> Result<()> {
        // SAFETY: the frame slot's previous work has completed.
        unsafe {
            self.device
                .raw
                .reset_command_pool(self.raw, vk::CommandPoolResetFlags::empty())
        }
        .check("vkResetCommandPool")?;
        self.used = 0;
        Ok(())
    }

    /// Begins recording a command buffer labelled `label`.
    pub(crate) fn begin(&mut self, label: &str) -> Result<CommandBuffer> {
        if self.used == self.buffers.len() {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.raw)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool is only used by the open frame.
            let allocated = unsafe { self.device.raw.allocate_command_buffers(&info) }
                .check("vkAllocateCommandBuffers")?;
            self.buffers.extend(allocated);
        }
        let raw = self.buffers[self.used];
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the buffer was reset with its pool and is not in use.
        unsafe { self.device.raw.begin_command_buffer(raw, &info) }
            .check("vkBeginCommandBuffer")?;
        self.used += 1;
        self.device.name(raw, label);
        Ok(CommandBuffer { raw })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        // SAFETY: pools are dropped with the RHI, after its work completed.
        unsafe { self.device.raw.destroy_command_pool(self.raw, None) };
    }
}

// Recording. Inputs were validated by the encoder, and every resource is from
// `device` and outlives the frame's work thanks to deferred destruction.
impl CommandBuffer {
    pub(crate) fn copy_buffer(
        self,
        device: &Device,
        src: &Buffer,
        dst: &Buffer,
        regions: &[BufferCopy],
    ) {
        let regions: Vec<_> = regions
            .iter()
            .map(|region| vk::BufferCopy {
                src_offset: region.src_offset,
                dst_offset: region.dst_offset,
                size: region.size,
            })
            .collect();
        // SAFETY: see the recording note above.
        unsafe {
            device
                .raw
                .cmd_copy_buffer(self.raw, src.raw, dst.raw, &regions);
        }
    }

    pub(crate) fn copy_buffer_to_image(
        self,
        device: &Device,
        src: &Buffer,
        dst: &Image,
        format: Format,
        regions: &[BufferImageCopy],
    ) {
        let regions = Self::buffer_image_regions(format, regions);
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_copy_buffer_to_image(
                self.raw,
                src.raw,
                dst.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    pub(crate) fn copy_image_to_buffer(
        self,
        device: &Device,
        src: &Image,
        dst: &Buffer,
        format: Format,
        regions: &[BufferImageCopy],
    ) {
        let regions = Self::buffer_image_regions(format, regions);
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_copy_image_to_buffer(
                self.raw,
                src.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.raw,
                &regions,
            );
        }
    }

    pub(crate) fn copy_image(
        self,
        device: &Device,
        src: &Image,
        dst: &Image,
        format: Format,
        regions: &[ImageCopy],
    ) {
        let aspects = format.vk_aspects();
        let regions: Vec<_> = regions
            .iter()
            .map(|region| vk::ImageCopy {
                src_subresource: region.src.vk_layers(aspects),
                src_offset: region.src.vk_start(),
                dst_subresource: region.dst.vk_layers(aspects),
                dst_offset: region.dst.vk_start(),
                extent: vk::Extent3D {
                    width: region.src.extent.width,
                    height: region.src.extent.height,
                    depth: region.src.extent.depth,
                },
            })
            .collect();
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_copy_image(
                self.raw,
                src.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Metal, where this can fail"
    )]
    pub(crate) fn blit_image(
        self,
        device: &Device,
        src: &Image,
        dst: &Image,
        format: Format,
        regions: &[ImageCopy],
        filter: Filter,
    ) -> Result<()> {
        let aspects = format.vk_aspects();
        let regions: Vec<_> = regions
            .iter()
            .map(|region| vk::ImageBlit {
                src_subresource: region.src.vk_layers(aspects),
                src_offsets: [region.src.vk_start(), region.src.vk_end()],
                dst_subresource: region.dst.vk_layers(aspects),
                dst_offsets: [region.dst.vk_start(), region.dst.vk_end()],
            })
            .collect();
        // SAFETY: see the recording note above.
        unsafe {
            device.raw.cmd_blit_image(
                self.raw,
                src.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
                filter.vk(),
            );
        }
        Ok(())
    }

    /// Generates every mip of `image` from mip 0, blitting each level into
    /// the next once the previous blit wrote it, and moves the whole image
    /// from `before` to `after`.
    pub(crate) fn generate_mips(
        self,
        device: &Device,
        image: &crate::Image,
        before: ImageAccess,
        after: ImageAccess,
    ) -> Result<()> {
        let info = image.info();
        let transition = |levels: Range<u32>, before, after| {
            let range = ImageRange {
                base_mip_level: levels.start,
                mip_level_count: levels.end - levels.start,
                ..ImageRange::ALL
            };
            let barrier = ImageBarrier {
                image,
                range,
                before,
                after,
                transfer: None,
            };
            self.barrier(device, QueueType::Graphics, &[], &[barrier], &[range]);
        };
        transition(0..1, before, ImageAccess::CopySource);
        if info.mip_levels > 1 {
            transition(1..info.mip_levels, before, ImageAccess::CopyDestination);
        }
        for level in 1..info.mip_levels {
            let region = ImageCopy {
                src: info.mip_region(level - 1),
                dst: info.mip_region(level),
            };
            let raw = &image.raw;
            self.blit_image(device, raw, raw, info.format, &[region], Filter::Linear)?;
            transition(
                level..level + 1,
                ImageAccess::CopyDestination,
                ImageAccess::CopySource,
            );
        }
        transition(0..info.mip_levels, ImageAccess::CopySource, after);
        Ok(())
    }

    /// Records `buffers` and `images` as one dependency, recorded on
    /// `queue`. `ranges` are the images' resolved ranges.
    pub(crate) fn barrier(
        self,
        device: &Device,
        queue: QueueType,
        buffers: &[BufferBarrier<'_>],
        images: &[ImageBarrier<'_>],
        ranges: &[ImageRange],
    ) {
        let buffers: Vec<_> = buffers
            .iter()
            .map(|barrier| {
                let scope = Scope::new(
                    device,
                    queue,
                    barrier.transfer,
                    barrier.before.vk(),
                    barrier.after.vk(),
                );
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(scope.src.0)
                    .src_access_mask(scope.src.1)
                    .dst_stage_mask(scope.dst.0)
                    .dst_access_mask(scope.dst.1)
                    .src_queue_family_index(scope.families.0)
                    .dst_queue_family_index(scope.families.1)
                    .buffer(barrier.buffer.raw.raw)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        let images: Vec<_> = images
            .iter()
            .zip(ranges)
            .map(|(barrier, range)| {
                let (before_stage, before_access, old_layout) = barrier.before.vk();
                let (after_stage, after_access, new_layout) = barrier.after.vk();
                let scope = Scope::new(
                    device,
                    queue,
                    barrier.transfer,
                    (before_stage, before_access),
                    (after_stage, after_access),
                );
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(scope.src.0)
                    .src_access_mask(scope.src.1)
                    .dst_stage_mask(scope.dst.0)
                    .dst_access_mask(scope.dst.1)
                    // Without a family change, the release half already
                    // changed the layout. Acquired images are discarded.
                    .old_layout(if scope.acquires_in_place {
                        new_layout
                    } else if barrier.before == ImageAccess::Present {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        old_layout
                    })
                    .new_layout(new_layout)
                    .src_queue_family_index(scope.families.0)
                    .dst_queue_family_index(scope.families.1)
                    .image(barrier.image.raw.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: barrier.image.info().format.vk_aspects(),
                        base_mip_level: range.base_mip_level,
                        level_count: range.mip_level_count,
                        base_array_layer: range.base_layer,
                        layer_count: range.layer_count,
                    })
            })
            .collect();
        let dependency = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffers)
            .image_memory_barriers(&images);
        // SAFETY: see the recording note above.
        unsafe { device.raw.cmd_pipeline_barrier2(self.raw, &dependency) };
    }

    /// Vulkan regions for buffer-image copies of a color `format`.
    fn buffer_image_regions(
        format: Format,
        regions: &[BufferImageCopy],
    ) -> Vec<vk::BufferImageCopy> {
        regions
            .iter()
            .map(|region| vk::BufferImageCopy {
                buffer_offset: region.buffer_offset,
                buffer_row_length: region.bytes_per_row / format.bytes_per_texel(),
                buffer_image_height: region.rows_per_image,
                image_subresource: region.image.vk_layers(vk::ImageAspectFlags::COLOR),
                image_offset: region.image.vk_start(),
                image_extent: vk::Extent3D {
                    width: region.image.extent.width,
                    height: region.image.extent.height,
                    depth: region.image.extent.depth,
                },
            })
            .collect()
    }
}

/// The half of a dependency this queue performs, and the queue families of
/// an ownership transfer.
struct Scope {
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
    families: (u32, u32),
    /// The acquire half of a transfer between queues of one family, whose
    /// release half already did the work.
    acquires_in_place: bool,
}

impl Scope {
    /// The scope of a barrier from `before` to `after` recorded on `queue`.
    /// A transfer's source queue only releases and its destination only
    /// acquires; between queues of one family no ownership moves.
    fn new(
        device: &Device,
        queue: QueueType,
        transfer: Option<QueueTransfer>,
        before: (vk::PipelineStageFlags2, vk::AccessFlags2),
        after: (vk::PipelineStageFlags2, vk::AccessFlags2),
    ) -> Self {
        let none = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
        let Some(QueueTransfer { from, to }) = transfer else {
            return Self {
                src: before,
                dst: after,
                families: (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                acquires_in_place: false,
            };
        };
        let families = (device.family(from), device.family(to));
        let same_family = families.0 == families.1;
        Self {
            src: if queue == to { none } else { before },
            dst: if queue == from { none } else { after },
            families: if same_family {
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            } else {
                families
            },
            acquires_in_place: same_family && queue == to,
        }
    }
}

impl ImageRegion {
    /// The layers of this region with `aspects`.
    fn vk_layers(&self, aspects: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspects,
            mip_level: self.mip_level,
            base_array_layer: self.base_layer,
            layer_count: self.layer_count,
        }
    }

    /// The region's first texel.
    fn vk_start(&self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.origin.x.cast_signed(),
            y: self.origin.y.cast_signed(),
            z: self.origin.z.cast_signed(),
        }
    }

    /// The corner past the region's last texel.
    fn vk_end(&self) -> vk::Offset3D {
        vk::Offset3D {
            x: (self.origin.x + self.extent.width).cast_signed(),
            y: (self.origin.y + self.extent.height).cast_signed(),
            z: (self.origin.z + self.extent.depth).cast_signed(),
        }
    }
}

impl ShaderStages {
    /// The pipeline stages running these shaders.
    fn vk(self) -> vk::PipelineStageFlags2 {
        [
            (Self::VERTEX, vk::PipelineStageFlags2::VERTEX_SHADER),
            (Self::FRAGMENT, vk::PipelineStageFlags2::FRAGMENT_SHADER),
            (Self::COMPUTE, vk::PipelineStageFlags2::COMPUTE_SHADER),
        ]
        .into_iter()
        .filter(|(stage, _)| self.contains(*stage))
        .fold(vk::PipelineStageFlags2::NONE, |all, (_, stage)| all | stage)
    }
}

impl BufferAccess {
    /// The stages and memory accesses of this access.
    fn vk(self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        match self {
            Self::None => (S::NONE, A::NONE),
            Self::CopySource => (S::COPY, A::TRANSFER_READ),
            Self::CopyDestination => (S::COPY, A::TRANSFER_WRITE),
            Self::Index => (S::INDEX_INPUT, A::INDEX_READ),
            Self::Indirect => (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ),
            Self::ShaderRead(stages) => (stages.vk(), A::SHADER_STORAGE_READ),
            Self::ShaderWrite(stages) => (
                stages.vk(),
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            ),
            Self::HostRead => (S::HOST, A::HOST_READ),
        }
    }
}

impl ImageAccess {
    /// The stages, memory accesses and layout of this access.
    fn vk(self) -> (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout) {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let fragment_tests = S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS;
        match self {
            Self::Undefined => (S::NONE, A::NONE, L::UNDEFINED),
            Self::CopySource => (S::COPY | S::BLIT, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
            Self::CopyDestination => (
                S::COPY | S::BLIT,
                A::TRANSFER_WRITE,
                L::TRANSFER_DST_OPTIMAL,
            ),
            // One read-only layout for sampling and depth reads, which the
            // heap's sampled descriptors name.
            Self::Sampled(stages) => (stages.vk(), A::SHADER_SAMPLED_READ, L::READ_ONLY_OPTIMAL),
            Self::StorageRead(stages) => (stages.vk(), A::SHADER_STORAGE_READ, L::GENERAL),
            Self::StorageWrite(stages) => (
                stages.vk(),
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
                L::GENERAL,
            ),
            Self::ColorAttachment => (
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
                L::COLOR_ATTACHMENT_OPTIMAL,
            ),
            Self::DepthStencilAttachment => (
                fragment_tests,
                A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            Self::DepthStencilRead(stages) => (
                fragment_tests | stages.vk(),
                if stages.is_empty() {
                    A::DEPTH_STENCIL_ATTACHMENT_READ
                } else {
                    A::DEPTH_STENCIL_ATTACHMENT_READ | A::SHADER_SAMPLED_READ
                },
                L::READ_ONLY_OPTIMAL,
            ),
            // Ordered after acquiring by the acquire semaphore's wait, and
            // before presenting by the present semaphore's signal.
            Self::Present => (PRESENT_STAGES, A::NONE, L::PRESENT_SRC_KHR),
        }
    }
}
