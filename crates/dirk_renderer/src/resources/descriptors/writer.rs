//! Batched descriptor set writes.

use ash::vk;

use crate::resources::descriptors::layouts::SetLayout;

use super::set::DescriptorSet;

enum WriteOp {
    UniformBuffer {
        set: vk::DescriptorSet,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    },
    CombinedImageSampler {
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    },
}

/// Collects descriptor writes and submits them in one Vulkan call.
pub struct DescriptorWriter<'dev> {
    device: &'dev ash::Device,
    ops: Vec<WriteOp>,
}

impl<'dev> DescriptorWriter<'dev> {
    /// Creates an empty writer.
    pub fn new(device: &'dev ash::Device) -> Self {
        Self {
            device,
            ops: Vec::new(),
        }
    }

    /// Adds a uniform-buffer descriptor write.
    pub fn uniform_buffer<L: SetLayout>(
        mut self,
        set: &DescriptorSet<L>,
        binding: u32,
        buffer: vk::Buffer,
        range: vk::DeviceSize,
    ) -> Self {
        debug_assert_binding::<L>(binding, vk::DescriptorType::UNIFORM_BUFFER);
        self.ops.push(WriteOp::UniformBuffer {
            set: set.raw(),
            binding,
            buffer,
            offset: 0,
            range,
        });
        self
    }

    /// Adds a combined image sampler descriptor write.
    pub fn combined_image_sampler<L: SetLayout>(
        mut self,
        set: &DescriptorSet<L>,
        binding: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Self {
        debug_assert_binding::<L>(binding, vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        self.ops.push(WriteOp::CombinedImageSampler {
            set: set.raw(),
            binding,
            view,
            sampler,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        });
        self
    }

    /// Submits all pending writes.
    pub fn flush(self) {
        if self.ops.is_empty() {
            return;
        }

        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();

        for op in &self.ops {
            match op {
                WriteOp::UniformBuffer {
                    buffer,
                    offset,
                    range,
                    ..
                } => {
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(*buffer)
                            .offset(*offset)
                            .range(*range),
                    );
                }
                WriteOp::CombinedImageSampler {
                    view,
                    sampler,
                    image_layout,
                    ..
                } => {
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_layout(*image_layout)
                            .image_view(*view)
                            .sampler(*sampler),
                    );
                }
            }
        }

        let mut buffer_index = 0usize;
        let mut image_index = 0usize;
        let writes = self
            .ops
            .iter()
            .map(|op| match op {
                WriteOp::UniformBuffer { set, binding, .. } => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_infos[buffer_index..=buffer_index]);
                    buffer_index += 1;
                    write
                }
                WriteOp::CombinedImageSampler { set, binding, .. } => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_infos[image_index..=image_index]);
                    image_index += 1;
                    write
                }
            })
            .collect::<Vec<_>>();

        // SAFETY: each write points at one entry of `buffer_infos` or
        // `image_infos`, which outlive the call, and its descriptor type
        // matches the layout binding (asserted in debug builds). Every caller
        // writes sets it has just allocated, which no command buffer uses yet.
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }
}

fn debug_assert_binding<L: SetLayout>(binding: u32, descriptor_type: vk::DescriptorType) {
    if cfg!(debug_assertions) {
        let layout_binding = L::BINDINGS
            .iter()
            .find(|layout_binding| layout_binding.binding == binding);
        debug_assert!(
            layout_binding.is_some(),
            "descriptor binding {binding} does not exist in layout"
        );
        let Some(layout_binding) = layout_binding else {
            return;
        };

        debug_assert_eq!(
            layout_binding.descriptor_type, descriptor_type,
            "descriptor binding {binding} has incompatible descriptor type"
        );
        debug_assert!(
            layout_binding.descriptor_count >= 1,
            "descriptor binding {binding} must have at least one descriptor"
        );
    }
}
