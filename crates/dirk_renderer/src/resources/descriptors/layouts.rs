use std::collections::HashMap;

use ash::vk;

use crate::Result;

pub trait SetLayout {
    const BINDINGS: &'static [vk::DescriptorSetLayoutBinding<'static>];

    // TODO: setup RAII
    fn create_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(Self::BINDINGS);

        // SAFETY: `info` points at the static `BINDINGS`.
        Ok(unsafe { device.create_descriptor_set_layout(&info, None)? })
    }

    fn pool_sizes(max_sets: u32) -> Vec<vk::DescriptorPoolSize> {
        let mut counts: HashMap<vk::DescriptorType, u32> = HashMap::new();

        for binding in Self::BINDINGS {
            *counts.entry(binding.descriptor_type).or_insert(0) += binding.descriptor_count;
        }

        counts
            .into_iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: count.saturating_mul(max_sets).max(1),
            })
            .collect()
    }
}
