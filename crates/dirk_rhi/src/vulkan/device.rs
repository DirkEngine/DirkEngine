use std::collections::BTreeSet;

use ash::vk;
use tracing::{error, info};

use super::{
    Check,
    adapter::{Adapter, RequiredFeatures},
    instance::Instance,
};
use crate::{Capabilities, Features, Result, RhiCreateInfo};

/// The logical device, shared by the RHI and every object created from it.
pub(crate) struct Device {
    pub(crate) capabilities: Capabilities,
    raw: ash::Device,
    /// Destroyed after the device, which `Drop` destroys first.
    instance: Instance,
}

impl Device {
    /// Creates an instance, selects an adapter and creates its device with
    /// every required feature and the optional ones it supports.
    pub(crate) fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        let instance = Instance::new(info)?;
        let adapter = Adapter::select(&instance)?;
        let capabilities = adapter.capabilities;
        info!(device = %capabilities.name, "selected Vulkan device");

        let priorities = [1.0];
        let queues = BTreeSet::from([adapter.families.graphics, adapter.families.copy])
            .into_iter()
            .map(|family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
            })
            .collect::<Vec<_>>();
        let optional = capabilities.features;
        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(true)
            .sampler_anisotropy(optional.contains(Features::SAMPLER_ANISOTROPY))
            .image_cube_array(optional.contains(Features::IMAGE_CUBE_ARRAY))
            .independent_blend(optional.contains(Features::INDEPENDENT_BLEND))
            .depth_bias_clamp(optional.contains(Features::DEPTH_BIAS_CLAMP));
        let mut required = RequiredFeatures::enabled();
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&adapter.extensions)
            .enabled_features(&features)
            .push_next(&mut required.vulkan12)
            .push_next(&mut required.vulkan13);
        // SAFETY: the adapter belongs to `instance`, and every pointer in
        // `create_info` outlives the call.
        let raw = unsafe { instance.raw.create_device(adapter.raw, &create_info, None) }
            .check("vkCreateDevice")?;
        Ok(Self {
            capabilities,
            raw,
            instance,
        })
    }

    /// Number of errors validation reported, or `None` if it does not run.
    pub(crate) fn validation_error_count(&self) -> Option<usize> {
        self.instance.validation_errors()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every object created from the device has been dropped, since
        // each holds the device alive; waiting makes their GPU work finish.
        unsafe {
            if let Err(error) = self.raw.device_wait_idle() {
                error!(%error, "failed to wait for the Vulkan device before destroying it");
            }
            self.raw.destroy_device(None);
        }
    }
}
