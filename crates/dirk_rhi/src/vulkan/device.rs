use std::{collections::BTreeSet, ffi::CString, mem::ManuallyDrop};

use ash::{ext::debug_utils, vk};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;
use tracing::{error, info, warn};

use super::{
    Check,
    adapter::{Adapter, RequiredFeatures},
    instance::Instance,
};
use crate::{Capabilities, Error, Features, Result, RhiCreateInfo};

/// The logical device, shared by the RHI and every object created from it.
pub(crate) struct Device {
    pub(crate) capabilities: Capabilities,
    pub(super) raw: ash::Device,
    /// Size non-coherent host memory is flushed and invalidated in.
    pub(super) non_coherent_atom_size: u64,
    /// Locked only to allocate and free memory, which `gpu-allocator` needs
    /// exclusive access for. Dropped by `Drop` before the device.
    allocator: Mutex<ManuallyDrop<Allocator>>,
    /// Names objects for debugging tools when validation is enabled.
    debug: Option<debug_utils::Device>,
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
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw.clone(),
            device: raw.clone(),
            physical_device: adapter.raw,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: AllocationSizes::default(),
        });
        let allocator = match allocator {
            Ok(allocator) => allocator,
            Err(error) => {
                // SAFETY: nothing has been created from the device yet.
                unsafe { raw.destroy_device(None) };
                return Err(Error::native("creating the memory allocator", error));
            }
        };
        Ok(Self {
            capabilities,
            debug: instance
                .debug_utils()
                .then(|| debug_utils::Device::new(&instance.raw, &raw)),
            raw,
            non_coherent_atom_size: adapter.non_coherent_atom_size,
            allocator: Mutex::new(ManuallyDrop::new(allocator)),
            instance,
        })
    }

    /// Allocates memory for a resource labelled `label`.
    pub(super) fn allocate(
        &self,
        label: &str,
        requirements: vk::MemoryRequirements,
        location: MemoryLocation,
        linear: bool,
    ) -> Result<Allocation> {
        self.allocator
            .lock()
            .allocate(&AllocationCreateDesc {
                name: label,
                requirements,
                location,
                linear,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|error| Error::native("allocating memory", error))
    }

    /// Frees `allocation`. Called from `Drop`, so failures are logged.
    pub(super) fn free(&self, allocation: Allocation) {
        if let Err(error) = self.allocator.lock().free(allocation) {
            error!(%error, "failed to free GPU memory");
        }
    }

    /// Names `object` `label` for debugging tools, when validation is enabled.
    pub(super) fn name(&self, object: impl vk::Handle, label: &str) {
        let Some(debug) = &self.debug else { return };
        let Ok(name) = CString::new(label) else {
            warn!(
                label,
                "cannot name a Vulkan object with a NUL byte in its label"
            );
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(&name);
        // SAFETY: `object` was created from this device.
        if let Err(error) = unsafe { debug.set_debug_utils_object_name(&info) } {
            warn!(%error, label, "failed to name a Vulkan object");
        }
    }

    /// Number of errors validation reported.
    pub(crate) fn validation_error_count(&self) -> usize {
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
            // The allocator frees its memory through the device.
            ManuallyDrop::drop(self.allocator.get_mut());
            self.raw.destroy_device(None);
        }
    }
}
