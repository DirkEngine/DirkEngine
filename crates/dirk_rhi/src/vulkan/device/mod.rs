use std::{collections::BTreeSet, ffi::CString, mem::ManuallyDrop};

use ash::{ext::debug_utils, vk};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;
use tracing::{error, info, warn};

use super::{
    Check, ImageId,
    adapter::{Adapter, RequiredFeatures},
    instance::Instance,
    queue::Queue,
};
use crate::{
    Capabilities, Error, Features, QueueType, Result, RhiCreateInfo, command::Transition,
    graveyard::Graveyard,
};

/// A native object whose owner dropped it, destroyed once the GPU is done.
pub(super) enum Garbage {
    Buffer(vk::Buffer, Allocation),
    Image {
        raw: vk::Image,
        view: Option<vk::ImageView>,
        allocation: Allocation,
    },
    Sampler(vk::Sampler),
}

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
    /// Indexed by `QueueType`.
    pub(super) queues: [Queue; QueueType::COUNT],
    /// The Vulkan queues submitted to, each locked while submitting.
    pub(super) vk_queues: [Mutex<vk::Queue>; QueueType::COUNT],
    graveyard: Graveyard<Garbage>,
    /// Transitions of new images from undefined contents to their resting
    /// access, recorded before the next graphics submission.
    initializations: Mutex<Vec<Transition>>,
    /// Images dropped since the last frame began, whose first transitions
    /// are forgotten when the next one begins, or as they are destroyed if
    /// that comes first.
    dropped_images: Mutex<Vec<ImageId>>,
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
        })
        .map_err(|error| Error::native("creating the memory allocator", error))
        .and_then(|allocator| Ok((allocator, Queue::create(&raw, adapter.families)?)));
        let (allocator, queues) = match allocator {
            Ok(parts) => parts,
            Err(error) => {
                // SAFETY: nothing created from the device is left.
                unsafe { raw.destroy_device(None) };
                return Err(error);
            }
        };
        // SAFETY: both families were requested with one queue each.
        let vk_queues = [adapter.families.graphics, adapter.families.copy]
            .map(|family| Mutex::new(unsafe { raw.get_device_queue(family, 0) }));
        Ok(Self {
            capabilities,
            debug: instance
                .debug_utils()
                .then(|| debug_utils::Device::new(&instance.raw, &raw)),
            raw,
            non_coherent_atom_size: adapter.non_coherent_atom_size,
            allocator: Mutex::new(ManuallyDrop::new(allocator)),
            queues,
            vk_queues,
            graveyard: Graveyard::new(),
            initializations: Mutex::default(),
            dropped_images: Mutex::default(),
            instance,
        })
    }

    /// Records `transition`, a new image's first, before the next graphics
    /// submission.
    pub(crate) fn initialize(&self, transition: Transition) {
        self.initializations.lock().push(transition);
    }

    /// The image transitions to record before the next graphics submission.
    pub(crate) fn take_initializations(&self) -> Vec<Transition> {
        std::mem::take(&mut *self.initializations.lock())
    }

    /// Drops `image`'s first transition, if it was not recorded yet: a
    /// submitted copy encoder took it over, or the image is destroyed.
    pub(crate) fn forget_initialization(&self, image: ImageId) {
        self.initializations
            .lock()
            .retain(|transition| transition.image != image);
    }

    /// Forgets the dropped `image`'s first transition, if still queued,
    /// when the next frame begins, or as the image is destroyed if that
    /// comes first. Until then, commands recorded before the drop may
    /// expect it. Call it before burying the image, so that an image a
    /// frame beginning on another thread misses is tagged with that frame
    /// (see [`begin_frame`](Self::begin_frame)).
    pub(super) fn forget_initialization_next_frame(&self, image: ImageId) {
        self.dropped_images.lock().push(image);
    }

    /// Keeps `garbage` until the work of the current frame completes.
    pub(super) fn bury(&self, garbage: Garbage) {
        self.graveyard.bury(garbage);
    }

    /// Tags objects dropped from now on with `frame`, then forgets the first
    /// transitions of the images dropped before: encoders cannot outlive
    /// their frame, so no command recorded from now on expects them.
    ///
    /// Tagging first matters when images drop on another thread meanwhile:
    /// an image dropped too late to be forgotten here was buried after the
    /// tag moved, so it is tagged with `frame` and outlives the work of
    /// `frame`, which may still record its transition.
    pub(crate) fn begin_frame(&self, frame: u64) {
        self.graveyard.begin_frame(frame);
        let dropped = std::mem::take(&mut *self.dropped_images.lock());
        self.initializations
            .lock()
            .retain(|transition| !dropped.contains(&transition.image));
    }

    /// Destroys every object dropped by the end of frame `completed`, whose
    /// work, and all work before it, has completed.
    pub(crate) fn collect(&self, completed: u64) {
        self.graveyard
            .collect(completed, |garbage| self.destroy(garbage));
    }

    /// The queue family `queue` submits to.
    pub(super) fn family(&self, queue: QueueType) -> u32 {
        self.queues[queue as usize].family
    }

    /// Destroys `garbage`, which the GPU no longer uses.
    fn destroy(&self, garbage: Garbage) {
        // SAFETY: the graveyard only hands out objects whose work completed.
        unsafe {
            match garbage {
                Garbage::Buffer(raw, allocation) => {
                    self.raw.destroy_buffer(raw, None);
                    self.free(allocation);
                }
                Garbage::Image {
                    raw,
                    view,
                    allocation,
                } => {
                    // Before the driver can reuse the handle, so a new image
                    // never matches what is left of this one.
                    self.dropped_images.lock().retain(|dropped| *dropped != raw);
                    self.forget_initialization(raw);
                    if let Some(view) = view {
                        self.raw.destroy_image_view(view, None);
                    }
                    self.raw.destroy_image(raw, None);
                    self.free(allocation);
                }
                Garbage::Sampler(raw) => self.raw.destroy_sampler(raw, None),
            }
        }
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
            self.collect(u64::MAX);
            for queue in &self.queues {
                queue.destroy(&self.raw);
            }
            // The allocator frees its memory through the device.
            ManuallyDrop::drop(self.allocator.get_mut());
            self.raw.destroy_device(None);
        }
    }
}

#[cfg(test)]
mod tests;
