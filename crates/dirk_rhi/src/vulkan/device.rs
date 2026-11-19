use std::{collections::BTreeSet, ffi::CString, mem::ManuallyDrop, sync::Arc};

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
    heap::{Heap, Slot},
    instance::Instance,
    queue::Queue,
};
use crate::{
    Capabilities, Error, Features, QueueType, Result, RhiCreateInfo, SurfaceTarget,
    command::Transition,
    graveyard::{Buried, Graveyard},
    heap::Reads,
};

/// A native object whose owner dropped it, destroyed once the GPU is done.
/// Heap slots are emptied before the objects they describe are destroyed.
pub(super) enum Garbage {
    Buffer {
        raw: vk::Buffer,
        allocation: Allocation,
        slot: Option<Slot>,
    },
    Image {
        /// `None` for swapchain images, which their swapchain owns.
        raw: Option<vk::Image>,
        view: Option<vk::ImageView>,
        allocation: Allocation,
        slots: [Option<Slot>; 2],
    },
    Sampler(vk::Sampler, Slot),
    Pipeline(vk::Pipeline),
    /// A retired swapchain and the semaphores of its presentation.
    Swapchain(vk::SwapchainKHR, Vec<vk::Semaphore>),
    /// A surface and the window it presents to, kept alive until now.
    Surface(vk::SurfaceKHR, Arc<dyn SurfaceTarget>),
}

impl Buried for Garbage {
    fn slots(&self) -> impl Iterator<Item = Slot> {
        let slots = match self {
            Self::Buffer { slot, .. } => [*slot, None],
            Self::Image { slots, .. } => *slots,
            Self::Sampler(_, slot) => [Some(*slot), None],
            Self::Pipeline(_) | Self::Swapchain(..) | Self::Surface(..) => [None; 2],
        };
        slots.into_iter().flatten()
    }
}

/// The logical device, shared by the RHI and every object created from it.
pub(crate) struct Device {
    pub(crate) capabilities: Capabilities,
    pub(super) raw: ash::Device,
    pub(super) physical: vk::PhysicalDevice,
    /// Swapchain functions, with presentation.
    pub(super) swapchains: Option<ash::khr::swapchain::Device>,
    /// The bindless heap and the pipeline layout every pipeline shares.
    pub(super) heap: Heap,
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
    /// Dropped objects, and the last frame whose work read each heap slot.
    pub(super) graveyard: Graveyard<Garbage>,
    /// Transitions of new images from undefined contents to their resting
    /// access, recorded before the next graphics submission.
    initializations: Mutex<Vec<Transition>>,
    /// Destroyed after the device, which `Drop` destroys first.
    pub(super) instance: Instance,
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
            .vertex_pipeline_stores_and_atomics(true)
            .fragment_stores_and_atomics(true)
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
            .push_next(&mut required.vulkan13)
            .push_next(&mut required.robustness2);
        // SAFETY: the adapter belongs to `instance`, and every pointer in
        // `create_info` outlives the call.
        let raw = unsafe { instance.raw.create_device(adapter.raw, &create_info, None) }
            .check("vkCreateDevice")?;
        let parts = Heap::new(&raw).and_then(|heap| {
            let rest = Allocator::new(&AllocatorCreateDesc {
                instance: instance.raw.clone(),
                device: raw.clone(),
                physical_device: adapter.raw,
                debug_settings: AllocatorDebugSettings::default(),
                buffer_device_address: false,
                allocation_sizes: AllocationSizes::default(),
            })
            .map_err(|error| Error::native("creating the memory allocator", error))
            .and_then(|allocator| Ok((allocator, Queue::create(&raw, adapter.families)?)));
            match rest {
                Ok((allocator, queues)) => Ok((heap, allocator, queues)),
                Err(error) => {
                    // SAFETY: nothing uses the heap yet.
                    unsafe { heap.destroy(&raw) };
                    Err(error)
                }
            }
        });
        let (heap, allocator, queues) = match parts {
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
            swapchains: instance
                .surface
                .is_some()
                .then(|| ash::khr::swapchain::Device::new(&instance.raw, &raw)),
            raw,
            physical: adapter.raw,
            heap,
            non_coherent_atom_size: adapter.non_coherent_atom_size,
            allocator: Mutex::new(ManuallyDrop::new(allocator)),
            queues,
            vk_queues,
            graveyard: Graveyard::new(),
            initializations: Mutex::default(),
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

    /// Drops `image`'s first transition, if it was not recorded yet: the
    /// image is destroyed, or a submitted copy encoder took it over.
    pub(crate) fn forget_initialization(&self, image: ImageId) {
        self.initializations
            .lock()
            .retain(|transition| transition.image != image);
    }

    /// Keeps `garbage` until the work of the current frame completes.
    pub(super) fn bury(&self, garbage: Garbage) {
        self.graveyard.bury(garbage);
    }

    /// The last frame whose work read each heap slot.
    pub(crate) fn reads(&self) -> &Reads {
        &self.graveyard.reads
    }

    /// Tags objects dropped from now on with `frame`.
    pub(crate) fn begin_frame(&self, frame: u64) {
        self.graveyard.begin_frame(frame);
    }

    /// Destroys every object dropped by the end of frame `completed`, whose
    /// work, and all work before it, has completed, unless later work reads
    /// its heap slots.
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
                Garbage::Buffer {
                    raw,
                    allocation,
                    slot,
                } => {
                    if let Some(slot) = slot {
                        self.remove_from_heap(slot);
                    }
                    self.raw.destroy_buffer(raw, None);
                    self.free(allocation);
                }
                Garbage::Image {
                    raw,
                    view,
                    allocation,
                    slots,
                } => {
                    for slot in slots.into_iter().flatten() {
                        self.remove_from_heap(slot);
                    }
                    if let Some(view) = view {
                        self.raw.destroy_image_view(view, None);
                    }
                    if let Some(raw) = raw {
                        // No graphics submission recorded its first
                        // transition since it was created; none will.
                        self.forget_initialization(raw);
                        self.raw.destroy_image(raw, None);
                        self.free(allocation);
                    }
                }
                Garbage::Sampler(raw, slot) => {
                    self.remove_from_heap(slot);
                    self.raw.destroy_sampler(raw, None);
                }
                Garbage::Pipeline(raw) => self.raw.destroy_pipeline(raw, None),
                Garbage::Swapchain(raw, semaphores) => {
                    // Presentation signals no timeline, so only an idle queue
                    // proves it is done with the swapchain and semaphores.
                    // Retired swapchains are rare and their frames are done,
                    // so this waits for at most the last frame's work.
                    let queue = self.lock_queue(QueueType::Graphics);
                    if let Err(error) = self.raw.queue_wait_idle(*queue) {
                        error!(%error, "failed to wait for presentation to finish");
                    }
                    drop(queue);
                    for semaphore in semaphores {
                        self.raw.destroy_semaphore(semaphore, None);
                    }
                    if let Some(swapchains) = &self.swapchains {
                        swapchains.destroy_swapchain(raw, None);
                    }
                }
                Garbage::Surface(raw, target) => {
                    if let Some((surfaces, _)) = &self.instance.surface {
                        surfaces.destroy_surface(raw, None);
                    }
                    // The window may close now; see `Surface`.
                    drop(target);
                }
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

    /// Opens a debug label region `label` in `commands`, when validation is
    /// enabled.
    pub(super) fn begin_label(&self, commands: vk::CommandBuffer, label: &str) {
        let Some(debug) = &self.debug else { return };
        let name = CString::new(label).unwrap_or_default();
        let info = vk::DebugUtilsLabelEXT::default().label_name(&name);
        // SAFETY: `commands` is recording and was allocated from this device.
        unsafe { debug.cmd_begin_debug_utils_label(commands, &info) };
    }

    /// Closes the debug label region opened last in `commands`.
    pub(super) fn end_label(&self, commands: vk::CommandBuffer) {
        if let Some(debug) = &self.debug {
            // SAFETY: `begin_label` opened a region in `commands`.
            unsafe { debug.cmd_end_debug_utils_label(commands) };
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
            // No frame follows to read anything.
            self.begin_frame(u64::MAX);
            self.collect(u64::MAX);
            for queue in &self.queues {
                queue.destroy(&self.raw);
            }
            self.heap.destroy(&self.raw);
            // The allocator frees its memory through the device.
            ManuallyDrop::drop(self.allocator.get_mut());
            self.raw.destroy_device(None);
        }
    }
}
