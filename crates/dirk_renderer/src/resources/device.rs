use std::{
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(validation)]
use ash::ext::debug_utils;
use ash::{
    khr::{surface, swapchain},
    vk,
};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;

use crate::{
    MAX_FRAMES_IN_FLIGHT, RendererProperties, Result,
    resources::{
        command_pool::{CommandPool, Graphics, Transfer},
        queues::Queues,
    },
};

/// The device that stores all vulkan objects.
#[derive(Clone)]
pub struct RenderDevice(Arc<RenderDeviceInner>);

impl Deref for RenderDevice {
    type Target = RenderDeviceInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct RenderDeviceInner {
    pub device: ash::Device,
    pub surface_loader: surface::Instance,
    pub swapchain_loader: swapchain::Device,

    pub instance: ash::Instance,
    pub entry: ash::Entry,
    pub physical_device: vk::PhysicalDevice,
    pub queues: Queues,
    /// For single use buffers.
    /// Used for texture uploads and layout transitions.
    pub transfer_pool: CommandPool<Transfer>,
    /// For single use buffers.
    /// Used for mip generation.
    pub graphics_pool: CommandPool<Graphics>,
    pub properties: RendererProperties,

    #[cfg(validation)]
    debug_utils_loader: debug_utils::Instance,
    #[cfg(validation)]
    debug_messenger: vk::DebugUtilsMessengerEXT,

    allocator: Option<Mutex<Allocator>>,
    deletion_queue: Mutex<DeletionQueue>,
    current_frame: Arc<AtomicUsize>,
    frame_count: Arc<AtomicUsize>,
}

pub struct FrameCounters {
    pub current_frame: Arc<AtomicUsize>,
    pub frame_count: Arc<AtomicUsize>,
}

impl RenderDevice {
    pub fn new(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        properties: RendererProperties,
        frame_counters: FrameCounters,
        #[cfg(validation)] debug_messenger: vk::DebugUtilsMessengerEXT,
    ) -> Result<Self> {
        // ALLOCATOR
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })?;

        // SWAP CHAIN
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        // QUEUES
        let queues = Queues::new(&instance, &device, &properties.queue_family_indices);

        // COMMAND POOLS
        let transfer_pool = CommandPool::build(
            &device,
            &properties.queue_family_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )?;
        let graphics_pool = CommandPool::build(
            &device,
            &properties.queue_family_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        )?;

        Ok(Self(Arc::new(RenderDeviceInner {
            device,
            surface_loader: surface::Instance::new(&entry, &instance),
            swapchain_loader,
            physical_device,
            queues,
            transfer_pool,
            graphics_pool,
            properties,
            allocator: Some(Mutex::new(allocator)),
            deletion_queue: Mutex::new(DeletionQueue::new(
                frame_counters.frame_count.clone(),
                MAX_FRAMES_IN_FLIGHT,
            )),
            current_frame: frame_counters.current_frame,
            frame_count: frame_counters.frame_count,

            #[cfg(validation)]
            debug_utils_loader: debug_utils::Instance::new(&entry, &instance),
            #[cfg(validation)]
            debug_messenger,

            instance,
            entry,
        })))
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc<'_>) -> Result<Allocation> {
        Ok(self
            .allocator
            .as_ref()
            .expect("allocator should exist")
            .lock()
            .allocate(desc)?)
    }

    pub fn destroy(&mut self, garbage: Garbage) {
        self.deletion_queue.lock().enqueue(garbage);
    }

    /// Call once per frame from your render loop.
    pub fn flush_deletions(&self) {
        let mut queue = self.deletion_queue.lock();
        queue.flush(self, self.frame_count.load(Ordering::Relaxed));
    }

    pub fn current_frame(&self) -> usize {
        self.current_frame.load(Ordering::Relaxed)
    }
}

impl RenderDeviceInner {
    fn free(&self, allocation: Allocation) -> Result<()> {
        Ok(self
            .allocator
            .as_ref()
            .expect("allocator should exist")
            .lock()
            .free(allocation)?)
    }
}

impl Drop for RenderDeviceInner {
    fn drop(&mut self) {
        self.graphics_pool.destroy();
        self.transfer_pool.destroy();

        self.deletion_queue.lock().flush_all(self);
        drop(self.allocator.take());

        unsafe {
            self.device.destroy_device(None);
            #[cfg(validation)]
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Everything that can be enqueued for deferred destruction.
/// Add variants as you add resource types.
pub enum Garbage {
    Allocation(Allocation),
    Buffer(vk::Buffer),
    Image(vk::Image),
    ImageView(vk::ImageView),
    Sampler(vk::Sampler),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    DescriptorSet {
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
    },
    DescriptorPool(vk::DescriptorPool),
    Semaphore(vk::Semaphore),

    Swapchain(vk::SwapchainKHR),
    Surface {
        surface: vk::SurfaceKHR,
        /// Keeps the native window alive until deferred surface destruction.
        target: Arc<dirk_platform::WindowSurfaceTarget>,
    },
    Shader(vk::ShaderModule),
}

struct PendingDeletion {
    garbage: Option<Garbage>,
    death_frame: usize, // monotonic frame count after which it's safe to delete
}

struct DeletionQueue {
    pending: Vec<PendingDeletion>,
    frame_count: Arc<AtomicUsize>,
    frames_in_flight: usize,
}

impl DeletionQueue {
    fn new(frame_count: Arc<AtomicUsize>, frames_in_flight: usize) -> Self {
        Self {
            pending: Vec::new(),
            frame_count,
            frames_in_flight,
        }
    }

    /// Call this when you're done with a resource.
    fn enqueue(&mut self, garbage: Garbage) {
        self.pending.push(PendingDeletion {
            garbage: Some(garbage),
            death_frame: self.frame_count.load(Ordering::Relaxed) + 2 * self.frames_in_flight, // wait two frames before destroying
        });
    }

    /// Call once per frame. Destroys anything safe to destroy.
    fn flush(&mut self, device: &RenderDevice, frame_count: usize) {
        self.pending.retain_mut(|item| {
            if frame_count >= item.death_frame {
                if let Some(garbage) = item.garbage.take() {
                    garbage.destroy(device);
                }
                false
            } else {
                true
            }
        });
    }

    /// Call on shutdown — destroys everything regardless of frame.
    fn flush_all(&mut self, device: &RenderDeviceInner) {
        for mut item in self.pending.drain(..) {
            if let Some(garbage) = item.garbage.take() {
                garbage.destroy(device);
            }
        }
    }
}

impl Garbage {
    fn destroy(self, render_device: &RenderDeviceInner) {
        let device = &render_device.device;
        unsafe {
            match self {
                Self::Allocation(allocation) => render_device
                    .free(allocation)
                    .expect("Failed to free allocation"),
                Self::Buffer(buffer) => device.destroy_buffer(buffer, None),
                Self::Image(image) => device.destroy_image(image, None),
                Self::ImageView(view) => device.destroy_image_view(view, None),
                Self::Sampler(s) => device.destroy_sampler(s, None),
                Self::Pipeline(p) => device.destroy_pipeline(p, None),
                Self::PipelineLayout(l) => device.destroy_pipeline_layout(l, None),
                Self::DescriptorSetLayout(layout) => {
                    device.destroy_descriptor_set_layout(layout, None);
                }
                Self::DescriptorSet { pool, set } => {
                    let _ = device.free_descriptor_sets(pool, &[set]);
                }
                Self::DescriptorPool(pool) => device.destroy_descriptor_pool(pool, None),
                Self::Semaphore(semaphore) => device.destroy_semaphore(semaphore, None),
                Self::Surface { surface, target } => {
                    render_device.surface_loader.destroy_surface(surface, None);
                    drop(target);
                }
                Self::Swapchain(swapchain) => render_device
                    .swapchain_loader
                    .destroy_swapchain(swapchain, None),
                Self::Shader(shader) => device.destroy_shader_module(shader, None),
            }
        }
    }
}
