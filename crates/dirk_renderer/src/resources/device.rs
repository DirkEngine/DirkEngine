use std::{
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::vk;
use parking_lot::Mutex;

use crate::{
    MAX_FRAMES_IN_FLIGHT, RendererProperties, Result,
    resources::{
        ActiveRhi,
        command_pool::{CommandPool, Graphics, Transfer},
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
    /// For single use buffers.
    /// Used for texture uploads and layout transitions.
    pub transfer_pool: CommandPool<Transfer>,
    /// For single use buffers.
    /// Used for mip generation.
    pub graphics_pool: CommandPool<Graphics>,
    pub properties: RendererProperties,

    deletion_queue: Mutex<DeletionQueue>,
    current_frame: Arc<AtomicUsize>,
    frame_count: Arc<AtomicUsize>,
    /// Active render hardware interface. Declared last so it outlives all
    /// native compatibility resources above.
    pub rhi: Arc<ActiveRhi>,
}

pub struct FrameCounters {
    pub current_frame: Arc<AtomicUsize>,
    pub frame_count: Arc<AtomicUsize>,
}

impl RenderDevice {
    pub fn new(
        rhi: Arc<ActiveRhi>,
        properties: RendererProperties,
        frame_counters: FrameCounters,
    ) -> Result<Self> {
        let device = rhi.device().clone();

        // COMMAND POOLS
        let transfer_pool = CommandPool::build(&rhi)?;
        let graphics_pool = CommandPool::build(&rhi)?;

        Ok(Self(Arc::new(RenderDeviceInner {
            device,
            transfer_pool,
            graphics_pool,
            properties,
            deletion_queue: Mutex::new(DeletionQueue::new(
                frame_counters.frame_count.clone(),
                MAX_FRAMES_IN_FLIGHT,
            )),
            current_frame: frame_counters.current_frame,
            frame_count: frame_counters.frame_count,
            rhi,
        })))
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

impl Drop for RenderDeviceInner {
    fn drop(&mut self) {
        self.deletion_queue.lock().flush_all(self);
    }
}

/// Everything that can be enqueued for deferred destruction.
/// Add variants as you add resource types.
pub enum Garbage {
    #[cfg(feature = "editor")]
    Sampler(vk::Sampler),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    DescriptorSet {
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
    },
    DescriptorPool(vk::DescriptorPool),
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
                #[cfg(feature = "editor")]
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
                Self::Shader(shader) => device.destroy_shader_module(shader, None),
            }
        }
    }
}
