use std::{
    env,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use dirk_shaders::bindless::heap;
use objc2_metal::{MTLArgumentBuffersTier, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};
use tracing::{error, info, warn};

use super::{Object, blit::Blitter, heap::Heap, heap::Slot, queue::Queue, timestamp::Clock};
use crate::{
    Capabilities, Error, Features, Format, FormatCapabilities, Limits, MAX_COLOR_TARGETS,
    QueueType, Result, RhiCreateInfo, graveyard::Graveyard,
};

/// A native object whose owner dropped it, released once the GPU is done.
/// Its heap slots are emptied first.
pub(super) struct Garbage {
    pub(super) slots: [Option<Slot>; 2],
    pub(super) object: Box<dyn Send>,
}

impl Garbage {
    /// `object`, which holds no heap slot.
    pub(super) fn object(object: impl Send + 'static) -> Self {
        Self {
            slots: [None; 2],
            object: Box::new(object),
        }
    }
}

/// Failures of submitted work, reported by command buffers when they
/// complete. Shared with their completion handlers.
#[derive(Default)]
pub(super) struct Health {
    /// Whether a command buffer failed. Its timeline may then never reach
    /// its value, so waits report the device as lost.
    lost: AtomicBool,
    /// Command buffers that failed.
    errors: AtomicUsize,
}

impl Health {
    /// Records the failure of command buffer `label`.
    pub(super) fn fail(&self, label: &str, reason: &str) {
        error!(target: "metal", command_buffer = label, "{reason}");
        self.errors.fetch_add(1, Ordering::Relaxed);
        self.lost.store(true, Ordering::Release);
    }

    /// Fails with [`Error::DeviceLost`] once submitted work has failed.
    pub(super) fn check(&self) -> Result<()> {
        if self.lost.load(Ordering::Acquire) {
            return Err(Error::DeviceLost);
        }
        Ok(())
    }
}

/// The Metal device, shared by the RHI and every object created from it.
pub(crate) struct Device {
    pub(crate) capabilities: Capabilities,
    pub(super) raw: Object<dyn MTLDevice>,
    /// Indexed by `QueueType`.
    pub(super) queues: [Queue; QueueType::COUNT],
    /// The bindless heap's argument buffer.
    pub(super) heap: Heap,
    /// Scaling blits, which Metal draws.
    pub(super) blitter: Blitter,
    /// How timestamps are sampled; `None` without support.
    pub(super) clock: Option<Clock>,
    pub(super) health: Arc<Health>,
    /// Whether command buffers report which encoder failed.
    pub(super) validation: bool,
    /// Whether the RHI was created for a display.
    pub(super) presentation: bool,
    graveyard: Graveyard<Garbage>,
}

impl Device {
    /// Creates the system's default device, if it meets the RHI's
    /// requirements: Metal 3 and tier-2 argument buffers that hold the heap.
    pub(crate) fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        let raw: Object<_> = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::NoSuitableDevice("no Metal device found".to_owned()))?
            .into();
        let name = raw.name().to_string();
        let mut missing = Vec::new();
        if !raw.supportsFamily(MTLGPUFamily::Metal3) {
            missing.push("Metal 3");
        }
        if raw.argumentBuffersSupport() != MTLArgumentBuffersTier::Tier2 {
            missing.push("tier-2 argument buffers");
        }
        if raw.maxArgumentBufferSamplerCount() < heap::SAMPLER_CAPACITY as usize {
            missing.push("argument buffers holding the heap's samplers");
        }
        if !missing.is_empty() {
            return Err(Error::NoSuitableDevice(format!(
                "{name} lacks {}",
                missing.join(", ")
            )));
        }
        info!(device = %name, "selected Metal device");
        if info.validation && env::var_os("MTL_DEBUG_LAYER").is_none() {
            warn!(
                "Metal API validation was requested; enable it by running with MTL_DEBUG_LAYER=1 \
                 (and MTL_SHADER_VALIDATION=1 for shaders)"
            );
        }

        let queues = [
            Queue::new(&raw, QueueType::Graphics, info.application_name)?,
            Queue::new(&raw, QueueType::Copy, info.application_name)?,
        ];
        let clock = Clock::new(&raw);
        Ok(Self {
            capabilities: Self::capabilities(&raw, name, clock.is_some()),
            heap: Heap::new(&raw, &queues[QueueType::Graphics as usize].raw)?,
            blitter: Blitter::default(),
            clock,
            health: Arc::default(),
            validation: info.validation,
            presentation: info.display.is_some(),
            graveyard: Graveyard::new(),
            queues,
            raw,
        })
    }

    /// What `raw`, named `name`, supports, with `timestamps` or not. Metal
    /// has no per-format queries, so formats follow Apple's feature tables;
    /// see `FormatCapabilities`.
    fn capabilities(raw: &Object<dyn MTLDevice>, name: String, timestamps: bool) -> Capabilities {
        let max_buffer_size = raw.maxBufferLength() as u64;
        Capabilities {
            name,
            limits: Limits {
                max_buffer_size,
                max_image_dimension_2d: 16384,
                max_image_dimension_3d: 2048,
                max_image_dimension_cube: 16384,
                max_image_array_layers: 2048,
                max_sampler_anisotropy: 16,
                #[expect(clippy::cast_possible_truncation, reason = "the limit is eight")]
                max_color_targets: MAX_COLOR_TARGETS as u32,
                // `setVertexBytes` takes up to 4 KiB.
                max_push_data_size: 4096,
                // Shaders bound storage buffer reads with 32-bit lengths.
                max_storage_buffer_size: max_buffer_size.min(u64::from(u32::MAX)),
                // Each indirect draw is its own Metal draw.
                max_draw_indirect_count: u32::MAX,
            },
            features: Features::all().difference(if timestamps {
                Features::empty()
            } else {
                Features::TIMESTAMPS
            }),
            dedicated_copy_queue: true,
            formats: std::array::from_fn(|index| {
                FormatCapabilities::metal(Format::ALL[index], raw)
            }),
        }
    }

    /// Keeps `garbage` until the work of the current frame completes.
    pub(super) fn bury(&self, garbage: Garbage) {
        self.graveyard.bury(garbage);
    }

    /// Tags objects dropped from now on with `frame`.
    pub(crate) fn begin_frame(&self, frame: u64) {
        self.graveyard.begin_frame(frame);
    }

    /// Releases every object dropped by the end of frame `completed`, whose
    /// work, and all work before it, has completed.
    pub(crate) fn collect(&self, completed: u64) {
        self.graveyard.collect(completed, |garbage| {
            for slot in garbage.slots.into_iter().flatten() {
                self.remove_from_heap(slot);
            }
            drop(garbage.object);
        });
    }

    /// Number of submitted command buffers that failed. Metal's API
    /// validation, enabled by `MTL_DEBUG_LAYER=1`, aborts on misuse instead
    /// of counting it.
    pub(crate) fn validation_error_count(&self) -> usize {
        self.health.errors.load(Ordering::Relaxed)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // Every object created from the device has been dropped, since each
        // holds the device alive; waiting lets their GPU work finish.
        if let Err(error) = self.wait(self.submitted(), Duration::MAX) {
            error!(%error, "failed to wait for the Metal device before releasing it");
        }
        self.collect(u64::MAX);
    }
}
