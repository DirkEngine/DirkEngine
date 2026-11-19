use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use ash::vk;
use parking_lot::MutexGuard;

use super::{
    Check, CommandBuffer, Device,
    adapter::QueueFamilies,
    presentation::{PRESENT_STAGES, Presentation},
};
use crate::{QueueType, Result};

/// A queue and the timeline its submissions signal, one value per
/// submission.
pub(super) struct Queue {
    pub(super) family: u32,
    timeline: vk::Semaphore,
    /// The value the last submission signals. Advanced while holding the
    /// queue's lock, read without it.
    submitted: AtomicU64,
    /// Index of the Vulkan queue submitted to in `Device::vk_queues`. Both
    /// queues share one when the device has no dedicated transfer family.
    submits_to: usize,
}

impl Queue {
    /// Creates a timeline for each queue in `families`.
    pub(super) fn create(device: &ash::Device, families: QueueFamilies) -> Result<[Self; 2]> {
        let timeline = || {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: `info` is valid for the device.
            unsafe { device.create_semaphore(&info, None) }.check("vkCreateSemaphore")
        };
        let graphics = timeline()?;
        let copy = timeline().inspect_err(|_| {
            // SAFETY: the semaphore was never used.
            unsafe { device.destroy_semaphore(graphics, None) };
        })?;
        let queue = |family, timeline, submits_to| Self {
            family,
            timeline,
            submitted: AtomicU64::new(0),
            submits_to,
        };
        Ok([
            queue(families.graphics, graphics, 0),
            queue(
                families.copy,
                copy,
                usize::from(families.copy != families.graphics),
            ),
        ])
    }

    /// Destroys the timeline. The GPU must be idle.
    ///
    /// # Safety
    ///
    /// No submission signalling or waiting on the timeline may be pending.
    pub(super) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller guarantees the timeline is unused.
        unsafe { device.destroy_semaphore(self.timeline, None) };
    }
}

impl Device {
    /// Submits recorded `commands`, in order, to `queue` after the GPU
    /// reaches each queue's value in `waits` and the acquisitions of
    /// `presentation`, which it then signals as rendered, and returns the
    /// value the submission signals on the queue's timeline. Only this
    /// queue's lock is held.
    pub(crate) fn submit(
        &self,
        queue: QueueType,
        commands: impl IntoIterator<Item = CommandBuffer>,
        waits: [u64; QueueType::COUNT],
        presentation: Presentation<'_>,
    ) -> Result<u64> {
        let commands = commands
            .into_iter()
            .map(|commands| {
                // SAFETY: the command buffer is recording and only used by
                // its encoder.
                unsafe { self.raw.end_command_buffer(commands.raw) }.check("vkEndCommandBuffer")?;
                Ok(vk::CommandBufferSubmitInfo::default().command_buffer(commands.raw))
            })
            .collect::<Result<Vec<_>>>()?;
        let semaphore = |semaphore, stages| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .stage_mask(stages)
        };
        let wait_infos: Vec<_> = self
            .queues
            .iter()
            .zip(waits)
            .filter(|&(_, value)| value > 0)
            .map(|(queue, value)| {
                semaphore(queue.timeline, vk::PipelineStageFlags2::ALL_COMMANDS).value(value)
            })
            // The first use of an acquired image writes it as an attachment
            // or by a transfer; see `ImageAccess::Present`.
            .chain(
                presentation
                    .acquired
                    .iter()
                    .map(|&acquired| semaphore(acquired, PRESENT_STAGES)),
            )
            .collect();
        let target = &self.queues[queue as usize];
        let vk_queue = self.lock_queue(queue);
        let value = target.submitted.load(Ordering::Relaxed) + 1;
        let signals: Vec<_> = std::iter::once(
            semaphore(target.timeline, vk::PipelineStageFlags2::ALL_COMMANDS).value(value),
        )
        .chain(
            presentation
                .rendered
                .iter()
                .map(|&rendered| semaphore(rendered, vk::PipelineStageFlags2::ALL_COMMANDS)),
        )
        .collect();
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals);
        // SAFETY: the queue is locked, the commands were recorded for its
        // family, waits refer to values already submitted or to acquired
        // images, and the presentation semaphores are unsignaled.
        unsafe {
            self.raw
                .queue_submit2(*vk_queue, &[submit], vk::Fence::null())
        }
        .check("vkQueueSubmit2")?;
        target.submitted.store(value, Ordering::Release);
        Ok(value)
    }

    /// Blocks until every queue reaches its value in `values`, or `timeout`
    /// passes. Returns whether they were reached. Holds no lock.
    pub(crate) fn wait(&self, values: [u64; QueueType::COUNT], timeout: Duration) -> Result<bool> {
        let mut semaphores = [vk::Semaphore::null(); QueueType::COUNT];
        let mut targets = [0; QueueType::COUNT];
        let mut count = 0;
        for (queue, value) in self.queues.iter().zip(values) {
            if value > 0 {
                semaphores[count] = queue.timeline;
                targets[count] = value;
                count += 1;
            }
        }
        if count == 0 {
            return Ok(true);
        }
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores[..count])
            .values(&targets[..count]);
        let timeout = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the timelines belong to this device; waiting needs no lock.
        match unsafe { self.raw.wait_semaphores(&info, timeout) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(error).check("vkWaitSemaphores"),
        }
    }

    /// Locks the Vulkan queue `queue` submits to, to submit or present.
    pub(super) fn lock_queue(&self, queue: QueueType) -> MutexGuard<'_, vk::Queue> {
        self.vk_queues[self.queues[queue as usize].submits_to].lock()
    }

    /// The value `queue`'s timeline has reached.
    pub(crate) fn completed(&self, queue: QueueType) -> Result<u64> {
        // SAFETY: the timeline belongs to this device.
        unsafe {
            self.raw
                .get_semaphore_counter_value(self.queues[queue as usize].timeline)
        }
        .check("vkGetSemaphoreCounterValue")
    }

    /// The value each queue's last submission signals.
    pub(crate) fn submitted(&self) -> [u64; QueueType::COUNT] {
        QueueType::ALL.map(|queue| {
            self.queues[queue as usize]
                .submitted
                .load(Ordering::Acquire)
        })
    }
}
