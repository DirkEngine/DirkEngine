use std::{
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use block2::RcBlock;
use objc2::{rc::autoreleasepool, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLSharedEvent,
};
use parking_lot::Mutex;

use objc2_foundation::NSString;

use super::{Check, CommandBuffer, Device, Object, Presentation};
use crate::{QueueType, Result};

/// How long a wait blocks before checking whether submitted work failed,
/// which may leave a timeline short of its value forever.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// A command queue and its timeline: an event its submissions signal, one
/// value each.
pub(super) struct Queue {
    pub(super) raw: Object<dyn MTLCommandQueue>,
    event: Object<dyn MTLSharedEvent>,
    /// The value the last submission signals. Advanced while holding
    /// `submitting`, read without it.
    submitted: AtomicU64,
    /// Held to signal the next value and commit, so values are committed in
    /// order.
    submitting: Mutex<()>,
}

impl Queue {
    /// Creates the queue of `kind`, labelled after it and `application`.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        kind: QueueType,
        application: &str,
    ) -> Result<Self> {
        let raw: Object<_> = device.newCommandQueue().check("newCommandQueue")?.into();
        raw.setLabel(Some(&NSString::from_str(&format!(
            "{application} {kind:?}"
        ))));
        Ok(Self {
            raw,
            event: device.newSharedEvent().check("newSharedEvent")?.into(),
            submitted: AtomicU64::new(0),
            submitting: Mutex::new(()),
        })
    }
}

impl Device {
    /// Submits recorded `commands`, in order, to `queue` after the GPU
    /// reaches each queue's value in `waits`, and returns the value the
    /// submission signals. Only this queue's lock is held, and the heap's
    /// while committing, so no slot the commands read is written meanwhile.
    /// Presentation needs no semaphores: drawables are presented after the
    /// work committed before them.
    ///
    /// Metal waits for events only before the passes encoded after the wait,
    /// and `waits` are known only now, after the passes. So they are encoded
    /// in a command buffer committed just before: a queue blocks at a wait
    /// until the event reaches its value, holding back what follows.
    pub(crate) fn submit(
        &self,
        queue: QueueType,
        commands: impl IntoIterator<Item = CommandBuffer>,
        waits: [u64; QueueType::COUNT],
        _presentation: Presentation<'_>,
    ) -> Result<u64> {
        let mut commands: Vec<_> = commands.into_iter().collect();
        for commands in &mut commands {
            commands.end_encoding();
        }
        let target = &self.queues[queue as usize];
        // A submission without commands still signals its value.
        let empty;
        let raws: Vec<_> = if commands.is_empty() {
            empty = autoreleasepool(|_| target.raw.commandBuffer()).check("commandBuffer")?;
            vec![&*empty]
        } else {
            commands.iter().map(CommandBuffer::raw).collect()
        };
        let _submitting = target.submitting.lock();
        if waits.iter().any(|&value| value > 0) {
            let wait = autoreleasepool(|_| target.raw.commandBuffer()).check("commandBuffer")?;
            wait.setLabel(Some(&NSString::from_str("waits")));
            for (queue, value) in self.queues.iter().zip(waits) {
                if value > 0 {
                    wait.encodeWaitForEvent_value(ProtocolObject::from_ref(&*queue.event), value);
                }
            }
            wait.commit();
        }
        let value = target.submitted.load(Ordering::Relaxed) + 1;
        #[expect(
            clippy::expect_used,
            reason = "A submission without commands commits an empty command buffer."
        )]
        let last = raws.last().expect("a submission has a command buffer");
        last.encodeSignalEvent_value(ProtocolObject::from_ref(&*target.event), value);
        self.submit_reading_heap(|| {
            for raw in raws {
                self.commit(raw);
            }
        });
        target.submitted.store(value, Ordering::Release);
        Ok(value)
    }

    /// Commits `raw`, recording its failure in the device's health.
    fn commit(&self, raw: &ProtocolObject<dyn MTLCommandBuffer>) {
        let health = Arc::clone(&self.health);
        let completed = RcBlock::new(move |raw: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal passes the completed command buffer.
            let raw = unsafe { raw.as_ref() };
            if raw.status() == MTLCommandBufferStatus::Error {
                let label = raw
                    .label()
                    .map_or_else(String::new, |label| label.to_string());
                let reason = raw.error().map_or_else(
                    || "failed without an error".to_owned(),
                    |error| error.to_string(),
                );
                health.fail(&label, &reason);
            }
        });
        // SAFETY: Metal copies the block, which only touches the shared
        // health record, before `commit` returns.
        unsafe { raw.addCompletedHandler(RcBlock::as_ptr(&completed)) };
        raw.commit();
    }

    /// Blocks until every queue reaches its value in `values`, or `timeout`
    /// passes. Returns whether they were reached. Holds no lock.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DeviceLost`](crate::Error::DeviceLost) once
    /// submitted work has failed.
    pub(crate) fn wait(&self, values: [u64; QueueType::COUNT], timeout: Duration) -> Result<bool> {
        let deadline = Instant::now().checked_add(timeout);
        for (queue, value) in self.queues.iter().zip(values) {
            loop {
                self.health.check()?;
                let remaining =
                    deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
                let slice = remaining.map_or(WAIT_SLICE, |remaining| remaining.min(WAIT_SLICE));
                #[expect(clippy::cast_possible_truncation, reason = "a slice is 100 ms at most")]
                let milliseconds = slice.as_millis() as u64;
                if queue
                    .event
                    .waitUntilSignaledValue_timeoutMS(value, milliseconds)
                {
                    break;
                }
                if remaining.is_some_and(|remaining| remaining <= WAIT_SLICE) {
                    return Ok(false);
                }
            }
        }
        self.health.check()?;
        Ok(true)
    }

    /// The value `queue`'s timeline has reached.
    pub(crate) fn completed(&self, queue: QueueType) -> Result<u64> {
        self.health.check()?;
        Ok(self.queues[queue as usize].event.signaledValue())
    }

    /// The value `queue`'s last submission signals.
    pub(crate) fn submitted_on(&self, queue: QueueType) -> u64 {
        self.queues[queue as usize]
            .submitted
            .load(Ordering::Acquire)
    }
}
