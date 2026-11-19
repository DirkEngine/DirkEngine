use std::{fmt, time::Duration};

use crate::{Error, Result, Rhi, native};

/// The queues work is submitted to. Each has one timeline that its
/// submissions signal in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Rendering, compute and copies.
    Graphics,
    /// Copies, on a dedicated transfer queue when the device has one.
    Copy,
}

impl QueueType {
    /// Every queue, indexed by `queue as usize`.
    pub const ALL: [Self; 2] = [Self::Graphics, Self::Copy];
    /// Number of queues.
    pub const COUNT: usize = Self::ALL.len();
}

mod sealed {
    /// Prevents queue kinds outside this crate.
    pub trait Sealed {}
}

/// A queue, as a type, so commands a queue cannot run do not compile.
pub trait QueueKind: sealed::Sealed {
    /// The queue this kind records for.
    const TYPE: QueueType;
}

/// The graphics queue: rendering, compute, copies and blits.
pub struct Graphics;

/// The copy queue: copies only.
///
/// Named `CopyQueue` rather than `Copy` so glob imports do not shadow the
/// `Copy` trait.
pub struct CopyQueue;

impl sealed::Sealed for Graphics {}
impl sealed::Sealed for CopyQueue {}

impl QueueKind for Graphics {
    const TYPE: QueueType = QueueType::Graphics;
}

impl QueueKind for CopyQueue {
    const TYPE: QueueType = QueueType::Copy;
}

/// A point on a queue's timeline, reached when a submission completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Completion {
    /// The id of the RHI the submission ran on.
    device: u64,
    pub(crate) queue: QueueType,
    pub(crate) value: u64,
}

impl Completion {
    /// The queue the submission ran on.
    #[must_use]
    pub fn queue(self) -> QueueType {
        self.queue
    }
}

impl fmt::Display for Completion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} submission {}", self.queue, self.value)
    }
}

/// Moves exclusive ownership of a resource between queues. The source queue
/// releases the resource with a barrier and the destination queue acquires it
/// with the same barrier, after waiting for the release's [`Completion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueueTransfer {
    /// The queue releasing the resource.
    pub from: QueueType,
    /// The queue acquiring the resource.
    pub to: QueueType,
}

impl Rhi {
    /// Blocks until `completion` is reached or `timeout` passes. No lock is
    /// held while waiting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] if `timeout` passes first,
    /// [`Error::Invalid`] if `completion` was not submitted on this device, or
    /// a native error such as [`Error::DeviceLost`].
    pub fn wait(&self, completion: Completion, timeout: Duration) -> Result<()> {
        self.check_submitted(completion)?;
        let mut values = [0; QueueType::COUNT];
        values[completion.queue as usize] = completion.value;
        if self.device.wait(values, timeout)? {
            Ok(())
        } else {
            Err(Error::Timeout {
                completion,
                timeout,
            })
        }
    }

    /// Whether `completion` has been reached, without blocking.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `completion` was not submitted on this
    /// device, or a native error such as [`Error::DeviceLost`].
    pub fn is_complete(&self, completion: Completion) -> Result<bool> {
        self.check_submitted(completion)?;
        Ok(self.device.completed(completion.queue)? >= completion.value)
    }

    /// The completion of submission `value` to `queue`.
    pub(crate) fn completion(&self, queue: QueueType, value: u64) -> Completion {
        Completion {
            device: self.id,
            queue,
            value,
        }
    }

    /// Rejects completions this device never submitted, which would never be
    /// reached.
    pub(crate) fn check_submitted(&self, completion: Completion) -> Result<()> {
        if completion.device != self.id
            || completion.value > self.device.submitted()[completion.queue as usize]
        {
            return Err(Error::invalid(
                "completion",
                &completion.to_string(),
                "was not submitted on this device",
            ));
        }
        Ok(())
    }
}

impl native::Device {
    /// The value each queue's last submission signals.
    pub(crate) fn submitted(&self) -> [u64; QueueType::COUNT] {
        QueueType::ALL.map(|queue| self.submitted_on(queue))
    }
}
