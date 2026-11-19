use std::{cell::RefCell, sync::Arc, time::Duration};

use tracing::error;

use crate::{Buffer, CommandEncoder, Completion, Error, QueueKind, QueueType, Result, Rhi, native};

/// Frames whose work may run on the GPU at once. Beginning a frame waits for
/// the frame this many frames earlier.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// The state of one of the [`FRAMES_IN_FLIGHT`] frame slots.
#[derive(Default)]
pub(crate) struct FrameSlot {
    /// Each queue's last submission when the frame using this slot began.
    started: [u64; QueueType::COUNT],
    /// Command pools of the frame using this slot, created on first use.
    /// Lent to the open frame.
    pools: [Option<native::CommandPool>; QueueType::COUNT],
}

/// A recording scope: commands are recorded and submitted within a frame.
/// Headless work uses frames too.
///
/// Beginning a frame waits until the frame [`FRAMES_IN_FLIGHT`] frames
/// earlier has completed on every queue, then frees the objects dropped by
/// its end. Finishing or dropping the frame closes it; encoders cannot
/// outlive it.
///
/// Only one frame is open at a time:
/// ```compile_fail
/// use dirk_rhi::{Result, Rhi};
/// fn twice(rhi: &mut Rhi) -> Result<()> {
///     let first = rhi.begin_frame()?;
///     let second = rhi.begin_frame()?;
///     drop(first);
///     Ok(())
/// }
/// ```
pub struct Frame<'r> {
    rhi: &'r mut Rhi,
    number: u64,
    /// This frame's slot's command pools, returned to the slot on close.
    pools: RefCell<[Option<native::CommandPool>; QueueType::COUNT]>,
}

impl Rhi {
    /// Begins the next frame. Waits for the work of the frame
    /// [`FRAMES_IN_FLIGHT`] frames earlier, frees the objects dropped by its
    /// end and recycles its command pools.
    ///
    /// # Errors
    ///
    /// Returns a native error, such as [`Error::DeviceLost`], if waiting or
    /// resetting the command pools fails.
    pub fn begin_frame(&mut self) -> Result<Frame<'_>> {
        let number = self.frame + 1;
        let slot = Self::slot(number);
        // Frames up to `number - FRAMES_IN_FLIGHT` were submitted before
        // frame `number - FRAMES_IN_FLIGHT + 1` began, which used the next slot.
        let oldest = self.slots[(slot + 1) % FRAMES_IN_FLIGHT].started;
        self.device.wait(oldest, Duration::MAX)?;
        self.device
            .collect(number.saturating_sub(FRAMES_IN_FLIGHT as u64));
        self.device.begin_frame(number);

        let state = &mut self.slots[slot];
        for pool in state.pools.iter_mut().flatten() {
            pool.reset()?;
        }
        state.started = self.device.submitted();
        let pools = std::mem::take(&mut state.pools);
        self.frame = number;
        Ok(Frame {
            rhi: self,
            number,
            pools: RefCell::new(pools),
        })
    }

    /// Waits until all submitted work completes, then frees every dropped
    /// object.
    ///
    /// # Errors
    ///
    /// Returns a native error, such as [`Error::DeviceLost`], if waiting
    /// fails.
    pub fn wait_idle(&mut self) -> Result<()> {
        self.device.wait(self.device.submitted(), Duration::MAX)?;
        self.device.collect(self.frame);
        Ok(())
    }

    /// The slot frame `number` uses.
    fn slot(number: u64) -> usize {
        // The remainder is below `FRAMES_IN_FLIGHT`, a `usize`.
        #[expect(clippy::cast_possible_truncation, reason = "the remainder fits")]
        let slot = (number % FRAMES_IN_FLIGHT as u64) as usize;
        slot
    }
}

impl Drop for Rhi {
    fn drop(&mut self) {
        if let Err(error) = self.wait_idle() {
            error!(%error, "failed to wait for the GPU before destroying the RHI");
        }
    }
}

impl Frame<'_> {
    /// The frame's number, counting from one.
    #[must_use]
    pub fn number(&self) -> u64 {
        self.number
    }

    /// The RHI, to create objects while the frame is open.
    #[must_use]
    pub fn rhi(&self) -> &Rhi {
        self.rhi
    }

    /// Begins recording commands for queue `Q`.
    ///
    /// # Errors
    ///
    /// Returns a native error if a command pool or buffer cannot be created
    /// or begun.
    pub fn encoder<Q: QueueKind>(&self, label: &str) -> Result<CommandEncoder<'_, Q>> {
        let mut pools = self.pools.borrow_mut();
        let pool = match &mut pools[Q::TYPE as usize] {
            Some(pool) => pool,
            empty @ None => empty.insert(native::CommandPool::new(&self.rhi.device, Q::TYPE)?),
        };
        Ok(CommandEncoder::new(self, pool.begin(label)?))
    }

    /// Submits `encoder`'s commands to its queue.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the encoder belongs to another frame, or
    /// a native error such as [`Error::DeviceLost`].
    pub fn submit<Q: QueueKind>(&self, encoder: CommandEncoder<'_, Q>) -> Result<Completion> {
        let (raw, waits) = self.take(encoder)?;
        let value = self.rhi.device.submit(Q::TYPE, raw, waits)?;
        Ok(self.rhi.completion(Q::TYPE, value))
    }

    /// The commands and waits of `encoder`, checking it belongs to this
    /// frame.
    pub(crate) fn take<Q: QueueKind>(
        &self,
        mut encoder: CommandEncoder<'_, Q>,
    ) -> Result<(native::CommandBuffer, [u64; QueueType::COUNT])> {
        encoder.end_pass();
        let CommandEncoder {
            frame, raw, waits, ..
        } = encoder;
        if !std::ptr::eq(frame, self) {
            return Err(Error::invalid(
                "submission",
                &format!("frame {}", self.number),
                "the encoder belongs to another frame",
            ));
        }
        Ok((raw, waits))
    }

    /// Closes the frame. Dropping it does the same.
    pub fn finish(self) {}

    /// The slot this frame uses.
    pub(crate) fn slot(&self) -> usize {
        Rhi::slot(self.number)
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        let slot = self.slot();
        self.rhi.slots[slot].pools = std::mem::take(self.pools.get_mut());
    }
}

/// One value per frame slot, such as a host-written buffer each frame
/// updates. Holding a [`Frame`] proves its slot's previous work completed,
/// so the slot's value can be written safely.
pub struct PerFrame<T> {
    /// Address of the device the values belong to, identifying the RHI.
    device: usize,
    /// One value per slot, `FRAMES_IN_FLIGHT` long.
    slots: Box<[T]>,
}

impl<T> PerFrame<T> {
    /// Creates one value per slot for frames of `rhi`.
    pub fn new(rhi: &Rhi, create: impl FnMut(usize) -> T) -> Self {
        Self {
            device: Arc::as_ptr(&rhi.device).addr(),
            slots: (0..FRAMES_IN_FLIGHT).map(create).collect(),
        }
    }

    /// Creates one value per slot for frames of `rhi`, stopping at the first
    /// error.
    ///
    /// # Errors
    ///
    /// Returns the first error `create` returns.
    pub fn try_new<E>(rhi: &Rhi, create: impl FnMut(usize) -> Result<T, E>) -> Result<Self, E> {
        Ok(Self {
            device: Arc::as_ptr(&rhi.device).addr(),
            slots: (0..FRAMES_IN_FLIGHT)
                .map(create)
                .collect::<Result<_, E>>()?,
        })
    }

    /// The value for `frame`'s slot.
    ///
    /// # Panics
    ///
    /// Panics if `frame` belongs to another RHI than the one these values
    /// were created for.
    #[must_use]
    pub fn get(&self, frame: &Frame<'_>) -> &T {
        &self.slots[self.slot(frame)]
    }

    /// The value for `frame`'s slot, which no earlier frame's work still
    /// uses.
    ///
    /// # Panics
    ///
    /// Panics if `frame` belongs to another RHI than the one these values
    /// were created for.
    pub fn get_mut(&mut self, frame: &Frame<'_>) -> &mut T {
        let slot = self.slot(frame);
        &mut self.slots[slot]
    }

    /// Whether these values were created for frames of `device`.
    pub(crate) fn is_from(&self, device: &native::Device) -> bool {
        self.device == std::ptr::from_ref(device).addr()
    }

    /// `frame`'s slot, checking it belongs to the right RHI.
    fn slot(&self, frame: &Frame<'_>) -> usize {
        assert_eq!(
            self.device,
            Arc::as_ptr(&frame.rhi.device).addr(),
            "PerFrame values are used with frames of the RHI that created them"
        );
        frame.slot()
    }
}

impl PerFrame<Buffer> {
    /// Writes `data` at `offset` into `frame`'s buffer. Safe because the
    /// frame waited for the previous work using this buffer; write before
    /// submitting this frame's work that reads it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the buffer is not host-visible or the
    /// range exceeds it.
    ///
    /// # Panics
    ///
    /// Panics if `frame` belongs to another RHI than the one these buffers
    /// were created for.
    pub fn write(&mut self, frame: &Frame<'_>, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self.get_mut(frame);
        // SAFETY: the slot's buffer is only reachable through frames using
        // this slot, and beginning `frame` waited for the previous one.
        unsafe { buffer.write(offset, data) }
    }
}
