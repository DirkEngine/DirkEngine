use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    sync::Arc,
    time::Duration,
};

use bytemuck::NoUninit;
use tracing::error;

use crate::{
    Buffer, BufferDesc, BufferSlice, BufferUsage, CommandEncoder, Completion, Error, Graphics,
    Memory, QueueKind, QueueType, Result, Rhi, native,
};

/// What every frame allocation starts at and reserves a multiple of, in
/// bytes: a multiple of every copy offset alignment and index size.
const ALIGNMENT: u64 = 16;

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
    /// Host-visible memory the frame using this slot allocates from;
    /// `None` without frame memory.
    pub(crate) memory: Option<Buffer>,
}

/// A recording scope: commands are recorded and submitted within a frame.
/// Headless work uses frames too.
///
/// Beginning a frame waits until the frame [`FRAMES_IN_FLIGHT`] frames
/// earlier has completed on every queue, then frees the objects dropped by
/// its end. Finishing or dropping the frame closes it; encoders and
/// [allocations](Frame::alloc) cannot outlive it.
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
    /// Bytes of the slot's memory allocated so far.
    allocated: Cell<u64>,
    /// Buffers of the allocations that did not fit the slot's memory, freed
    /// once the frame's work completes. Only ever added to while the frame
    /// is open, so the buffers stay where they are.
    oversized: RefCell<Vec<Arc<Buffer>>>,
}

/// `len` elements of `T` a frame copied into host-visible memory that only
/// its own work reads. The memory has [`BufferUsage::COPY_SRC`],
/// [`BufferUsage::INDEX`] and [`BufferUsage::STORAGE`], so the elements can
/// be copied from, drawn as indices or read by shaders through
/// [`index`](Self::index).
///
/// It borrows its frame: the memory is reused once the frame's slot comes
/// round again, after the frame's work completed.
pub struct FrameSlice<'f, T> {
    buffer: &'f Buffer,
    /// Index of the first element in the buffer, read as an array of `T`.
    first: u32,
    len: u32,
    element: PhantomData<fn() -> T>,
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
            allocated: Cell::new(0),
            oversized: RefCell::default(),
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

    /// Creates host-visible memory frames allocate from, of `size` bytes.
    pub(crate) fn create_frame_memory(&self, size: u64) -> Result<Buffer> {
        self.create_buffer(&BufferDesc {
            label: "frame memory",
            size,
            usage: BufferUsage::COPY_SRC | BufferUsage::INDEX | BufferUsage::STORAGE,
            memory: Memory::Upload,
        })
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
    pub fn submit<Q: QueueKind>(&self, mut encoder: CommandEncoder<'_, Q>) -> Result<Completion> {
        encoder.end_pass();
        if !std::ptr::eq(encoder.frame, self) {
            return Err(Error::invalid(
                "submission",
                &format!("frame {}", self.number),
                "the encoder belongs to another frame",
            ));
        }
        encoder.take_over_initializations();
        let CommandEncoder { raw, waits, .. } = encoder;
        let initializations = if Q::TYPE == QueueType::Graphics {
            self.initializations()?
        } else {
            None
        };
        let value =
            self.rhi
                .device
                .submit(Q::TYPE, initializations.into_iter().chain([raw]), waits)?;
        Ok(self.rhi.completion(Q::TYPE, value))
    }

    /// Records the first transition of each image created since the last
    /// graphics submission, from undefined contents to its resting access,
    /// to submit before graphics work; `None` if there are none.
    fn initializations(&self) -> Result<Option<native::CommandBuffer>> {
        let transitions = self.rhi.device.take_initializations();
        if transitions.is_empty() {
            return Ok(None);
        }
        let mut encoder = self.encoder::<Graphics>("image initialization")?;
        encoder.transition(&transitions);
        Ok(Some(encoder.raw))
    }

    /// Copies `data` into memory of this frame, which only work it records
    /// reads, and returns where it is. Uploading to the GPU this way is
    /// safe: no submitted work reads fresh memory, and the memory is reused
    /// only once this frame's work completed.
    ///
    /// Allocations come from the frame slot's memory
    /// ([`RhiCreateInfo::frame_memory`](crate::RhiCreateInfo::frame_memory)
    /// bytes). One that does not fit gets a buffer of its own, freed once the
    /// frame's work completes. Each starts at a multiple of 16 bytes, a
    /// valid offset for copies of any format and for indices, and of
    /// `size_of::<T>()`, so shaders read it as an array of `T`; the bytes
    /// after it up to the next multiple of 16 are reserved for it too, so
    /// copies may round its size up to whole words.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the data ends past element `u32::MAX`
    /// of its buffer, or a buffer of its own exceeds the device's limits; or
    /// a native error if memory cannot be allocated or flushed.
    pub fn alloc<T: NoUninit>(&self, data: &[T]) -> Result<FrameSlice<'_, T>> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let reserved = (bytes.len() as u64).next_multiple_of(ALIGNMENT);
        let offset = self
            .allocated
            .get()
            .next_multiple_of(FrameSlice::<T>::START_ALIGNMENT);
        let (buffer, offset) = match &self.rhi.slots[self.slot()].memory {
            Some(memory) if offset + reserved <= memory.info().size => {
                self.allocated.set(offset + reserved);
                (memory, offset)
            }
            _ => (self.oversized(reserved)?, 0),
        };
        let first = offset / FrameSlice::<T>::ELEMENT_SIZE;
        let end = first + data.len() as u64;
        let (Ok(first), Ok(end)) = (u32::try_from(first), u32::try_from(end)) else {
            return Err(Error::invalid(
                "frame allocation",
                buffer.label(),
                format!("its elements {first}..{end} end past element {}", u32::MAX),
            ));
        };
        if !bytes.is_empty() {
            // SAFETY: the range is inside host-visible memory that work
            // submitted earlier does not read: nothing was allocated there
            // since beginning the frame waited for the slot's previous work.
            unsafe { buffer.raw.write(offset, bytes)? };
        }
        Ok(FrameSlice {
            buffer,
            first,
            len: end - first,
            element: PhantomData,
        })
    }

    /// A buffer of its own for an allocation reserving `size` bytes, kept
    /// until the frame closes.
    fn oversized(&self, size: u64) -> Result<&Buffer> {
        let buffer = Arc::new(self.rhi.create_frame_memory(size.max(ALIGNMENT))?);
        let lent = Arc::as_ptr(&buffer);
        self.oversized.borrow_mut().push(buffer);
        // SAFETY: the `Arc` keeps the buffer where it is until the frame,
        // which the returned reference borrows, drops it.
        Ok(unsafe { &*lent })
    }

    /// Closes the frame. Dropping it does the same.
    pub fn finish(self) {}

    /// The slot this frame uses.
    fn slot(&self) -> usize {
        Rhi::slot(self.number)
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        let slot = self.slot();
        self.rhi.slots[slot].pools = std::mem::take(self.pools.get_mut());
    }
}

impl<'f, T> FrameSlice<'f, T> {
    /// The size of `T`. Allocating zero-sized types fails to build.
    const ELEMENT_SIZE: u64 = {
        assert!(
            size_of::<T>() > 0,
            "frames do not allocate zero-sized types"
        );
        size_of::<T>() as u64
    };

    /// What allocations of `T` start at a multiple of: the least common
    /// multiple of [`ALIGNMENT`] and the size of `T`.
    const START_ALIGNMENT: u64 = {
        let (mut a, mut b) = (Self::ELEMENT_SIZE, ALIGNMENT);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Self::ELEMENT_SIZE / a * ALIGNMENT
    };

    /// The buffer holding the elements.
    #[must_use]
    pub fn buffer(&self) -> &'f Buffer {
        self.buffer
    }

    /// Offset of the first element in the buffer, in bytes.
    #[must_use]
    pub fn offset(&self) -> u64 {
        u64::from(self.first) * Self::ELEMENT_SIZE
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size of the elements, in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        u64::from(self.len) * Self::ELEMENT_SIZE
    }

    /// Where shaders read the elements: the slice of the heap's storage
    /// buffer holding them.
    ///
    /// # Panics
    ///
    /// Never: frame memory is a storage buffer, which has a heap slot.
    #[must_use]
    pub fn index(&self) -> BufferSlice<T> {
        let buffer = self
            .buffer
            .index()
            .expect("frame memory is a storage buffer, which has an index");
        BufferSlice::new(buffer, self.first, self.len)
    }
}

// Implemented by hand: deriving would require `T` to implement them too.
impl<T> Clone for FrameSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for FrameSlice<'_, T> {}

/// One value per frame slot, such as each frame's query pool. Holding a
/// [`Frame`] proves its slot's previous work completed, so the slot's value
/// can be reused safely.
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
