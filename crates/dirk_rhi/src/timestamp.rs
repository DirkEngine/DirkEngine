use std::time::Duration;

use crate::{CommandEncoder, Error, Features, Frame, Graphics, PerFrame, Result, Rhi, native};

/// GPU timestamps, `count` per frame: graphics encoders write them between
/// commands, and they are read once the frame that wrote them completed.
///
/// Each frame slot has its own queries. [`Timestamps::begin`] returns the
/// timestamps the slot's previous frame wrote, which the frame's beginning
/// waited for, then clears them for this frame:
///
/// ```no_run
/// # use dirk_rhi::{Graphics, Result, Rhi, Timestamps};
/// # fn render(rhi: &mut Rhi, timestamps: &mut Timestamps) -> Result<()> {
/// let frame = rhi.begin_frame()?;
/// if let [Some(start), Some(end)] = timestamps.begin(&frame)? {
///     println!("an earlier frame took {:?} on the GPU", end.saturating_sub(*start));
/// }
/// let mut encoder = frame.encoder::<Graphics>("scene")?;
/// encoder.write_timestamp(timestamps, 0)?;
/// // ... record the frame ...
/// encoder.write_timestamp(timestamps, 1)?;
/// frame.submit(encoder)?;
/// # Ok(())
/// # }
/// ```
pub struct Timestamps {
    /// Each frame slot's queries.
    slots: PerFrame<Queries>,
    /// Timestamps per frame.
    count: u32,
    /// The frame whose timestamps were last cleared by `begin`.
    frame: u64,
    /// The values `begin` last read.
    results: Box<[Option<Duration>]>,
    /// Raw values read, before conversion; `None` where unavailable.
    ticks: Box<[Option<u64>]>,
    label: Box<str>,
}

/// One frame slot's timestamps. Separate per slot, so the queries of one
/// slot are reset while another's are in use.
struct Queries {
    raw: native::Timestamps,
    /// Which timestamps the slot's last frame wrote.
    written: Box<[bool]>,
}

impl Rhi {
    /// Creates `count` timestamps per frame.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `count` is zero or the device lacks
    /// [`Features::TIMESTAMPS`], or a native error if the queries cannot be
    /// created.
    pub fn create_timestamps(&self, label: &str, count: u32) -> Result<Timestamps> {
        let invalid = |reason| Err(Error::invalid("timestamps", label, reason));
        if !self.capabilities().features.contains(Features::TIMESTAMPS) {
            return invalid("timestamps are not supported on this device");
        }
        if count == 0 {
            return invalid("count is zero");
        }
        Ok(Timestamps {
            slots: PerFrame::try_new(self, |_| {
                Ok::<_, Error>(Queries {
                    raw: native::Timestamps::new(&self.device, count, label)?,
                    written: vec![false; count as usize].into(),
                })
            })?,
            count,
            frame: 0,
            results: vec![None; count as usize].into(),
            ticks: vec![None; count as usize].into(),
            label: label.into(),
        })
    }
}

impl Timestamps {
    /// Begins using these timestamps in `frame`, returning the ones the last
    /// frame using its slot wrote, as times since an arbitrary point, or
    /// `None` where it wrote none, or wrote one in an encoder it did not
    /// submit. Call once per frame before writing them;
    /// later calls in the same frame return the same values.
    ///
    /// Differences between timestamps measure GPU time; they are only
    /// meaningful within one submission or between submissions known to
    /// have run back to back.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the timestamps belong to another
    /// device, or a native error if they cannot be read.
    pub fn begin(&mut self, frame: &Frame<'_>) -> Result<&[Option<Duration>]> {
        self.check_device(&frame.rhi().device)?;
        if self.frame == frame.number() {
            return Ok(&self.results);
        }
        let queries = self.slots.get_mut(frame);
        // Beginning `frame` waited for the slot's previous frame, so what it
        // wrote is complete.
        queries.raw.read(&mut self.ticks)?;
        let period = queries.raw.period();
        for ((result, &ticks), &written) in self
            .results
            .iter_mut()
            .zip(&*self.ticks)
            .zip(&*queries.written)
        {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "timestamps are positive, and nanoseconds need no more than 52 bits of \
                          precision"
            )]
            let nanoseconds = |ticks: u64| (ticks as f64 * period) as u64;
            *result = ticks
                .filter(|_| written)
                .map(|ticks| Duration::from_nanos(nanoseconds(ticks)));
        }
        queries.raw.reset();
        queries.written.fill(false);
        self.frame = frame.number();
        Ok(&self.results)
    }

    /// Timestamps per frame.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The timestamps' debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Rejects `device` unless the timestamps were created from it.
    fn check_device(&self, device: &native::Device) -> Result<()> {
        if !self.slots.is_from(device) {
            return Err(Error::invalid(
                "timestamps",
                &self.label,
                "belong to another device",
            ));
        }
        Ok(())
    }
}

impl CommandEncoder<'_, Graphics> {
    /// Writes timestamp `index` of `timestamps` after the commands recorded
    /// before it. On Vulkan, it is written once they have all finished. On
    /// Metal, it is sampled when the GPU reaches it, which nothing orders
    /// after the earlier passes finish, since Metal overlaps passes that do
    /// not depend on each other: differences between Metal timestamps
    /// approximate GPU time.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the timestamps belong to another
    /// device, [`Timestamps::begin`] was not called in this frame, `index`
    /// is out of range, or this frame already wrote it.
    pub fn write_timestamp(&mut self, timestamps: &mut Timestamps, index: u32) -> Result<()> {
        timestamps.check_device(self.device())?;
        let invalid = |reason: String| Err(Error::invalid("timestamps", &timestamps.label, reason));
        if timestamps.frame != self.frame.number() {
            return invalid(format!(
                "were not begun in frame {}; call `begin` first",
                self.frame.number()
            ));
        }
        if index >= timestamps.count {
            return invalid(format!(
                "index {index} is outside the {} timestamps",
                timestamps.count
            ));
        }
        let queries = timestamps.slots.get_mut(self.frame);
        let written = &mut queries.written[index as usize];
        if *written {
            return Err(Error::invalid(
                "timestamps",
                &timestamps.label,
                format!("timestamp {index} was already written this frame"),
            ));
        }
        *written = true;
        self.raw.write_timestamp(self.device(), &queries.raw, index);
        Ok(())
    }
}
