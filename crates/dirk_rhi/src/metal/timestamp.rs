use std::{ptr::NonNull, sync::Arc, time::Instant};

use objc2::{rc::autoreleasepool, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitPassDescriptor, MTLBuffer, MTLCommandBuffer,
    MTLCommonCounterSetTimestamp, MTLCounterDontSample, MTLCounterErrorValue,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLStorageMode,
};

use super::{Check, CommandBuffer, Device, Object, command::Encoder, device::Garbage};
use crate::Result;

/// How the device samples timestamps, and the clocks they are calibrated
/// against.
pub(super) struct Clock {
    /// The timestamp counter set.
    counters: Object<dyn MTLCounterSet>,
    /// Where timestamps are sampled.
    sampling: Sampling,
    /// The CPU and GPU clocks when the device was created. GPU ticks have
    /// no documented period, so it is measured against the CPU's.
    origin: (Instant, u64),
}

impl Clock {
    /// The timestamp clock of `device`, or `None` if it cannot sample
    /// timestamps between commands.
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        // SAFETY: the counter set name is a constant string.
        let name = unsafe { MTLCommonCounterSetTimestamp };
        let counters = device
            .counterSets()
            .and_then(|sets| sets.iter().find(|set| &*set.name() == name))?;
        let sampling = if device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            Sampling::StageBoundary
        } else if device.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary) {
            Sampling::BlitBoundary
        } else {
            return None;
        };
        Some(Self {
            counters: counters.into(),
            sampling,
            origin: (Instant::now(), Self::gpu_time(device)),
        })
    }

    /// The GPU's clock now, in ticks.
    fn gpu_time(device: &ProtocolObject<dyn MTLDevice>) -> u64 {
        let (mut cpu, mut gpu) = (0, 0);
        // SAFETY: both pointers are valid for writes.
        unsafe {
            device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu));
        };
        gpu
    }

    /// Nanoseconds per GPU tick, measured since the device was created, so
    /// increasingly accurate.
    fn period(&self, device: &ProtocolObject<dyn MTLDevice>) -> f64 {
        let (instant, origin) = self.origin;
        let ticks = Self::gpu_time(device).saturating_sub(origin);
        #[expect(
            clippy::cast_precision_loss,
            reason = "the period needs no more than 52 bits of precision"
        )]
        let (elapsed, ticks) = (instant.elapsed().as_nanos() as f64, ticks as f64);
        if ticks > 0.0 { elapsed / ticks } else { 1.0 }
    }
}

/// Where a device samples timestamps.
enum Sampling {
    /// At the start of encoders, as on Apple GPUs. Metal drops the samples
    /// of encoders without commands, so sampling passes fill the device's
    /// scratch buffer.
    StageBoundary,
    /// Between blits.
    BlitBoundary,
}

/// A Metal counter sample buffer of timestamps.
pub(crate) struct Timestamps {
    device: Arc<Device>,
    raw: Object<dyn MTLCounterSampleBuffer>,
    label: String,
    /// Nanoseconds per tick, measured when last read.
    period: f64,
}

impl Timestamps {
    /// Creates `count` timestamps. The device supports timestamps.
    pub(crate) fn new(device: &Arc<Device>, count: u32, label: &str) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            raw: Self::samples(device, count as usize, label)?,
            label: label.to_owned(),
            period: 1.0,
        })
    }

    /// A counter sample buffer of `count` unwritten timestamps.
    fn samples(
        device: &Device,
        count: usize,
        label: &str,
    ) -> Result<Object<dyn MTLCounterSampleBuffer>> {
        let clock = device
            .clock
            .as_ref()
            .check("finding the timestamp counter set")?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&*clock.counters));
        desc.setStorageMode(MTLStorageMode::Shared);
        desc.setLabel(&NSString::from_str(label));
        // SAFETY: Metal accepts any positive sample count up to its limit,
        // and reports larger ones as an error.
        unsafe { desc.setSampleCount(count) };
        device
            .raw
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .check("newCounterSampleBufferWithDescriptor")
            .map(Object::from)
    }

    /// Reads every timestamp into `ticks` and measures the tick period,
    /// then starts over with unwritten timestamps, since Metal cannot clear
    /// samples and keeps those of encoders never submitted unwritten.
    /// Unwritten ones, which Metal resolves to zero (Apple GPUs) or its
    /// error value, read as `None`.
    ///
    /// No pending work may write them.
    pub(crate) fn read(&mut self, ticks: &mut [Option<u64>]) -> Result<()> {
        let range = NSRange::new(0, ticks.len());
        // SAFETY: the range is inside the buffer, and no pending work writes
        // it.
        let data = autoreleasepool(|_| unsafe { self.raw.resolveCounterRange(range) })
            .check("resolveCounterRange")?
            .to_vec();
        let (results, _) = data.as_chunks::<{ size_of::<MTLCounterResultTimestamp>() }>();
        for (ticks, result) in ticks.iter_mut().zip(results) {
            *ticks = Some(u64::from_ne_bytes(*result))
                .filter(|&ticks| ticks != 0 && ticks != MTLCounterErrorValue);
        }
        if let Some(clock) = &self.device.clock {
            self.period = clock.period(&self.device.raw);
        }
        // No pending work uses the samples read, so they are released now.
        self.raw = Self::samples(&self.device, ticks.len(), &self.label)?;
        Ok(())
    }

    /// Nothing to reset: reading starts over with unwritten timestamps.
    #[expect(
        clippy::unused_self,
        reason = "Vulkan resets queries before they are written again"
    )]
    pub(crate) fn reset(&self) {}

    /// Nanoseconds per tick, as measured when last read.
    pub(crate) fn period(&self) -> f64 {
        self.period
    }
}

impl Drop for Timestamps {
    fn drop(&mut self) {
        self.device.bury(Garbage::object(self.raw.clone()));
    }
}

impl CommandBuffer {
    /// Writes timestamp `query` after the commands recorded before it: at
    /// the start of a blit pass filling the scratch buffer on GPUs sampling
    /// at stage boundaries, or between blits on others. Like any encoder,
    /// the pass waits only for the encoders before the last barrier, so
    /// earlier passes may still run when it samples.
    pub(crate) fn write_timestamp(&mut self, device: &Device, timestamps: &Timestamps, query: u32) {
        let sampling = device.clock.as_ref().map(|clock| &clock.sampling);
        if let Some(Sampling::StageBoundary) = sampling {
            let pass = MTLBlitPassDescriptor::new();
            // SAFETY: attachment zero always exists, and the sample index is
            // inside the buffer.
            unsafe {
                let attachment = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                attachment.setSampleBuffer(Some(&*timestamps.raw));
                attachment.setStartOfEncoderSampleIndex(query as usize);
                attachment.setEndOfEncoderSampleIndex(MTLCounterDontSample);
            }
            self.open(|commands| {
                Encoder::Blit(
                    commands
                        .blitCommandEncoderWithDescriptor(&pass)
                        .expect("Metal creates encoders for command buffers without an open one"),
                )
            });
            let scratch = &device.scratch;
            self.blit()
                .fillBuffer_range_value(scratch, NSRange::new(0, scratch.length()), 0);
            self.end_encoding();
        } else {
            // SAFETY: the sample index is inside the buffer.
            unsafe {
                self.blit()
                    .sampleCountersInBuffer_atSampleIndex_withBarrier(
                        &timestamps.raw,
                        query as usize,
                        true,
                    );
            }
        }
    }
}
