use std::sync::Arc;

use ash::vk;

use super::{Check, CommandBuffer, Device, device::Garbage};
use crate::Result;

/// A Vulkan timestamp query pool, reset from the host.
pub(crate) struct Timestamps {
    device: Arc<Device>,
    raw: vk::QueryPool,
    /// Number of queries.
    count: u32,
    /// Results read back: a value and an availability word per query, which
    /// lets unwritten queries be read without waiting for them.
    results: Vec<[u64; 2]>,
}

impl Timestamps {
    /// Creates `count` timestamp queries, all reset.
    pub(crate) fn new(device: &Arc<Device>, count: u32, label: &str) -> Result<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        // SAFETY: `info` is valid for the device.
        let raw =
            unsafe { device.raw.create_query_pool(&info, None) }.check("vkCreateQueryPool")?;
        device.name(raw, label);
        let timestamps = Self {
            device: device.clone(),
            raw,
            count,
            results: vec![[0; 2]; count as usize],
        };
        timestamps.reset();
        Ok(timestamps)
    }

    /// Reads every timestamp into `ticks`, without the bits the queue does
    /// not write. Unwritten ones read as zero.
    ///
    /// No pending work may write them.
    pub(crate) fn read(&mut self, ticks: &mut [u64]) -> Result<()> {
        // SAFETY: the queries exist, and no pending work writes them.
        let read = unsafe {
            self.device.raw.get_query_pool_results(
                self.raw,
                0,
                &mut self.results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        match read {
            Ok(()) | Err(vk::Result::NOT_READY) => {}
            Err(error) => return Err(error).check("vkGetQueryPoolResults"),
        }
        for (ticks, &[value, available]) in ticks.iter_mut().zip(&self.results) {
            *ticks = if available == 0 {
                0
            } else {
                value & self.device.timestamp_mask
            };
        }
        Ok(())
    }

    /// Resets every timestamp, so they can be written again.
    ///
    /// No pending work may use them.
    pub(crate) fn reset(&self) {
        // SAFETY: host query reset is enabled with timestamps, and no
        // pending work uses these queries.
        unsafe { self.device.raw.reset_query_pool(self.raw, 0, self.count) };
    }

    /// Nanoseconds per tick.
    pub(crate) fn period(&self) -> f64 {
        f64::from(self.device.timestamp_period)
    }
}

impl Drop for Timestamps {
    fn drop(&mut self) {
        self.device.bury(Garbage::QueryPool(self.raw));
    }
}

impl CommandBuffer {
    /// Writes timestamp `query` once every earlier command has finished.
    pub(crate) fn write_timestamp(self, device: &Device, timestamps: &Timestamps, query: u32) {
        // SAFETY: the query was reset since its last write, as the encoder
        // checked, and the pool outlives the frame's work.
        unsafe {
            device.raw.cmd_write_timestamp2(
                self.raw,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                timestamps.raw,
                query,
            );
        }
    }
}
