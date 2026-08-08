//! Synchronization wrappers backed by the active RHI.

use dirk_rhi_vulkan::{VulkanFence, VulkanTimelineSemaphore};

use crate::{Result, resources::ActiveRhi};

/// Reusable per-frame completion fence.
pub struct Fence {
    rhi: ActiveRhi,
    inner: VulkanFence,
}

impl Fence {
    /// Creates a signaled fence.
    pub fn signaled(rhi: &ActiveRhi) -> Result<Self> {
        Ok(Self {
            inner: rhi.create_fence(true)?,
            rhi: rhi.clone(),
        })
    }

    /// Waits for the submission associated with this fence to complete.
    pub fn wait(&self, timeout: u64) -> Result<()> {
        self.rhi.wait_fence(&self.inner, timeout)?;
        Ok(())
    }

    /// Resets this fence before its next submission.
    pub fn reset(&self) -> Result<()> {
        self.rhi.reset_fence(&self.inner)?;
        Ok(())
    }

    /// Returns the backend fence used in an RHI submission.
    pub(crate) fn rhi(&self) -> &VulkanFence {
        &self.inner
    }
}

/// Renderer timeline semaphore used to order viewport output.
#[derive(Clone)]
pub struct TimelineSemaphore {
    rhi: ActiveRhi,
    inner: VulkanTimelineSemaphore,
}

impl TimelineSemaphore {
    /// Creates a timeline semaphore with the supplied initial value.
    pub fn create(rhi: &ActiveRhi, initial_value: u64) -> Result<Self> {
        Ok(Self {
            inner: rhi.create_timeline_semaphore(initial_value)?,
            rhi: rhi.clone(),
        })
    }

    /// Waits until the timeline reaches `value`.
    #[allow(unused)]
    pub fn wait(&self, value: u64, timeout: u64) -> Result<()> {
        self.rhi.wait_timeline(&self.inner, value, timeout)?;
        Ok(())
    }

    /// Returns the current timeline value.
    #[allow(unused)]
    pub fn value(&self) -> Result<u64> {
        Ok(self.rhi.timeline_value(&self.inner)?)
    }

    pub(crate) fn rhi(&self) -> &VulkanTimelineSemaphore {
        &self.inner
    }
}
