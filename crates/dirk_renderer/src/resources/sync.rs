//! Renderer timeline synchronization.
use crate::{
    Result,
    resources::{ActiveRhi, ActiveTimelineSemaphore},
};

/// Renderer timeline semaphore used to order viewport output.
#[derive(Clone)]
pub struct TimelineSemaphore {
    inner: ActiveTimelineSemaphore,
}

impl TimelineSemaphore {
    /// Creates a timeline semaphore with the supplied initial value.
    pub fn create(rhi: &ActiveRhi, initial_value: u64) -> Result<Self> {
        Ok(Self {
            inner: rhi.create_timeline_semaphore(initial_value)?,
        })
    }

    /// Waits until the timeline reaches `value`.
    #[allow(unused)]
    pub fn wait(&self, value: u64, timeout: u64) -> Result<()> {
        self.inner.wait(value, timeout)?;
        Ok(())
    }

    /// Returns the current timeline value.
    #[allow(unused)]
    pub fn value(&self) -> Result<u64> {
        Ok(self.inner.value()?)
    }

    pub(crate) fn rhi(&self) -> &ActiveTimelineSemaphore {
        &self.inner
    }
}
