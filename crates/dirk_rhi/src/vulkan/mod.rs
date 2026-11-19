//! Vulkan 1.3 backend.

mod adapter;
mod convert;
mod device;
mod instance;

pub(crate) use device::Device;

use crate::{Error, Result};
use ash::vk;

/// Converts native results into RHI results that name the failed operation.
trait Check<T> {
    /// Maps a native error from `operation`, keeping device loss distinct.
    fn check(self, operation: &'static str) -> Result<T>;
}

impl<T> Check<T> for ash::prelude::VkResult<T> {
    fn check(self, operation: &'static str) -> Result<T> {
        self.map_err(|result| match result {
            vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            result => Error::native(operation, result),
        })
    }
}
