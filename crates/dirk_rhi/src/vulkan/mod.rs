//! Vulkan 1.3 backend.

mod adapter;
mod buffer;
mod convert;
mod device;
mod image;
mod instance;
mod sampler;

pub(crate) use buffer::Buffer;
pub(crate) use device::Device;
pub(crate) use image::{Image, ImageView};
pub(crate) use sampler::Sampler;

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
