//! Vulkan 1.3 backend.

mod adapter;
mod buffer;
mod command;
mod convert;
mod device;
mod image;
mod instance;
mod queue;
mod sampler;

pub(crate) use buffer::Buffer;
pub(crate) use command::{CommandBuffer, CommandPool};
pub(crate) use device::Device;
pub(crate) use image::{Image, ImageId, ImageView};
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

/// Implements `is_from` for native objects holding their device.
macro_rules! device_children {
    ($($object:ty),+) => {$(
        impl $object {
            /// Whether this object was created from `device`.
            pub(crate) fn is_from(&self, device: &Device) -> bool {
                std::ptr::eq(&*self.device, device)
            }
        }
    )+};
}

device_children!(Buffer, Image);
