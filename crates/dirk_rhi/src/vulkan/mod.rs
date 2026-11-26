//! Vulkan 1.3 backend.

mod adapter;
mod buffer;
mod command;
mod convert;
mod device;
mod heap;
mod image;
mod instance;
mod pass;
mod pipeline;
mod presentation;
mod queue;
mod sampler;

pub(crate) use buffer::{Buffer, BufferId};
pub(crate) use command::{CommandBuffer, CommandPool};
pub(crate) use device::Device;
pub(crate) use image::{Image, ImageId};
pub(crate) use pipeline::{GraphicsPipeline, PipelineId, ShaderModule};
pub(crate) use presentation::{Presentation, Semaphore, Surface, Swapchain};
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

device_children!(Buffer, Image, ShaderModule, GraphicsPipeline, Swapchain);
