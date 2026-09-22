//! Vulkan 1.3 implementation of the RHI.
//!
//! Native types remain private to the platform-selected RHI implementation.

mod backend;
mod command;
mod convert;
mod device;
mod presentation;
mod resource;

pub use backend::VulkanBackend;
pub use resource::{
    VulkanBindGroup, VulkanBuffer, VulkanFence, VulkanGraphicsPipeline, VulkanImage,
    VulkanImageView, VulkanPipelineLayout,
};

use crate::Error;

fn vk_error(error: ash::vk::Result) -> Error {
    match error {
        ash::vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
        ash::vk::Result::ERROR_OUT_OF_DATE_KHR => Error::SwapchainOutOfDate,
        ash::vk::Result::ERROR_SURFACE_LOST_KHR => Error::SurfaceLost,
        error => Error::Backend(anyhow::anyhow!("Vulkan operation failed: {error:?}")),
    }
}

fn backend_error(error: impl Into<anyhow::Error>) -> Error {
    Error::Backend(error.into())
}
