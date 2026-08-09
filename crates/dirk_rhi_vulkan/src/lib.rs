//! Vulkan 1.3 implementation of [`dirk_rhi`].
//!
//! Native handles are exposed only on concrete Vulkan types so renderer
//! integrations can opt in without leaking Vulkan through the neutral RHI.

mod backend;
mod command;
mod convert;
mod device;
mod presentation;
mod resource;

pub use backend::VulkanBackend;
pub use command::{VulkanCommandBuffer, VulkanCommandPool};
pub use presentation::{VulkanSurface, VulkanSurfaceFrame, VulkanSwapchain};
pub use resource::{
    VulkanBindGroup, VulkanBindGroupLayout, VulkanBuffer, VulkanFence, VulkanGraphicsPipeline,
    VulkanImage, VulkanImageView, VulkanPipelineLayout, VulkanSampler, VulkanShader,
    VulkanTimelineSemaphore,
};

use dirk_rhi::Error;

fn vk_error(error: ash::vk::Result) -> Error {
    match error {
        ash::vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
        ash::vk::Result::ERROR_OUT_OF_DATE_KHR => Error::SurfaceOutOfDate,
        error => Error::Backend(anyhow::anyhow!("Vulkan operation failed: {error:?}")),
    }
}

fn backend_error(error: impl Into<anyhow::Error>) -> Error {
    Error::Backend(error.into())
}
