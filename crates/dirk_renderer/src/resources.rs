#![allow(unsafe_code)]

pub mod buffer;
pub mod command_pool;
pub mod descriptors;
pub mod device;
pub mod image;
pub mod swapchain;
pub mod sync;

pub(crate) type ActiveBackend = dirk_rhi_vulkan::VulkanBackend;
pub(crate) type ActiveRhi = dirk_rhi::Rhi<ActiveBackend>;
