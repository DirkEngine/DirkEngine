//! Compile-time backend selection.
#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(not(target_vendor = "apple"))]
mod vulkan;
#[cfg(target_vendor = "apple")]
pub use metal::MetalBackend as SelectedBackend;
#[cfg(not(target_vendor = "apple"))]
pub use vulkan::VulkanBackend as SelectedBackend;
