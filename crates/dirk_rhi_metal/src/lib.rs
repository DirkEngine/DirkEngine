//! Metal implementation of [`dirk_rhi`].
//!
//! The implementation is available on Apple targets. MSL shaders use a simple
//! binding convention: bind-group buffers, textures, and samplers are flattened
//! independently by group and binding number, while vertex buffers start at
//! Metal buffer index 16.
//!
//! Metal's native mipmap generator implements linear, same-image mip blits.
//! Cross-image blits are supported when source and destination extents match;
//! scaled cross-image blits return [`dirk_rhi::Error::Unsupported`].

#[cfg(target_vendor = "apple")]
mod backend;
#[cfg(target_vendor = "apple")]
mod command;
#[cfg(target_vendor = "apple")]
mod convert;
#[cfg(target_vendor = "apple")]
mod presentation;
#[cfg(target_vendor = "apple")]
mod resource;

#[cfg(target_vendor = "apple")]
pub use backend::MetalBackend;
#[cfg(target_vendor = "apple")]
pub use command::{MetalCommandBuffer, MetalCommandPool};
#[cfg(target_vendor = "apple")]
pub use presentation::{MetalSurface, MetalSurfaceFrame, MetalSwapchain};
#[cfg(target_vendor = "apple")]
pub use resource::{
    MetalBindGroup, MetalBindGroupLayout, MetalBuffer, MetalFence, MetalGraphicsPipeline,
    MetalImage, MetalImageView, MetalPipelineLayout, MetalSampler, MetalShader,
    MetalTimelineSemaphore,
};

#[cfg(target_vendor = "apple")]
fn backend_error(error: impl std::fmt::Display) -> dirk_rhi::Error {
    dirk_rhi::Error::Backend(anyhow::anyhow!("{error}"))
}
