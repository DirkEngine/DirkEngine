//! Metal implementation of [`dirk_rhi`].
//!
//! The implementation is available on Apple targets. Shader bindings use the
//! shared [`dirk_rhi::BindingMap`], independently for each shader stage. Vertex
//! buffers start at Metal buffer index 16. Use [`dirk_rhi::Rhi`] for shared
//! resource retention, validation, and recording scopes.
//!
//! Exact-region filtered blits are currently unsupported. Callers query this
//! before recording and select an explicit fallback; whole-chain native mipmap
//! generation is not substituted for a regional operation.

#![cfg(target_vendor = "apple")]

mod backend;
mod command;
mod convert;
mod presentation;
mod resource;

pub use backend::MetalBackend;
pub use command::{MetalCommandBuffer, MetalCommandPool};
pub use presentation::{MetalSurface, MetalSurfaceFrame, MetalSwapchain};
pub use resource::{
    MetalBindGroup, MetalBindGroupLayout, MetalBuffer, MetalFence, MetalGraphicsPipeline,
    MetalImage, MetalImageView, MetalPipelineLayout, MetalSampler, MetalShader,
    MetalTimelineSemaphore,
};

fn backend_error(error: impl std::fmt::Display) -> dirk_rhi::Error {
    dirk_rhi::Error::Backend(anyhow::anyhow!("{error}"))
}
