//! Safe backend-neutral access to graphics devices.
#![allow(
    unsafe_code,
    reason = "the shared safe layer enforces the native backend contract"
)]
//!
//! [`Rhi`] owns portable validation, recording scopes, submission lifetimes, and
//! host access exclusion. [`Backend`] is the unsafe native implementation contract;
//! [`Api`] supplies the resource family used by borrowed descriptors.
//!
//! The render graph plans semantic resource dependencies. The RHI reconciles image
//! states at submission and keeps native objects alive through actual completion.
//! This first safe scheduler uses one native graphics queue for all semantic queue
//! capabilities; independently executing queues and compute dispatch remain future work.
//! Shader binaries are imported through an explicit unsafe trust boundary.

mod backend;
mod command;
mod error;
mod flags;
mod presentation;
mod resource;
mod types;

pub use backend::{
    Api, Backend, Capabilities, Fence, FormatCapabilities, RhiCreateInfo, Submission,
    TimelineSemaphore,
};
pub use command::{
    BufferBarrier, BufferCopy, BufferImageCopy, ColorAttachment, CommandBuffer, DependencyInfo,
    DepthAttachment, ImageBarrier, ImageBlit, ImageCopy, MemoryBarrier, RenderingInfo,
    TimelinePoint,
};
pub use error::{
    Error, InvalidResource, InvalidResourceKind, Result, ShaderLanguage, UnsupportedOperation,
};
pub use presentation::{SurfaceCreateInfo, SurfaceFrame, SurfaceTarget, Swapchain, SwapchainDesc};
pub use raw_window_handle;
pub use resource::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingResource,
    BlendComponent, BlendState, Buffer, BufferDesc, ColorTargetState, DepthBiasState, DepthState,
    GraphicsPipelineDesc, ImageDesc, ImageViewDesc, PipelineLayoutDesc, RasterState, SamplerDesc,
    ShaderDesc, ShaderSource, StencilFaceState, StencilState,
};
pub use types::{
    AccessTypes, AddressMode, BindingType, BlendFactor, BlendOp, BufferUsages, Color, ColorSpace,
    ColorWrites, CompareOp, CullMode, Extent3d, FilterMode, FrontFace, ImageAspects,
    ImageDimension, ImageState, ImageUsages, ImageViewType, IndexFormat, LoadOp, MemoryDomain,
    Origin3d, PipelineStages, PresentMode, PrimitiveTopology, QueueTransfer, QueueType, Rect,
    SampleCount, SampleCounts, ShaderStage, ShaderStages, StencilOp, StoreOp, SurfaceFormat,
    SurfaceStatus, TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode, Viewport,
};

#[cfg(test)]
mod tests;

mod access;
pub use access::*;
mod safe;
pub use safe::*;
