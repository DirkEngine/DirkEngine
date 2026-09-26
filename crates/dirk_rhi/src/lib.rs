//! A compile-time-selected abstraction over Vulkan and Metal.
//! Resources have unique Rust owners. Dropping an owner queues native destruction;
//! the RHI collects it after at least three submission cycles and completion of
//! their graphics and transfer work. Record and submit within the same cycle.
//!
//! CPU mutation uses exclusive borrows. GPU recording, host access, and submission
//! have explicit unsafe contracts: callers provide synchronization, resource states,
//! valid non-owning bindings, and shader/resource bounds. No resource tracker or
//! implicit state reconciliation runs at submission. See `Rhi::finish_cycle`.
#![allow(
    unsafe_code,
    reason = "native GPU operations and their explicit caller contracts"
)]

mod backend;
mod command;
mod error;
mod flags;
mod presentation;
mod resource;
mod types;

pub use backend::{Capabilities, FormatCapabilities, RhiCreateInfo};
pub use command::{
    BufferBarrier, BufferCopy, BufferImageCopy, ColorAttachment, DependencyInfo, DepthAttachment,
    ImageBarrier, ImageBlit, ImageCopy, MemoryBarrier, RenderingInfo,
};
pub use error::{
    Error, InvalidResource, InvalidResourceKind, Result, ShaderLanguage, UnsupportedOperation,
};
pub use presentation::{SurfaceCreateInfo, SurfaceTarget, SwapchainDesc};
pub use raw_window_handle;
pub use resource::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingResource,
    BlendComponent, BlendState, BufferDesc, ColorTargetState, DepthBiasState, DepthState,
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
mod api;
pub use api::{BufferInfo, CopyQueue, Graphics, QueueKind, SamplerInfo, ViewMetadata};
pub(crate) use backend::{Api, Backend, NativeFence, NativeTimelineSemaphore, Submission};
pub(crate) use command::{NativeCommandBuffer, TimelinePoint};
pub(crate) use presentation::{NativeSurfaceFrame, NativeSwapchain};
pub(crate) use resource::NativeBuffer;

/// Device owning the platform backend, queues, and retirement cycles.
pub type Rhi = api::Rhi<native::SelectedBackend>;
/// Unique buffer allocation.
pub type Buffer = api::GpuBuffer<native::SelectedBackend>;
/// Unique image allocation.
pub type Image = api::GpuImage<native::SelectedBackend>;
/// Non-owning view of an image allocation.
pub type ImageView = api::GpuImageView<native::SelectedBackend>;
/// Sampling configuration.
pub type Sampler = api::GpuSampler<native::SelectedBackend>;
/// Compiled shader module.
pub type Shader = api::GpuShader<native::SelectedBackend>;
/// Immutable binding declarations.
pub type BindGroupLayout = api::GpuBindGroupLayout<native::SelectedBackend>;
/// Immutable resource bindings. Referenced resource owners must outlive GPU use.
pub type BindGroup = api::GpuBindGroup<native::SelectedBackend>;
/// Ordered binding layouts.
pub type PipelineLayout = api::GpuPipelineLayout<native::SelectedBackend>;
/// Graphics pipeline.
pub type GraphicsPipeline = api::GpuGraphicsPipeline<native::SelectedBackend>;
/// Platform presentation target.
pub type Surface = api::GpuSurface<native::SelectedBackend>;
/// Presentation image chain.
pub type Swapchain = api::GpuSwapchain<native::SelectedBackend>;
/// Acquired presentation frame.
pub type SurfaceFrame = api::GpuSurfaceFrame<native::SelectedBackend>;
/// Graphics or transfer recording scope.
pub type CommandEncoder<Q = Graphics> = api::CommandEncoder<native::SelectedBackend, Q>;
/// Finished single-submit commands.
pub type RecordedCommands<Q = Graphics> = api::RecordedCommands<native::SelectedBackend, Q>;
/// Borrowed rendering scope.
pub type RenderPass<'a> = api::RenderPass<'a, native::SelectedBackend>;
/// Semantic submission queue.
pub type Queue<Q = Graphics> = api::Queue<native::SelectedBackend, Q>;
/// Explicit submission dependencies.
pub type SubmitInfo<'a> = api::SubmitInfo<'a, native::SelectedBackend>;
/// Evidence of submitted work.
pub type Completion = api::Completion<native::SelectedBackend>;

mod native;

mod retirement;
