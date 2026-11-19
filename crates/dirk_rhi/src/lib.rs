//! The engine's render hardware interface (RHI).
//!
//! One native backend is compiled per target and wrapped by the safe types in
//! this crate. There are no backend traits or generic backend parameters: each
//! native module exposes the same crate-private types and methods, and the
//! public types hold a native object plus the immutable information needed to
//! validate its use. Requests are validated once, when objects are created.
//!
//! The binding design lives in `docs/rhi/architecture.md` at the repository
//! root.

mod access;
mod buffer;
mod command;
mod device;
mod error;
mod format;
mod frame;
mod graveyard;
mod heap;
mod image;
mod pass;
mod pipeline;
mod presentation;
mod queue;
mod sampler;
mod shader;

// Every target uses Vulkan, Apple targets through MoltenVK, until the Metal
// backend selects `metal/mod.rs` on Apple targets.
#[path = "vulkan/mod.rs"]
mod native;

pub use access::{BufferAccess, ImageAccess, ShaderStages};
pub use buffer::{Buffer, BufferDesc, BufferInfo, BufferUsage, Memory};
pub use command::{
    BufferBarrier, BufferCopy, BufferImageCopy, CommandEncoder, ImageBarrier, ImageCopy,
};
pub use device::{Capabilities, Features, Limits, Rhi, RhiCreateInfo};
pub use dirk_shaders::bindless::{
    BufferIndex, BufferSlice, HeapIndex, HeapIndices, SamplerIndex, StorageImageIndex, TextureIndex,
};
pub use error::{Error, Result};
pub use format::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};
pub use frame::{FRAMES_IN_FLIGHT, Frame, FrameSlice, PerFrame};
pub use image::{
    Extent3d, Image, ImageDesc, ImageDimension, ImageInfo, ImageRange, ImageRegion, ImageView,
    Origin3d, SampleCount, ViewType,
};
pub use pass::{
    ColorAttachment, DepthAttachment, IndexFormat, Indices, LoadOp, Rect, RenderPass,
    RenderPassDesc, StoreOp, Viewport,
};
pub use pipeline::{
    BlendComponent, BlendFactor, BlendOp, BlendState, ColorTarget, ColorWrites, CompareOp,
    CullMode, DepthBias, DepthState, FrontFace, GraphicsPipeline, GraphicsPipelineDesc,
    MAX_COLOR_TARGETS, Topology,
};
pub use presentation::{
    AcquiredImage, PresentMode, Surface, SurfaceTarget, Swapchain, SwapchainDesc,
};
pub use queue::{Completion, CopyQueue, Graphics, QueueKind, QueueTransfer, QueueType};
pub use sampler::{AddressMode, Filter, Sampler, SamplerDesc, SamplerInfo};
pub use shader::{MAX_PUSH_DATA_SIZE, PushData, Shader, ShaderModule, ShaderStage};
