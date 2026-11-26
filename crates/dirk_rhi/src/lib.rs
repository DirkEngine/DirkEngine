#![doc = include_str!("../README.md")]

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

/// Implements `is_from` for native objects holding their device.
macro_rules! device_children {
    ($($object:ty),+) => {$(
        impl $object {
            /// Whether this object was created from `device`.
            pub(crate) fn is_from(&self, device: &Device) -> bool {
                std::ptr::eq(&*self.device, device)
            }
        }
    )+};
}

// One native backend per target: Metal on Apple targets (macOS and iOS),
// Vulkan elsewhere.
#[expect(
    unsafe_code,
    reason = "The backend calls the native graphics API, and every unsafe block says why its call is sound."
)]
#[cfg_attr(any(target_os = "macos", target_os = "ios"), path = "metal/mod.rs")]
#[cfg_attr(
    not(any(target_os = "macos", target_os = "ios")),
    path = "vulkan/mod.rs"
)]
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
