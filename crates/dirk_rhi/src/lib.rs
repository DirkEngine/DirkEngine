#![doc = include_str!("../README.md")]

mod access;
mod buffer;
mod command;
mod device;
mod error;
mod format;
mod frame;
mod graveyard;
mod image;
mod queue;
mod sampler;

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
pub use error::{Error, Result};
pub use format::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};
pub use frame::{FRAMES_IN_FLIGHT, Frame, FrameSlice, PerFrame};
pub use image::{
    Extent3d, Image, ImageDesc, ImageDimension, ImageInfo, ImageRange, ImageRegion, ImageView,
    Origin3d, SampleCount, ViewType,
};
pub use queue::{Completion, CopyQueue, Graphics, QueueKind, QueueTransfer, QueueType};
pub use sampler::{AddressMode, Filter, Sampler, SamplerDesc, SamplerInfo};
