#![doc = include_str!("../README.md")]

mod buffer;
mod device;
mod error;
mod format;
mod image;
mod sampler;

// Every target uses Vulkan, Apple targets through MoltenVK, until the Metal
// backend selects `metal/mod.rs` on Apple targets.
#[expect(
    unsafe_code,
    reason = "The backend calls the native graphics API, and every unsafe block says why its call is sound."
)]
#[path = "vulkan/mod.rs"]
mod native;

pub use buffer::{Buffer, BufferDesc, BufferInfo, BufferUsage, Memory};
pub use device::{Capabilities, Features, Limits, Rhi, RhiCreateInfo};
pub use error::{Error, Result};
pub use format::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};
pub use image::{
    Extent3d, Image, ImageDesc, ImageDimension, ImageInfo, ImageView, SampleCount, ViewType,
};
pub use sampler::{AddressMode, Filter, Sampler, SamplerDesc, SamplerInfo};
