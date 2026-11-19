#![doc = include_str!("../README.md")]

mod device;
mod error;
mod format;

// Every target uses Vulkan, Apple targets through MoltenVK, until the Metal
// backend selects `metal/mod.rs` on Apple targets.
#[expect(
    unsafe_code,
    reason = "The backend calls the native graphics API, and every unsafe block says why its call is sound."
)]
#[path = "vulkan/mod.rs"]
mod native;

pub use device::{Capabilities, Features, Limits, Rhi, RhiCreateInfo};
pub use error::{Error, Result};
pub use format::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};
