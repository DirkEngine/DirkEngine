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

mod device;
mod error;
mod format;

// Every target uses Vulkan, Apple targets through MoltenVK, until the Metal
// backend selects `metal/mod.rs` on Apple targets.
#[path = "vulkan/mod.rs"]
mod native;

pub use device::{Capabilities, Features, Limits, Rhi, RhiCreateInfo};
pub use error::{Error, Result};
pub use format::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};
