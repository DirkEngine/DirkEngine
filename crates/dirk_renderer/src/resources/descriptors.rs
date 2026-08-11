//! Safe descriptor layout and descriptor pool helpers.

mod allocator;
mod set;

pub mod layouts;
pub mod sets;

pub use allocator::DescriptorAllocator;
pub use set::DescriptorSet;
