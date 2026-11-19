//! Safe descriptor layout and descriptor pool helpers.

mod allocator;
mod set;
mod writer;

pub mod layouts;
pub mod sets;

pub use allocator::DescriptorAllocator;
pub use set::DescriptorSet;
pub use writer::DescriptorWriter;
