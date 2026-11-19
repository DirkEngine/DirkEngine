#![expect(
    unsafe_code,
    reason = "Every resource wraps a Vulkan object created and destroyed through ash."
)]

pub mod buffer;
pub mod command_pool;
pub mod descriptors;
pub mod device;
pub mod image;
pub mod queues;
pub mod swapchain;
pub mod sync;
