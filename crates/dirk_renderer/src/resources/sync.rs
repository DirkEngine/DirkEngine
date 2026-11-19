//! Synchronization resource wrappers.

use ash::{Device, prelude::VkResult, vk};

use crate::resources::device::{Garbage, RenderDevice};

/// RAII wrapper for a Vulkan fence.
pub struct Fence {
    device: Device,
    raw: vk::Fence,
}

impl Fence {
    /// Creates a fence with the supplied creation flags.
    pub fn create(device: &Device, flags: vk::FenceCreateFlags) -> VkResult<Self> {
        let info = vk::FenceCreateInfo::default().flags(flags);
        let raw = unsafe { device.create_fence(&info, None)? };

        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    /// Creates an unsignaled fence.
    pub fn unsignaled(device: &Device) -> VkResult<Self> {
        Self::create(device, vk::FenceCreateFlags::empty())
    }

    /// Creates a signaled fence.
    pub fn signaled(device: &Device) -> VkResult<Self> {
        Self::create(device, vk::FenceCreateFlags::SIGNALED)
    }

    /// Returns the raw Vulkan fence handle.
    pub fn raw(&self) -> vk::Fence {
        self.raw
    }

    /// Returns the raw null fence handle for Vulkan calls that take an optional fence.
    pub(crate) fn null_handle() -> vk::Fence {
        vk::Fence::null()
    }

    /// Waits for this fence to become signaled.
    pub fn wait(&self, timeout: u64) -> VkResult<()> {
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&self.raw), true, timeout)
        }
    }

    /// Resets this fence to the unsignaled state.
    pub fn reset(&self) -> VkResult<()> {
        unsafe { self.device.reset_fences(std::slice::from_ref(&self.raw)) }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.raw, None);
        }
    }
}

/// RAII wrapper for a Vulkan timeline semaphore.
pub struct TimelineSemaphore {
    device: RenderDevice,
    raw: vk::Semaphore,
}

impl TimelineSemaphore {
    /// Creates a timeline semaphore with the supplied initial counter value.
    pub fn create(device: &RenderDevice, initial_value: u64) -> VkResult<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let raw = unsafe { device.device.create_semaphore(&info, None)? };

        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    /// Returns the raw Vulkan semaphore handle.
    pub fn raw(&self) -> vk::Semaphore {
        self.raw
    }

    /// Waits until the timeline semaphore reaches at least `value`.
    #[expect(
        dead_code,
        reason = "Nothing waits on a timeline value from the CPU yet."
    )]
    pub fn wait(&self, value: u64, timeout: u64) -> VkResult<()> {
        let semaphores = [self.raw];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.device.device.wait_semaphores(&info, timeout) }
    }

    /// Returns the current timeline counter value.
    #[expect(dead_code, reason = "Nothing reads a timeline value from the CPU yet.")]
    pub fn value(&self) -> VkResult<u64> {
        unsafe { self.device.device.get_semaphore_counter_value(self.raw) }
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        self.device.destroy(Garbage::Semaphore(self.raw));
    }
}
