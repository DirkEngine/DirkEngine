//! This module holds a safe abstraction to the [`vk::Queue`]s used by the renderer.

use ash::{Device, Instance, khr::swapchain, prelude::VkResult, vk};

use crate::{physical_device::QueueFamilyIndices, resources::sync::Fence};

/// The type of queue you would like to submit to.
#[derive(Copy, Clone)]
pub enum QueueType {
    Graphics,
    Transfer,
}

pub struct Queues {
    device: Device,
    swapchain_loader: swapchain::Device,
    graphics: vk::Queue,
    transfer: vk::Queue,
    present: vk::Queue,
}

impl Queues {
    pub fn new(instance: &Instance, device: &Device, indices: &QueueFamilyIndices) -> Self {
        Self {
            device: device.clone(),
            swapchain_loader: swapchain::Device::new(instance, device),
            graphics: unsafe { device.get_device_queue(indices.graphics, 0) },
            present: unsafe { device.get_device_queue(indices.present, 0) },
            transfer: unsafe { device.get_device_queue(indices.transfer, 0) },
        }
    }

    pub fn submit(
        &self,
        queue_type: QueueType,
        submits: &[vk::SubmitInfo],
        fence: Option<&Fence>,
    ) -> VkResult<()> {
        let queue = match queue_type {
            QueueType::Graphics => self.graphics,
            QueueType::Transfer => self.transfer,
        };
        let fence = fence.map_or_else(Fence::null_handle, Fence::raw);
        unsafe { self.device.queue_submit(queue, submits, fence) }
    }

    /// Returns the raw Vulkan queue for integrations that record their own
    /// short-lived uploads.
    #[cfg_attr(not(feature = "editor"), allow(unused))]
    pub fn raw(&self, queue_type: QueueType) -> vk::Queue {
        match queue_type {
            QueueType::Graphics => self.graphics,
            QueueType::Transfer => self.transfer,
        }
    }

    pub fn present(&self, info: &vk::PresentInfoKHR) -> VkResult<bool> {
        unsafe { self.swapchain_loader.queue_present(self.present, info) }
    }
}
