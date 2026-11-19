//! This module holds a safe abstraction to the [`vk::Queue`]s used by the renderer.

use ash::{Device, Instance, khr::swapchain, prelude::VkResult, vk};

use crate::{physical_device::QueueFamilyIndices, resources::sync::Fence};

/// The type of queue you would like to submit to.
#[derive(Copy, Clone)]
pub enum QueueType {
    Graphics,
    Transfer,
    #[expect(dead_code, reason = "legacy compute pool has no current caller")]
    Compute,
}

/// The device's queues.
///
/// Vulkan requires external synchronization of a queue on submit and present.
/// `submit` and `present` take `&self` and lock nothing: they rely on every
/// call coming from the renderer thread.
pub struct Queues {
    device: Device,
    swapchain_loader: swapchain::Device,
    graphics: vk::Queue,
    compute: vk::Queue,
    transfer: vk::Queue,
    present: vk::Queue,
}

impl Queues {
    pub fn new(instance: &Instance, device: &Device, indices: &QueueFamilyIndices) -> Self {
        // SAFETY: `create_device` requested one queue from each of these
        // families, so queue 0 exists in each.
        let queue = |family| unsafe { device.get_device_queue(family, 0) };
        Self {
            device: device.clone(),
            swapchain_loader: swapchain::Device::new(instance, device),
            graphics: queue(indices.graphics),
            present: queue(indices.present),
            compute: queue(indices.compute),
            transfer: queue(indices.transfer),
        }
    }

    pub fn submit(
        &self,
        queue_type: QueueType,
        submits: &[vk::SubmitInfo],
        fence: Option<&Fence>,
    ) -> VkResult<()> {
        let queue = match queue_type {
            QueueType::Compute => self.compute,
            QueueType::Graphics => self.graphics,
            QueueType::Transfer => self.transfer,
        };
        let fence = fence.map_or_else(Fence::null_handle, Fence::raw);
        // SAFETY: the queue and fence come from `self.device`; the submit infos
        // and the slices they point to outlive the call. The renderer thread
        // is the only one submitting (see `Queues`).
        unsafe { self.device.queue_submit(queue, submits, fence) }
    }

    /// Returns the raw Vulkan queue for integrations that record their own
    /// short-lived uploads.
    #[cfg_attr(
        not(feature = "editor"),
        expect(
            dead_code,
            reason = "Only the editor's egui renderer records its own commands."
        )
    )]
    pub fn raw(&self, queue_type: QueueType) -> vk::Queue {
        match queue_type {
            QueueType::Compute => self.compute,
            QueueType::Graphics => self.graphics,
            QueueType::Transfer => self.transfer,
        }
    }

    pub fn present(&self, info: &vk::PresentInfoKHR) -> VkResult<bool> {
        // SAFETY: the present queue comes from this device and `info` outlives
        // the call. The renderer thread is the only one presenting (see `Queues`).
        unsafe { self.swapchain_loader.queue_present(self.present, info) }
    }
}
