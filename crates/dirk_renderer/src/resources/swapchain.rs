//! Renderer presentation wrappers backed by the active RHI.

use std::sync::Arc;

use ash::vk;
use dirk_rhi::{Backend as _, SurfaceFrame as _, Swapchain as _, SwapchainDesc};
use dirk_rhi_vulkan::{VulkanSurface, VulkanSurfaceFrame, VulkanSwapchain};

use crate::{
    Result,
    frame_graph::ImportedTexture,
    resources::{ActiveRhi, device::RenderDevice},
};

/// Acquired image from an RHI swapchain.
pub struct RenderImage {
    inner: VulkanSurfaceFrame,
}

impl RenderImage {
    /// Imports the acquired image into the renderer graph.
    pub fn import(&self) -> ImportedTexture {
        ImportedTexture {
            image: self.inner.image().clone(),
            view: self.inner.view().clone(),
            aspects: dirk_rhi::ImageAspects::COLOR,
            initial_state: dirk_rhi::ImageState::Undefined,
            final_state: dirk_rhi::ImageState::Present,
        }
    }

    pub(crate) fn rhi(&self) -> &VulkanSurfaceFrame {
        &self.inner
    }

    pub fn format(&self) -> dirk_rhi::Format {
        self.inner.format()
    }
}

/// Reconfigurable renderer swapchain.
pub struct Swapchain {
    rhi: Arc<ActiveRhi>,
    inner: VulkanSwapchain,
    extent: vk::Extent2D,
}

impl Swapchain {
    /// Creates a swapchain for a renderer window.
    pub fn build(
        device: &RenderDevice,
        surface: &VulkanSurface,
        window_size: vk::Extent2D,
    ) -> Result<Self> {
        let inner = device.rhi.create_swapchain(&SwapchainDesc {
            label: "renderer window swapchain",
            surface,
            width: window_size.width,
            height: window_size.height,
            usage: dirk_rhi::ImageUsages::COLOR_ATTACHMENT
                | dirk_rhi::ImageUsages::COPY_DST
                | dirk_rhi::ImageUsages::PRESENT,
        })?;
        Ok(Self {
            rhi: device.rhi.clone(),
            inner,
            extent: window_size,
        })
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn acquire_next_image(&mut self) -> Result<RenderImage> {
        let inner = self.inner.acquire()?;
        let extent = inner.extent();
        self.extent = vk::Extent2D {
            width: extent.width,
            height: extent.height,
        };
        Ok(RenderImage { inner })
    }

    pub fn recreate(&mut self, window_size: vk::Extent2D) -> Result<()> {
        self.rhi.wait_idle()?;
        self.inner.resize(window_size.width, window_size.height)?;
        self.extent = window_size;
        Ok(())
    }

    /// Presents an image acquired from this swapchain.
    pub fn present(&mut self, image: RenderImage) -> Result<()> {
        self.inner.present(image.inner)?;
        Ok(())
    }
}
