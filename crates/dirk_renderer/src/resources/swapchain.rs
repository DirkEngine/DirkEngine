//! Renderer presentation wrappers backed by the active RHI.

use ash::vk;
use dirk_rhi::{SurfaceFrame as _, SwapchainDesc};
use dirk_rhi_vulkan::{VulkanSurface, VulkanSurfaceFrame, VulkanSwapchain};

use crate::{
    Result,
    frame_graph::ImportedTexture,
    resources::{ActiveRhi, device::RenderDevice},
};

/// Acquired image from an RHI swapchain.
pub struct RenderImage {
    rhi: ActiveRhi,
    inner: VulkanSurfaceFrame,
}

impl RenderImage {
    /// Presents this acquired image.
    pub fn present(self) -> Result<()> {
        self.rhi.present(self.inner)?;
        Ok(())
    }

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
    rhi: ActiveRhi,
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
        let inner = self.rhi.acquire_frame(&mut self.inner)?;
        let extent = inner.extent();
        self.extent = vk::Extent2D {
            width: extent.width,
            height: extent.height,
        };
        Ok(RenderImage {
            rhi: self.rhi.clone(),
            inner,
        })
    }

    pub fn recreate(&mut self, window_size: vk::Extent2D) -> Result<()> {
        self.rhi.wait_idle()?;
        self.rhi
            .resize_swapchain(&mut self.inner, window_size.width, window_size.height)?;
        self.extent = window_size;
        Ok(())
    }
}
