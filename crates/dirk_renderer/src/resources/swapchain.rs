//! Renderer presentation wrappers backed by the active RHI.

use std::sync::Arc;

use dirk_rhi::{Backend as _, Extent3d, Format, SurfaceFrame as _, Swapchain as _, SwapchainDesc};

use crate::{
    Result,
    frame_graph::ImportedTexture,
    resources::{ActiveRhi, ActiveSurface, ActiveSurfaceFrame, ActiveSwapchain},
};

/// Acquired image from an RHI swapchain.
pub struct RenderImage {
    inner: ActiveSurfaceFrame,
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

    pub(crate) fn rhi(&self) -> &ActiveSurfaceFrame {
        &self.inner
    }

    pub fn format(&self) -> Format {
        self.inner.format()
    }
}

/// Reconfigurable renderer swapchain.
pub struct Swapchain {
    rhi: Arc<ActiveRhi>,
    inner: ActiveSwapchain,
}

impl Swapchain {
    /// Creates a swapchain for a renderer window.
    pub fn build(
        rhi: &Arc<ActiveRhi>,
        surface: &ActiveSurface,
        window_size: Extent3d,
    ) -> Result<Self> {
        let inner = rhi.create_swapchain(&SwapchainDesc {
            label: "renderer window swapchain",
            surface,
            width: window_size.width,
            height: window_size.height,
            usage: dirk_rhi::ImageUsages::COLOR_ATTACHMENT
                | dirk_rhi::ImageUsages::COPY_DST
                | dirk_rhi::ImageUsages::PRESENT,
        })?;
        Ok(Self {
            rhi: rhi.clone(),
            inner,
        })
    }

    pub fn extent(&self) -> Extent3d {
        self.inner.extent()
    }

    pub fn format(&self) -> Format {
        self.inner.format()
    }

    pub fn acquire_next_image(&mut self) -> Result<RenderImage> {
        Ok(RenderImage {
            inner: self.inner.acquire()?,
        })
    }

    pub fn recreate(&mut self, window_size: Extent3d) -> Result<()> {
        self.rhi.wait_idle()?;
        self.inner.resize(window_size.width, window_size.height)?;
        Ok(())
    }

    /// Presents an image acquired from this swapchain.
    pub fn present(&mut self, image: RenderImage) -> Result<()> {
        self.inner.present(image.inner)?;
        Ok(())
    }
}
