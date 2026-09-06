//! Renderer presentation wrappers backed by the active RHI.

use std::{num::NonZeroU32, sync::Arc};

use dirk_rhi::{
    Backend as _, ColorSpace, Extent3d, SurfaceFormat, SurfaceFrame as _, Swapchain as _,
    SwapchainDesc, TextureFormat,
};

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

    pub fn format(&self) -> TextureFormat {
        self.inner.format().texture
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
        let width = NonZeroU32::new(window_size.width)
            .ok_or(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty))?;
        let height = NonZeroU32::new(window_size.height)
            .ok_or(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty))?;
        let preferred_formats = [
            SurfaceFormat {
                texture: TextureFormat::Bgra8Unorm,
                color_space: ColorSpace::Srgb,
            },
            SurfaceFormat {
                texture: TextureFormat::Rgba8Unorm,
                color_space: ColorSpace::Srgb,
            },
        ];
        let inner = rhi.create_swapchain(&SwapchainDesc {
            label: "renderer window swapchain",
            surface,
            width,
            height,
            usage: dirk_rhi::ImageUsages::COLOR_ATTACHMENT
                | dirk_rhi::ImageUsages::COPY_DST
                | dirk_rhi::ImageUsages::PRESENT,
            preferred_formats: &preferred_formats,
            desired_image_count: NonZeroU32::new(3),
            present_mode: dirk_rhi::PresentMode::Mailbox,
        })?;
        Ok(Self {
            rhi: rhi.clone(),
            inner,
        })
    }

    pub fn extent(&self) -> Extent3d {
        self.inner.extent()
    }

    pub fn format(&self) -> TextureFormat {
        self.inner.format().texture
    }

    pub fn acquire_next_image(&mut self) -> Result<RenderImage> {
        Ok(RenderImage {
            inner: self.inner.acquire(u64::MAX)?,
        })
    }

    pub fn recreate(&mut self, window_size: Extent3d) -> Result<()> {
        self.rhi.wait_idle()?;
        let width = NonZeroU32::new(window_size.width)
            .ok_or(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty))?;
        let height = NonZeroU32::new(window_size.height)
            .ok_or(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty))?;
        self.inner.resize(width, height)?;
        Ok(())
    }

    /// Presents an image acquired from this swapchain.
    pub fn present(&mut self, image: RenderImage) -> Result<()> {
        self.inner.present(image.inner)?;
        Ok(())
    }
}
