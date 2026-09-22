//! Window policy and presentation orchestration, using the RHI swapchain directly.
use crate::Result;
use dirk_platform::WindowId;
use dirk_rhi::{
    ColorSpace, Extent3d, PresentMode, Rhi, SurfaceFormat, SurfaceFrame, Swapchain, SwapchainDesc,
    TextureFormat,
};
use std::num::NonZeroU32;

pub struct Window {
    id: WindowId,
    swapchain: Swapchain,
    requested_extent: Extent3d,
    occluded: bool,
    recreate: bool,
}
impl Window {
    pub fn build(rhi: &Rhi, window: &dirk_platform::Window) -> Result<Self> {
        let surface =
            rhi.create_surface(dirk_rhi::SurfaceCreateInfo::new(window.surface_target()))?;
        let size = window.size();
        let extent = Extent3d::new_2d(size.width, size.height);
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
        let swapchain = rhi.create_swapchain(&SwapchainDesc {
            label: "renderer window",
            surface: &surface,
            width: NonZeroU32::new(size.width.max(1)).expect("clamped"),
            height: NonZeroU32::new(size.height.max(1)).expect("clamped"),
            usage: dirk_rhi::ImageUsages::COLOR_ATTACHMENT
                | dirk_rhi::ImageUsages::COPY_DST
                | dirk_rhi::ImageUsages::PRESENT,
            preferred_formats: &preferred_formats,
            desired_image_count: NonZeroU32::new(3),
            present_mode: PresentMode::Mailbox,
        })?;
        Ok(Self {
            id: window.id(),
            swapchain,
            requested_extent: extent,
            occluded: false,
            recreate: false,
        })
    }
    pub fn id(&self) -> WindowId {
        self.id
    }
    pub fn extent(&self) -> Extent3d {
        self.requested_extent
    }
    pub fn format(&self) -> TextureFormat {
        self.swapchain.format().texture
    }
    pub fn resize(&mut self, extent: Extent3d) {
        self.requested_extent = extent;
        self.recreate = true;
    }
    pub fn next_image(&mut self) -> Result<Option<SurfaceFrame>> {
        let (Some(width), Some(height)) = (
            NonZeroU32::new(self.requested_extent.width),
            NonZeroU32::new(self.requested_extent.height),
        ) else {
            return Ok(None);
        };
        if self.occluded {
            return Ok(None);
        }
        if self.recreate {
            self.swapchain.resize(width, height)?;
            self.recreate = false;
        }
        match self.swapchain.acquire(u64::MAX) {
            Ok(frame) => {
                self.recreate = frame.status() == dirk_rhi::SurfaceStatus::Suboptimal;
                Ok(Some(frame))
            }
            Err(dirk_rhi::Error::SwapchainOutOfDate) => {
                self.recreate = true;
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }
    pub fn present(&mut self, image: SurfaceFrame) -> Result<()> {
        match self.swapchain.present(image) {
            Ok(status) => self.recreate |= status == dirk_rhi::SurfaceStatus::Suboptimal,
            Err(dirk_rhi::Error::SwapchainOutOfDate) => self.recreate = true,
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }
}
