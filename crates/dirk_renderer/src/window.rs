//! Window policy and presentation orchestration, using the RHI swapchain directly.
use crate::Result;
use dirk_platform::WindowId;
use dirk_rhi::{
    ColorSpace, Extent3d, PresentMode, Rhi, Surface, SurfaceFormat, SurfaceFrame, Swapchain,
    SwapchainDesc, TextureFormat,
};
use std::num::NonZeroU32;

const PREFERRED_FORMATS: [SurfaceFormat; 2] = [
    SurfaceFormat {
        texture: TextureFormat::Bgra8Srgb,
        color_space: ColorSpace::Srgb,
    },
    SurfaceFormat {
        texture: TextureFormat::Rgba8Srgb,
        color_space: ColorSpace::Srgb,
    },
];

pub struct Window {
    id: WindowId,
    /// `None` until the surface has a presentable extent. Declared before
    /// `surface` so the chain is dropped first.
    swapchain: Option<Swapchain>,
    surface: Surface,
    #[cfg(not(feature = "editor"))]
    pub(crate) presenter: crate::presentation::Presenter,
    requested_extent: Extent3d,
    occluded: bool,
    recreate: bool,
}
impl Window {
    pub fn build(rhi: &Rhi, window: &dirk_platform::Window) -> Result<Self> {
        let surface =
            rhi.create_surface(dirk_rhi::SurfaceCreateInfo::new(window.surface_target()))?;
        let size = window.size();
        let requested_extent = Extent3d::new_2d(size.width, size.height);
        let swapchain = Self::create_swapchain(rhi, &surface, requested_extent)?;
        let window = Self {
            #[cfg(not(feature = "editor"))]
            presenter: crate::presentation::Presenter::new(
                rhi,
                swapchain
                    .as_ref()
                    .map_or(PREFERRED_FORMATS[0].texture, |swapchain| {
                        swapchain.format().texture
                    }),
            )?,
            id: window.id(),
            swapchain,
            surface,
            requested_extent,
            occluded: false,
            recreate: false,
        };
        Ok(window)
    }

    /// Creates a chain for `surface`, or returns `None` while the surface has
    /// no presentable extent, such as a minimized window.
    fn create_swapchain(
        rhi: &Rhi,
        surface: &Surface,
        extent: Extent3d,
    ) -> Result<Option<Swapchain>> {
        let (Some(width), Some(height)) = (
            NonZeroU32::new(extent.width),
            NonZeroU32::new(extent.height),
        ) else {
            return Ok(None);
        };
        match rhi.create_swapchain(&SwapchainDesc {
            label: "renderer window",
            surface,
            width,
            height,
            usage: dirk_rhi::ImageUsages::COLOR_ATTACHMENT
                | dirk_rhi::ImageUsages::COPY_DST
                | dirk_rhi::ImageUsages::PRESENT,
            preferred_formats: &PREFERRED_FORMATS,
            desired_image_count: NonZeroU32::new(3),
            present_mode: PresentMode::Mailbox,
        }) {
            Ok(swapchain) => Ok(Some(swapchain)),
            Err(dirk_rhi::Error::SwapchainOutOfDate) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub fn id(&self) -> WindowId {
        self.id
    }
    pub fn extent(&self) -> Extent3d {
        self.requested_extent
    }
    /// Format of the window's images, or the preferred format while the
    /// window has no swapchain yet.
    pub fn format(&self) -> TextureFormat {
        self.swapchain
            .as_ref()
            .map_or(PREFERRED_FORMATS[0].texture, |swapchain| {
                swapchain.format().texture
            })
    }
    pub fn resize(&mut self, extent: Extent3d) {
        self.requested_extent = extent;
        self.recreate = true;
    }

    /// Acquires the next image, or returns `None` when the window cannot
    /// present this frame; presentation is retried on later frames.
    pub fn next_image(&mut self, rhi: &Rhi) -> Result<Option<SurfaceFrame>> {
        let (Some(width), Some(height)) = (
            NonZeroU32::new(self.requested_extent.width),
            NonZeroU32::new(self.requested_extent.height),
        ) else {
            return Ok(None);
        };
        if self.occluded {
            return Ok(None);
        }
        match &mut self.swapchain {
            // A new chain already has the requested extent.
            None => {
                self.swapchain = Self::create_swapchain(rhi, &self.surface, self.requested_extent)?;
            }
            Some(swapchain) if self.recreate => match swapchain.resize(width, height) {
                Ok(()) => {}
                // The surface has no presentable extent yet.
                Err(dirk_rhi::Error::SwapchainOutOfDate) => return Ok(None),
                Err(error) => return Err(error.into()),
            },
            Some(_) => {}
        }
        let Some(swapchain) = &mut self.swapchain else {
            return Ok(None);
        };
        self.recreate = false;
        #[cfg(not(feature = "editor"))]
        self.presenter
            .set_target_format(rhi, swapchain.format().texture)?;
        match swapchain.acquire(u64::MAX) {
            Ok(frame) => {
                self.recreate = frame.status() == dirk_rhi::SurfaceStatus::Suboptimal;
                Ok(Some(frame))
            }
            Err(dirk_rhi::Error::SwapchainOutOfDate) => {
                self.recreate = true;
                Ok(None)
            }
            // No image became available in time; try again next frame.
            Err(dirk_rhi::Error::Timeout) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
    pub fn present(&mut self, image: SurfaceFrame) -> Result<()> {
        let Some(swapchain) = &mut self.swapchain else {
            return Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::BadState).into());
        };
        match swapchain.present(image) {
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
