use ash::vk;
use dirk_platform::WindowId;
use dirk_rhi::Backend as _;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Result,
    resources::{
        device::RenderDevice,
        swapchain::{RenderImage, Swapchain},
    },
};

/// The renderer's representation of a platform window.
/// Holds the swapchain, surface & other related state.
/// Doesn't actually do any of the rendering of the game.
pub struct Window {
    id: WindowId,
    swapchain: Swapchain,

    // TODO: stop rendering when the window is occluded
    occluded: bool,
}

impl Window {
    pub fn build(device: &RenderDevice, plat_window: &dirk_platform::Window) -> Result<Self> {
        let surface = device.rhi.create_surface(dirk_rhi::SurfaceCreateInfo {
            display: plat_window.display_handle()?.as_raw(),
            window: plat_window.window_handle()?.as_raw(),
        })?;

        let window_size = plat_window.size();
        let size = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };

        let swapchain = Swapchain::build(device, &surface, size)?;

        Ok(Self {
            id: plat_window.id(),
            swapchain,
            occluded: false,
        })
    }
    /// Returns the window's ID
    pub fn id(&self) -> WindowId {
        self.id
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent()
    }
    pub fn next_image(&mut self) -> Result<RenderImage> {
        self.swapchain.acquire_next_image()
    }
    pub fn resize(&mut self, extent: vk::Extent2D) -> Result<()> {
        self.swapchain.recreate(extent)
    }
    pub fn present(&mut self, image: RenderImage) -> Result<()> {
        self.swapchain.present(image)
    }
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }
}
