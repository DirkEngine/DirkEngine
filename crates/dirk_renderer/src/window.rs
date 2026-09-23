use ash::vk;
use dirk_platform::WindowId;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Result,
    resources::{
        device::{Garbage, RenderDevice},
        swapchain::{RenderImage, Swapchain},
    },
};

/// The renderer's representation of a platform window.
/// Holds the swapchain, surface & other related state.
/// Doesn't actually do any of the rendering of the game.
pub struct Window {
    device: RenderDevice,
    surface_target: std::sync::Arc<dirk_platform::WindowSurfaceTarget>,

    id: WindowId,
    surface: vk::SurfaceKHR,
    swapchain: Swapchain,

    // TODO: stop rendering when the window is occluded
    occluded: bool,
}

impl Window {
    pub fn build(device: &RenderDevice, plat_window: &dirk_platform::Window) -> Result<Self> {
        let surface_target = plat_window.surface_target();
        let surface = unsafe {
            ash_window::create_surface(
                &device.entry,
                &device.instance,
                surface_target.display_handle()?.as_raw(),
                surface_target.window_handle()?.as_raw(),
                None,
            )?
        };

        let window_size = plat_window.size();
        let size = vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        };

        let swapchain = match Swapchain::build(device, surface, size) {
            Ok(swapchain) => swapchain,
            Err(error) => {
                // The target remains alive until the failed surface is destroyed.
                unsafe { device.surface_loader.destroy_surface(surface, None) };
                return Err(error);
            }
        };

        Ok(Self {
            id: plat_window.id(),
            device: device.clone(),
            surface_target,
            surface,
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
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        self.swapchain.destroy();
        self.device.destroy(Garbage::Surface {
            surface: self.surface,
            target: self.surface_target.clone(),
        });
    }
}
