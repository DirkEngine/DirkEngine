use std::sync::Arc;

use dirk_platform::WindowId;
use dirk_rhi::{Backend as _, Extent3d, TextureFormat};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Result,
    resources::{
        ActiveRhi,
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
    pub fn build(rhi: &Arc<ActiveRhi>, plat_window: &dirk_platform::Window) -> Result<Self> {
        // The platform window outlives the surface created from these handles.
        let (display, handle) = unsafe {
            (
                raw_window_handle::DisplayHandle::borrow_raw(
                    plat_window.display_handle()?.as_raw(),
                ),
                raw_window_handle::WindowHandle::borrow_raw(plat_window.window_handle()?.as_raw()),
            )
        };
        let surface = rhi.create_surface(dirk_rhi::SurfaceCreateInfo {
            display,
            window: handle,
        })?;

        let window_size = plat_window.size();
        let size = Extent3d::new_2d(window_size.width, window_size.height);

        let swapchain = Swapchain::build(rhi, &surface, size)?;

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
    pub fn extent(&self) -> Extent3d {
        self.swapchain.extent()
    }
    pub fn format(&self) -> TextureFormat {
        self.swapchain.format()
    }
    pub fn next_image(&mut self) -> Result<RenderImage> {
        self.swapchain.acquire_next_image()
    }
    pub fn resize(&mut self, extent: Extent3d) -> Result<()> {
        self.swapchain.recreate(extent)
    }
    pub fn present(&mut self, image: RenderImage) -> Result<()> {
        self.swapchain.present(image)
    }
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }
}
