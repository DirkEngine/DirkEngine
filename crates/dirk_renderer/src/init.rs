use ash::{Entry, khr::surface, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Error, MAX_FRAMES_IN_FLIGHT, Renderer, Result,
    resources::{command_pool::CommandPool, device::RenderDevice, sync::Fence},
    utils::Frame,
};

impl Renderer {
    pub(crate) fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &dirk_platform::Window,
    ) -> Result<(surface::Instance, vk::SurfaceKHR)> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        Ok((surface::Instance::new(entry, instance), surface))
    }

    pub(crate) fn surface_format(
        surface_loader: &surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceFormatKHR> {
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or(Error::NoSupportedFormat)
    }

    pub(crate) fn build_frames(
        render_device: &RenderDevice,
    ) -> Result<[Frame; MAX_FRAMES_IN_FLIGHT]> {
        let build_frame = || -> Result<Frame> {
            Ok(Frame {
                command_pool: CommandPool::build(&render_device.rhi)?,
                submitted_command_buffers: Vec::new(),
                fence: Fence::signaled(&render_device.rhi)?,
            })
        };

        Ok([build_frame()?, build_frame()?])
    }
}
