#![allow(unsafe_code)]

use ash::{Device, Entry, khr::surface, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashSet;
use tracing::info;

use crate::{
    DEVICE_EXTENSIONS, Error, MAX_FRAMES_IN_FLIGHT, Renderer, Result, physical_device,
    resources::{command_pool::CommandPool, device::RenderDevice, sync::Fence},
    utils::{Frame, RendererProperties},
};

impl Renderer {
    pub(crate) fn select_physical_device(
        entry: &Entry,
        instance: &ash::Instance,
        window: &dirk_platform::Window,
    ) -> Result<(vk::PhysicalDevice, RendererProperties)> {
        let (loader, surface) = Self::create_surface(entry, instance, window)?;
        let selection = Self::query_physical_device(instance, &loader, surface);

        unsafe { loader.destroy_surface(surface, None) };

        selection
    }

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

    pub(crate) fn query_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, RendererProperties)> {
        let (device_info, queues) = physical_device::PhysicalDeviceSelector::new()
            .require_extensions(DEVICE_EXTENSIONS)
            .require(|info| info.vulkan12_features.vulkan_memory_model == vk::TRUE)
            .require(|info| info.vulkan12_features.timeline_semaphore == vk::TRUE)
            .select(instance, surface_loader, surface)
            .ok_or(Error::NoDeviceFound)?;

        info!(
            "Physical device selected: {:#?} (vendor: {}, id: {}, api: {}, driver: {})",
            device_info
                .properties
                .device_name_as_c_str()
                .unwrap_or_default(),
            device_info.properties.vendor_id,
            device_info.properties.device_id,
            device_info.properties.api_version,
            device_info.properties.driver_version
        );

        Ok((
            device_info.handle,
            RendererProperties {
                msaa_samples: Self::msaa_samples(&device_info),
                anisotropy: device_info.features.sampler_anisotropy == vk::TRUE,
                surface_format: Self::surface_format(surface_loader, device_info.handle, surface)?,
                queue_family_indices: queues,
                depth_format: Self::depth_format(instance, device_info.handle)?,
                present_mode: Self::present_mode(surface_loader, device_info.handle, surface)?,
            },
        ))
    }

    pub(crate) fn surface_format(
        surface_loader: &surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceFormatKHR> {
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };

        Ok(formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(formats[0]))
    }

    pub(crate) fn depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::Format> {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

        candidates
            .iter()
            .find(|&&format| {
                let properties = unsafe {
                    instance.get_physical_device_format_properties(physical_device, format)
                };
                properties.optimal_tiling_features.contains(features)
            })
            .copied()
            .ok_or(Error::NoSupportedFormat)
    }

    pub(crate) fn msaa_samples(
        device_info: &physical_device::PhysicalDeviceInfo,
    ) -> vk::SampleCountFlags {
        let counts = device_info
            .properties
            .limits
            .framebuffer_color_sample_counts
            & device_info
                .properties
                .limits
                .framebuffer_depth_sample_counts;

        [
            vk::SampleCountFlags::TYPE_64,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ]
        .iter()
        .find(|&flag| counts.contains(*flag))
        .copied()
        .unwrap_or(vk::SampleCountFlags::TYPE_1)
    }

    pub(crate) fn present_mode(
        surface_loader: &surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::PresentModeKHR> {
        let modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        Ok(*modes
            .iter()
            .find(|&mode| *mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO))
    }

    pub(crate) fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        properties: &RendererProperties,
    ) -> Result<Device> {
        let unique_families: HashSet<u32> = [
            properties.queue_family_indices.graphics,
            properties.queue_family_indices.present,
            properties.queue_family_indices.compute,
            properties.queue_family_indices.transfer,
        ]
        .iter()
        .copied()
        .collect();

        let queue_priorities = [1.0_f32];
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        let physical_device_features =
            vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .vulkan_memory_model(true)
            .timeline_semaphore(true);
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let extensions: Vec<*const i8> = DEVICE_EXTENSIONS
            .iter()
            .map(|name| name.as_ptr().cast())
            .collect();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&extensions)
            .push_next(&mut vulkan12_features)
            .push_next(&mut vulkan13_features);

        Ok(unsafe { instance.create_device(physical_device, &device_create_info, None)? })
    }

    pub(crate) fn build_frames(
        device: &Device,
        render_device: &RenderDevice,
    ) -> Result<[Frame; MAX_FRAMES_IN_FLIGHT]> {
        let build_frame = || -> Result<Frame> {
            let command_pool = CommandPool::build(
                device,
                &render_device.properties.queue_family_indices,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;
            let fence = Fence::signaled(device)?;
            Ok(Frame {
                command_pool,
                submitted_command_buffers: Vec::new(),
                fence,
            })
        };

        Ok([build_frame()?, build_frame()?])
    }
}
