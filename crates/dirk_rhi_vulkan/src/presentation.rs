use std::{ffi::CStr, sync::Arc};

use ash::vk;
use dirk_rhi::{
    Error, Extent3d, ImageUsages, InvalidResource as Ir, PresentMode, Result, SurfaceCreateInfo,
    SurfaceFrame, SurfaceStatus, Swapchain, SwapchainDesc, TextureFormat,
};

use crate::{
    VulkanBackend, VulkanImage, VulkanImageView, convert,
    device::{Context, Garbage, Retained},
    vk_error,
};

#[derive(Clone)]
/// Vulkan presentation surface tied to the backend instance.
pub struct VulkanSurface(Arc<SurfaceInner>);

struct SurfaceInner {
    context: Arc<Context>,
    raw: vk::SurfaceKHR,
}

impl VulkanSurface {
    pub(crate) fn create(context: &Arc<Context>, info: SurfaceCreateInfo) -> Result<Self> {
        let required_extensions =
            ash_window::enumerate_required_extensions(info.display.into()).map_err(vk_error)?;
        if required_extensions.iter().any(|&extension| {
            let extension = unsafe { CStr::from_ptr(extension) }.to_string_lossy();
            !context
                .enabled_instance_extensions
                .contains(extension.as_ref())
        }) {
            return Err(Error::Backend(anyhow::anyhow!(
                "the RHI was not created with the instance extensions required by this surface"
            )));
        }
        let raw = unsafe {
            ash_window::create_surface(
                &context.entry,
                &context.instance,
                info.display.into(),
                info.window.into(),
                None,
            )
        }
        .map_err(vk_error)?;
        let supported = unsafe {
            context.surface_loader.get_physical_device_surface_support(
                context.physical_device,
                context.families.present,
                raw,
            )
        }
        .map_err(vk_error)?;
        if !supported {
            unsafe { context.surface_loader.destroy_surface(raw, None) };
            return Err(Error::Backend(anyhow::anyhow!(
                "the selected Vulkan queue family cannot present to this surface"
            )));
        }
        Ok(Self(Arc::new(SurfaceInner {
            context: context.clone(),
            raw,
        })))
    }

    #[must_use]
    /// Returns the native surface handle.
    pub fn raw(&self) -> vk::SurfaceKHR {
        self.0.raw
    }
}

impl Drop for SurfaceInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::Surface(self.raw));
    }
}

pub(crate) struct SwapchainGeneration {
    pub(crate) context: Arc<Context>,
    surface: VulkanSurface,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    semaphores: Vec<(vk::Semaphore, vk::Semaphore)>,
    format: TextureFormat,
    extent: Extent3d,
}

impl SwapchainGeneration {
    pub(crate) fn retain(self: &Arc<Self>) -> Retained {
        self.clone()
    }

    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "swapchain creation keeps queried capabilities, format preference, and presentation policy together"
    )]
    fn create(
        context: &Arc<Context>,
        surface: &VulkanSurface,
        width: u32,
        height: u32,
        usage: ImageUsages,
        preferred_formats: &[TextureFormat],
        present_mode: PresentMode,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Arc<Self>> {
        if !Arc::ptr_eq(context, &surface.0.context) {
            return Err(Ir::ForeignInstance.into());
        }
        let capabilities = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_capabilities(context.physical_device, surface.raw())
        }
        .map_err(vk_error)?;
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        let formats = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_formats(context.physical_device, surface.raw())
        }
        .map_err(vk_error)?;
        let surface_format = preferred_formats
            .iter()
            .find_map(|preferred| {
                let requested = convert::format(*preferred);
                formats
                    .iter()
                    .find(|format| {
                        format.format == requested
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .copied()
            })
            .or_else(|| {
                formats
                    .iter()
                    .find(|format| {
                        format.format == vk::Format::B8G8R8A8_SRGB
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .copied()
            })
            .or_else(|| {
                formats
                    .iter()
                    .find(|format| convert::rhi_format(format.format).is_some())
                    .copied()
            })
            .ok_or_else(|| {
                Error::Backend(anyhow::anyhow!(
                    "the surface exposes no color format representable by the RHI"
                ))
            })?;
        let format = convert::rhi_format(surface_format.format).ok_or_else(|| {
            Error::Backend(anyhow::anyhow!(
                "surface format is not represented by the RHI"
            ))
        })?;
        let present_modes = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_present_modes(context.physical_device, surface.raw())
        }
        .map_err(vk_error)?;
        let present_mode = if present_modes.contains(&convert::present_mode(present_mode)) {
            convert::present_mode(present_mode)
        } else {
            vk::PresentModeKHR::FIFO
        };
        let image_usage = convert::image_usage(usage);
        if image_usage.is_empty() || !capabilities.supported_usage_flags.contains(image_usage) {
            return Err(Error::Backend(anyhow::anyhow!(
                "the surface does not support the requested swapchain image usage"
            )));
        }
        let mut image_count = capabilities.min_image_count.saturating_add(1);
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let mut queue_families = vec![context.families.graphics, context.families.present];
        queue_families.sort_unstable();
        queue_families.dedup();
        let (sharing_mode, family_slice) = if queue_families.len() > 1 {
            (vk::SharingMode::CONCURRENT, queue_families.as_slice())
        } else {
            (vk::SharingMode::EXCLUSIVE, &[][..])
        };
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|mode| capabilities.supported_composite_alpha.contains(*mode))
        .ok_or_else(|| {
            Error::Backend(anyhow::anyhow!(
                "the surface exposes no supported composite alpha mode"
            ))
        })?;
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(family_slice)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);
        let raw = unsafe {
            context
                .swapchain_loader
                .create_swapchain(&create_info, None)
        }
        .map_err(vk_error)?;
        let images = match unsafe { context.swapchain_loader.get_swapchain_images(raw) } {
            Ok(images) => images,
            Err(error) => {
                unsafe { context.swapchain_loader.destroy_swapchain(raw, None) };
                return Err(vk_error(error));
            }
        };
        let mut views = Vec::with_capacity(images.len());
        for &image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            match unsafe { context.device.create_image_view(&view_info, None) } {
                Ok(view) => views.push(view),
                Err(error) => {
                    for view in views {
                        unsafe { context.device.destroy_image_view(view, None) };
                    }
                    unsafe { context.swapchain_loader.destroy_swapchain(raw, None) };
                    return Err(vk_error(error));
                }
            }
        }
        let mut semaphores = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            let create_info = vk::SemaphoreCreateInfo::default();
            let image_available =
                match unsafe { context.device.create_semaphore(&create_info, None) } {
                    Ok(semaphore) => semaphore,
                    Err(error) => {
                        destroy_partial(context, raw, views, semaphores);
                        return Err(vk_error(error));
                    }
                };
            let render_finished =
                match unsafe { context.device.create_semaphore(&create_info, None) } {
                    Ok(semaphore) => semaphore,
                    Err(error) => {
                        unsafe { context.device.destroy_semaphore(image_available, None) };
                        destroy_partial(context, raw, views, semaphores);
                        return Err(vk_error(error));
                    }
                };
            semaphores.push((image_available, render_finished));
        }
        Ok(Arc::new(Self {
            context: context.clone(),
            surface: surface.clone(),
            raw,
            images,
            views,
            semaphores,
            format,
            extent: Extent3d::new_2d(extent.width, extent.height),
        }))
    }
}

impl Drop for SwapchainGeneration {
    fn drop(&mut self) {
        let views = std::mem::take(&mut self.views);
        let semaphores = std::mem::take(&mut self.semaphores)
            .into_iter()
            .flat_map(|(available, finished)| [available, finished])
            .collect();
        self.context.retire(Garbage::Swapchain {
            raw: self.raw,
            views,
            semaphores,
        });
    }
}

/// Vulkan swapchain and its current recreatable generation.
pub struct VulkanSwapchain {
    generation: Arc<SwapchainGeneration>,
    usage: ImageUsages,
    preferred_formats: Vec<TextureFormat>,
    present_mode: PresentMode,
    semaphore_index: usize,
}

impl VulkanSwapchain {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &SwapchainDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        Ok(Self {
            generation: SwapchainGeneration::create(
                context,
                desc.surface,
                desc.width,
                desc.height,
                desc.usage,
                desc.preferred_formats,
                desc.present_mode,
                vk::SwapchainKHR::null(),
            )?,
            usage: desc.usage,
            preferred_formats: desc.preferred_formats.to_vec(),
            present_mode: desc.present_mode,
            semaphore_index: 0,
        })
    }

    #[must_use]
    /// Returns the native handle for the current swapchain generation.
    pub fn raw(&self) -> vk::SwapchainKHR {
        self.generation.raw
    }
}

impl Swapchain<VulkanBackend> for VulkanSwapchain {
    fn format(&self) -> TextureFormat {
        self.generation.format
    }

    fn extent(&self) -> Extent3d {
        self.generation.extent
    }

    fn acquire(&mut self) -> Result<VulkanSurfaceFrame> {
        let semaphore_index = self.semaphore_index;
        self.semaphore_index = (self.semaphore_index + 1) % self.generation.semaphores.len();
        let image_available = self.generation.semaphores[semaphore_index].0;
        let (image_index, suboptimal) = unsafe {
            self.generation.context.swapchain_loader.acquire_next_image(
                self.generation.raw,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        }
        .map_err(vk_error)?;
        let generation = self.generation.clone();
        let index = usize::try_from(image_index).map_err(|_| {
            Error::Backend(anyhow::anyhow!(
                "Vulkan returned an invalid swapchain image index"
            ))
        })?;
        let image = VulkanImage::surface(
            generation.clone(),
            generation.images[index],
            generation.format,
            generation.extent,
        );
        let view = VulkanImageView::surface(generation.clone(), generation.views[index]);
        Ok(VulkanSurfaceFrame {
            generation,
            image,
            view,
            image_index,
            semaphore_index,
            status: if suboptimal {
                SurfaceStatus::Suboptimal
            } else {
                SurfaceStatus::Optimal
            },
        })
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        let generation = SwapchainGeneration::create(
            &self.generation.context,
            &self.generation.surface,
            width,
            height,
            self.usage,
            &self.preferred_formats,
            self.present_mode,
            self.generation.raw,
        )?;
        self.generation = generation;
        self.semaphore_index = 0;
        Ok(())
    }

    fn present(&mut self, frame: VulkanSurfaceFrame) -> Result<SurfaceStatus> {
        if !Arc::ptr_eq(&self.generation.context, frame.context()) {
            return Err(Ir::ForeignInstance.into());
        }
        let waits = [frame.render_finished()];
        let swapchains = [frame.generation.raw];
        let image_indices = [frame.image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe {
            self.generation
                .context
                .swapchain_loader
                .queue_present(self.generation.context.queues.present, &present_info)
        } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(SurfaceStatus::Suboptimal),
            Ok(false) => Ok(SurfaceStatus::Optimal),
            Err(error) => Err(vk_error(error)),
        }
    }
}

/// Acquired Vulkan presentation image and its binary synchronization.
pub struct VulkanSurfaceFrame {
    pub(crate) generation: Arc<SwapchainGeneration>,
    image: VulkanImage,
    view: VulkanImageView,
    pub(crate) image_index: u32,
    pub(crate) semaphore_index: usize,
    status: SurfaceStatus,
}

impl VulkanSurfaceFrame {
    pub(crate) fn image_available(&self) -> vk::Semaphore {
        self.generation.semaphores[self.semaphore_index].0
    }

    pub(crate) fn render_finished(&self) -> vk::Semaphore {
        self.generation.semaphores[self.semaphore_index].1
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.generation.context
    }
}

impl SurfaceFrame<VulkanBackend> for VulkanSurfaceFrame {
    fn image(&self) -> &VulkanImage {
        &self.image
    }

    fn view(&self) -> &VulkanImageView {
        &self.view
    }

    fn format(&self) -> TextureFormat {
        self.generation.format
    }

    fn extent(&self) -> Extent3d {
        self.generation.extent
    }

    fn status(&self) -> SurfaceStatus {
        self.status
    }
}

fn destroy_partial(
    context: &Context,
    swapchain: vk::SwapchainKHR,
    views: Vec<vk::ImageView>,
    semaphores: Vec<(vk::Semaphore, vk::Semaphore)>,
) {
    unsafe {
        for view in views {
            context.device.destroy_image_view(view, None);
        }
        for (available, finished) in semaphores {
            context.device.destroy_semaphore(available, None);
            context.device.destroy_semaphore(finished, None);
        }
        context.swapchain_loader.destroy_swapchain(swapchain, None);
    }
}
