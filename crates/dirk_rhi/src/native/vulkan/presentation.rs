use std::{
    ffi::CStr,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    Error, Extent3d, ImageUsages, InvalidResourceKind as Ir, NativeSurfaceFrame, NativeSwapchain,
    PresentMode, Result, SurfaceCreateInfo, SurfaceFormat, SurfaceStatus, SwapchainDesc,
};
use ash::vk;
use parking_lot::Mutex;

use super::{
    VulkanBackend, VulkanFence, VulkanImage, VulkanImageView, convert,
    device::{Context, Garbage},
    vk_error,
};

#[derive(Clone)]
/// Vulkan presentation surface tied to the backend instance.
pub struct VulkanSurface(Arc<SurfaceInner>);

struct SurfaceInner {
    context: Arc<Context>,
    raw: vk::SurfaceKHR,
    target: Arc<dyn crate::SurfaceTarget>,
}

impl VulkanSurface {
    pub(crate) fn create(context: &Arc<Context>, info: &SurfaceCreateInfo) -> Result<Self> {
        let display = info.display_handle().map_err(|error| {
            Error::Backend(anyhow::anyhow!("display handle is unavailable: {error:?}"))
        })?;
        let window = info.window_handle().map_err(|error| {
            Error::Backend(anyhow::anyhow!("window handle is unavailable: {error:?}"))
        })?;
        let required_extensions =
            ash_window::enumerate_required_extensions(display.as_raw()).map_err(vk_error)?;
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
                display.as_raw(),
                window.as_raw(),
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
        };
        let supported = match supported {
            Ok(supported) => supported,
            Err(error) => {
                unsafe { context.surface_loader.destroy_surface(raw, None) };
                return Err(vk_error(error));
            }
        };
        if !supported {
            unsafe { context.surface_loader.destroy_surface(raw, None) };
            return Err(Error::Backend(anyhow::anyhow!(
                "the selected Vulkan queue family cannot present to this surface"
            )));
        }
        Ok(Self(Arc::new(SurfaceInner {
            context: context.clone(),
            raw,
            target: info.target().clone(),
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
        self.context
            .retire(Garbage::Surface(self.raw, self.target.clone()));
    }
}

#[derive(Default)]
struct AcquireSlot {
    acquired: bool,
    completion: Option<VulkanFence>,
}

pub(crate) struct SwapchainGeneration {
    pub(crate) context: Arc<Context>,
    surface: VulkanSurface,
    raw: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    semaphores: Vec<(vk::Semaphore, vk::Semaphore)>,
    acquisitions: Mutex<Vec<AcquireSlot>>,
    format: SurfaceFormat,
    extent: Extent3d,
    image_count: NonZeroU32,
    /// Images the application may hold while acquisition still makes progress.
    max_acquired: NonZeroU32,
}

/// Caller preferences reapplied whenever the swapchain is recreated.
struct SwapchainPolicy {
    usage: ImageUsages,
    preferred_formats: Vec<SurfaceFormat>,
    desired_image_count: Option<NonZeroU32>,
    present_mode: PresentMode,
}

impl SwapchainGeneration {
    fn create(
        context: &Arc<Context>,
        surface: &VulkanSurface,
        policy: &SwapchainPolicy,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Arc<Self>> {
        if !Arc::ptr_eq(context, &surface.0.context) {
            return Err(Ir::ForeignInstance
                .with_detail("surface belongs to another Vulkan device")
                .into());
        }
        let capabilities = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_capabilities(context.physical_device, surface.raw())
        }
        .map_err(vk_error)?;
        let extent = select_extent(&capabilities, width, height)?;
        let (surface_format, format) = select_format(context, surface, &policy.preferred_formats)?;
        let image_usage = convert::image_usage(policy.usage);
        if image_usage.is_empty() || !capabilities.supported_usage_flags.contains(image_usage) {
            return Err(Error::Backend(anyhow::anyhow!(
                "the surface does not support the requested swapchain image usage"
            )));
        }
        let mut image_count = policy.desired_image_count.map_or_else(
            || capabilities.min_image_count.saturating_add(1),
            NonZeroU32::get,
        );
        image_count = image_count.max(capabilities.min_image_count);
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let composite_alpha = select_composite_alpha(&capabilities)?;
        let mut queue_families = vec![context.families.graphics, context.families.present];
        queue_families.sort_unstable();
        queue_families.dedup();
        let (sharing_mode, family_slice) = if queue_families.len() > 1 {
            (vk::SharingMode::CONCURRENT, queue_families.as_slice())
        } else {
            (vk::SharingMode::EXCLUSIVE, &[][..])
        };
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
            .present_mode(select_present_mode(context, surface, policy.present_mode)?)
            .clipped(true)
            .old_swapchain(old_swapchain);
        let raw = unsafe {
            context
                .swapchain_loader
                .create_swapchain(&create_info, None)
        }
        .map_err(vk_error)?;
        let (images, views, semaphores) =
            match create_image_resources(context, raw, surface_format.format) {
                Ok(resources) => resources,
                Err(error) => {
                    unsafe { context.swapchain_loader.destroy_swapchain(raw, None) };
                    return Err(error);
                }
            };
        let Some(image_count) = u32::try_from(images.len()).ok().and_then(NonZeroU32::new) else {
            destroy_image_resources(context, views, semaphores);
            unsafe { context.swapchain_loader.destroy_swapchain(raw, None) };
            return Err(Error::Backend(anyhow::anyhow!(
                "Vulkan returned an unusable swapchain image count"
            )));
        };
        // VUID-vkAcquireNextImageKHR-surface-07783: an unbounded acquire is
        // only guaranteed to progress while at most `images - minImageCount`
        // images are already held.
        let max_acquired = image_count
            .get()
            .checked_sub(capabilities.min_image_count)
            .and_then(|spare| NonZeroU32::new(spare + 1))
            .unwrap_or(NonZeroU32::MIN);
        Ok(Arc::new(Self {
            context: context.clone(),
            surface: surface.clone(),
            raw,
            acquisitions: Mutex::new((0..images.len()).map(|_| AcquireSlot::default()).collect()),
            images,
            views,
            semaphores,
            format,
            extent: Extent3d::new_2d(extent.width, extent.height),
            image_count,
            max_acquired,
        }))
    }
}

/// Uses the surface's fixed extent, or the requested one clamped to its bounds.
fn select_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> Result<vk::Extent2D> {
    let extent = if capabilities.current_extent.width == u32::MAX {
        vk::Extent2D {
            width: width
                .max(capabilities.min_image_extent.width)
                .min(capabilities.max_image_extent.width),
            height: height
                .max(capabilities.min_image_extent.height)
                .min(capabilities.max_image_extent.height),
        }
    } else {
        capabilities.current_extent
    };
    if extent.width == 0 || extent.height == 0 {
        // A minimized window has no presentable extent; retry once it is visible.
        return Err(Error::SwapchainOutOfDate);
    }
    Ok(extent)
}

fn select_composite_alpha(
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> Result<vk::CompositeAlphaFlagsKHR> {
    [
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
    })
}

/// Picks the first supported preference, then sRGB BGRA8, then any representable format.
fn select_format(
    context: &Context,
    surface: &VulkanSurface,
    preferred_formats: &[SurfaceFormat],
) -> Result<(vk::SurfaceFormatKHR, SurfaceFormat)> {
    let formats = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, surface.raw())
    }
    .map_err(vk_error)?;
    let find = |predicate: &dyn Fn(&vk::SurfaceFormatKHR) -> bool| {
        formats.iter().copied().find(|format| predicate(format))
    };
    preferred_formats
        .iter()
        .find_map(|preferred| {
            find(&|format| {
                format.format == convert::format(preferred.texture)
                    && format.color_space == convert::color_space(preferred.color_space)
            })
        })
        .or_else(|| {
            find(&|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
        })
        .into_iter()
        .chain(formats.iter().copied())
        .find_map(|native| {
            Some((
                native,
                SurfaceFormat {
                    texture: convert::rhi_format(native.format)?,
                    color_space: convert::rhi_color_space(native.color_space)?,
                },
            ))
        })
        .ok_or_else(|| {
            Error::Backend(anyhow::anyhow!(
                "the surface exposes no color format representable by the RHI"
            ))
        })
}

/// Uses the requested mode when available and FIFO, which is always supported, otherwise.
fn select_present_mode(
    context: &Context,
    surface: &VulkanSurface,
    requested: PresentMode,
) -> Result<vk::PresentModeKHR> {
    let modes = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_present_modes(context.physical_device, surface.raw())
    }
    .map_err(vk_error)?;
    let requested = convert::present_mode(requested);
    Ok(if modes.contains(&requested) {
        requested
    } else {
        vk::PresentModeKHR::FIFO
    })
}

type ImageResources = (
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    Vec<(vk::Semaphore, vk::Semaphore)>,
);

/// Creates per-image views and binary semaphores, destroying partial results on failure.
fn create_image_resources(
    context: &Context,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
) -> Result<ImageResources> {
    let images =
        unsafe { context.swapchain_loader.get_swapchain_images(swapchain) }.map_err(vk_error)?;
    let mut views = Vec::with_capacity(images.len());
    for &image in &images {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
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
                destroy_image_resources(context, views, Vec::new());
                return Err(vk_error(error));
            }
        }
    }
    let mut semaphores = Vec::with_capacity(images.len());
    let create_info = vk::SemaphoreCreateInfo::default();
    for _ in 0..images.len() {
        let pair =
            unsafe { context.device.create_semaphore(&create_info, None) }.and_then(|available| {
                match unsafe { context.device.create_semaphore(&create_info, None) } {
                    Ok(finished) => Ok((available, finished)),
                    Err(error) => {
                        unsafe { context.device.destroy_semaphore(available, None) };
                        Err(error)
                    }
                }
            });
        match pair {
            Ok(pair) => semaphores.push(pair),
            Err(error) => {
                destroy_image_resources(context, views, semaphores);
                return Err(vk_error(error));
            }
        }
    }
    Ok((images, views, semaphores))
}

fn destroy_image_resources(
    context: &Context,
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
    policy: SwapchainPolicy,
    next_acquisition: usize,
}

impl VulkanSwapchain {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &SwapchainDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        let policy = SwapchainPolicy {
            usage: desc.usage,
            preferred_formats: desc.preferred_formats.to_vec(),
            desired_image_count: desc.desired_image_count,
            present_mode: desc.present_mode,
        };
        Ok(Self {
            generation: SwapchainGeneration::create(
                context,
                desc.surface,
                &policy,
                desc.width.get(),
                desc.height.get(),
                vk::SwapchainKHR::null(),
            )?,
            policy,
            next_acquisition: 0,
        })
    }

    #[must_use]
    /// Returns the native handle for the current swapchain generation.
    pub fn raw(&self) -> vk::SwapchainKHR {
        self.generation.raw
    }

    /// Replaces the current generation, retiring the old one.
    fn recreate(&mut self, width: u32, height: u32) -> Result<()> {
        let generation = SwapchainGeneration::create(
            &self.generation.context,
            &self.generation.surface,
            &self.policy,
            width,
            height,
            self.generation.raw,
        )?;
        self.generation = generation;
        self.next_acquisition = 0;
        Ok(())
    }
}

unsafe impl NativeSwapchain<VulkanBackend> for VulkanSwapchain {
    fn format(&self) -> SurfaceFormat {
        self.generation.format
    }

    fn extent(&self) -> Extent3d {
        self.generation.extent
    }

    fn image_count(&self) -> NonZeroU32 {
        self.generation.image_count
    }

    fn max_acquired_frames(&self) -> NonZeroU32 {
        self.generation.max_acquired
    }

    unsafe fn acquire(&mut self, timeout_ns: u64) -> Result<VulkanSurfaceFrame> {
        let mut acquisitions = self.generation.acquisitions.lock();
        let semaphore_index = (0..acquisitions.len())
            .map(|offset| (self.next_acquisition + offset) % acquisitions.len())
            .find(|&index| !acquisitions[index].acquired)
            .ok_or_else(|| {
                Ir::BadState.with_detail("all swapchain acquisition slots are in use")
            })?;
        let slot = &mut acquisitions[semaphore_index];
        if let Some(completion) = &slot.completion {
            // The prior submission must have consumed this binary semaphore.
            unsafe { crate::NativeFence::wait(completion, timeout_ns) }?;
        }
        let image_available = self.generation.semaphores[semaphore_index].0;
        let (image_index, suboptimal) = unsafe {
            self.generation.context.swapchain_loader.acquire_next_image(
                self.generation.raw,
                timeout_ns,
                image_available,
                vk::Fence::null(),
            )
        }
        .map_err(vk_error)?;
        self.next_acquisition = (semaphore_index + 1) % self.generation.semaphores.len();
        slot.acquired = true;
        slot.completion = None;
        let generation = self.generation.clone();
        let index = usize::try_from(image_index).map_err(|_| {
            Error::Backend(anyhow::anyhow!(
                "Vulkan returned an invalid swapchain image index"
            ))
        })?;
        let image = VulkanImage::surface(
            generation.clone(),
            generation.images[index],
            generation.format.texture,
            generation.extent,
        );
        let view = VulkanImageView::surface(generation.clone(), generation.views[index]);
        Ok(VulkanSurfaceFrame {
            generation,
            resources: Some((image, view)),
            image_index,
            semaphore_index,
            status: if suboptimal {
                SurfaceStatus::Suboptimal
            } else {
                SurfaceStatus::Optimal
            },
            submitted: AtomicBool::new(false),
        })
    }

    unsafe fn discard(&mut self, frame: VulkanSurfaceFrame) -> Result<()> {
        if !Arc::ptr_eq(&self.generation, &frame.generation) {
            return Err(Ir::ForeignInstance
                .with_detail("frame belongs to another swapchain generation")
                .into());
        }
        if frame.submitted.load(Ordering::Acquire) {
            return Err(Ir::BadState
                .with_detail("a submitted frame must be presented")
                .into());
        }
        // Without swapchain_maintenance1 an acquired image can only be
        // returned by presenting it, so abandoning it recreates the chain.
        let extent = self.extent();
        unsafe { self.generation.context.device.device_wait_idle() }.map_err(vk_error)?;
        self.recreate(extent.width, extent.height)
    }

    unsafe fn resize(&mut self, width: NonZeroU32, height: NonZeroU32) -> Result<()> {
        self.recreate(width.get(), height.get())
    }

    unsafe fn present(&mut self, frame: VulkanSurfaceFrame) -> Result<SurfaceStatus> {
        if !Arc::ptr_eq(&self.generation.context, frame.context()) {
            return Err(Ir::ForeignInstance
                .with_detail("frame belongs to another Vulkan device")
                .into());
        }
        if !frame.submitted.load(Ordering::Acquire) {
            return Err(Ir::BadState
                .with_detail("frame must be submitted exactly once before presentation")
                .into());
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
    resources: Option<(VulkanImage, VulkanImageView)>,
    pub(crate) image_index: u32,
    pub(crate) semaphore_index: usize,
    status: SurfaceStatus,
    submitted: AtomicBool,
}

impl VulkanSurfaceFrame {
    pub(crate) fn image_available(&self) -> vk::Semaphore {
        self.generation.semaphores[self.semaphore_index].0
    }

    pub(crate) fn render_finished(&self) -> vk::Semaphore {
        // Reacquiring this image establishes completion of its prior presentation wait.
        self.generation.semaphores[self.image_index as usize].1
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.generation.context
    }

    pub(crate) fn mark_submitted(&self) -> Result<()> {
        self.submitted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| {
                Ir::BadState
                    .with_detail("surface frame was already submitted")
                    .into()
            })
    }

    pub(crate) fn track_completion(&self, fence: &VulkanFence) {
        let mut acquisitions = self.generation.acquisitions.lock();
        let slot = &mut acquisitions[self.semaphore_index];
        slot.completion = Some(fence.clone());
        slot.acquired = false;
    }

    pub(crate) fn unmark_submitted(&self) {
        self.submitted.store(false, Ordering::Release);
    }
}

unsafe impl NativeSurfaceFrame<VulkanBackend> for VulkanSurfaceFrame {
    fn take_resources(&mut self) -> Result<(VulkanImage, VulkanImageView)> {
        self.resources.take().ok_or_else(|| Ir::BadState.into())
    }

    fn format(&self) -> SurfaceFormat {
        self.generation.format
    }

    fn extent(&self) -> Extent3d {
        self.generation.extent
    }

    fn status(&self) -> SurfaceStatus {
        self.status
    }
}
