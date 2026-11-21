//! Surfaces, swapchains and presentation.

use std::sync::Arc;

use ash::vk;

use super::{Check, Device, Image, device::Garbage};
use crate::{
    Error, FRAMES_IN_FLIGHT, Format, PresentMode, QueueType, Result, SurfaceTarget, SwapchainDesc,
    presentation::Acquired,
};

/// Stages that may first access an acquired image: attachment writes and
/// copies. Submissions wait for the acquire there, and
/// [`ImageAccess::Present`](crate::ImageAccess::Present) barriers start there.
pub(super) const PRESENT_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
        | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
);

/// A binary semaphore ordering presentation and the work around it.
pub(crate) type Semaphore = vk::Semaphore;

/// Binary semaphores a submission waits for or signals for presentation.
#[derive(Clone, Copy, Default)]
pub(crate) struct Presentation<'a> {
    /// Signaled when images the submission may render to are acquired.
    pub(crate) acquired: &'a [Semaphore],
    /// Signaled by the submission, for presentation to wait for.
    pub(crate) rendered: &'a [Semaphore],
}

/// A Vulkan surface and the window it presents to.
pub(crate) struct Surface {
    device: Arc<Device>,
    raw: vk::SurfaceKHR,
    /// The instance's surface functions.
    functions: ash::khr::surface::Instance,
    /// Kept alive until the surface is destroyed.
    target: Arc<dyn SurfaceTarget>,
}

impl Surface {
    /// Creates a surface for `target`'s window, which the graphics queue
    /// must be able to present to.
    pub(crate) fn new(device: &Arc<Device>, target: &Arc<dyn SurfaceTarget>) -> Result<Self> {
        let invalid = |reason: &str| Error::invalid("surface", "", reason);
        let (surfaces, enabled) = device.instance.surface.as_ref().ok_or_else(|| {
            invalid("presentation is disabled; create the RHI with `RhiCreateInfo::display`")
        })?;
        let display = target
            .display_handle()
            .map_err(|error| Error::native("getting the window's display", error))?;
        let window = target
            .window_handle()
            .map_err(|error| Error::native("getting the window", error))?;
        let required = ash_window::enumerate_required_extensions(display.as_raw())
            .check("listing the display's surface extensions")?;
        let displayable = required.iter().all(|&name| {
            // SAFETY: ash-window lists pointers to static C strings.
            enabled.contains(&unsafe { std::ffi::CStr::from_ptr(name) })
        });
        if !displayable {
            return Err(invalid(
                "the window is on another kind of display than the RHI was created for",
            ));
        }
        let instance = &device.instance;
        // SAFETY: the handles are valid while `target` lives, which the
        // surface keeps alive, and their display's extensions are enabled.
        let raw = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.raw,
                display.as_raw(),
                window.as_raw(),
                None,
            )
        }
        .check("creating a surface")?;
        // Owned from here on, so a failure below destroys it.
        let surface = Self {
            device: device.clone(),
            raw,
            functions: surfaces.clone(),
            target: target.clone(),
        };
        // SAFETY: the surface and the queue family belong to this device's
        // instance and adapter.
        let supported = unsafe {
            surfaces.get_physical_device_surface_support(
                device.physical,
                device.family(QueueType::Graphics),
                raw,
            )
        }
        .check("vkGetPhysicalDeviceSurfaceSupportKHR")?;
        if !supported {
            return Err(invalid("the graphics queue cannot present to this window"));
        }
        Ok(surface)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.device
            .bury(Garbage::Surface(self.raw, self.target.clone()));
    }
}

/// One configuration of a swapchain: its images' semaphores. Replaced
/// whenever the swapchain is recreated, and retired once the GPU is done.
struct Generation {
    raw: vk::SwapchainKHR,
    /// Signaled by acquisitions, one per frame slot: beginning a frame
    /// waits for the work that waited on its slot's semaphore.
    acquired: [vk::Semaphore; FRAMES_IN_FLIGHT],
    /// Signaled by the submission rendering to each image and waited for by
    /// its presentation. Reused when the image is acquired again, which
    /// implies that presentation finished with it.
    rendered: Vec<vk::Semaphore>,
}

/// A Vulkan swapchain for a surface.
pub(crate) struct Swapchain {
    pub(super) device: Arc<Device>,
    /// `None` until configured, and after a failed configuration.
    generation: Option<Generation>,
    /// Destroyed after the generations presenting to it.
    surface: Surface,
    /// The device's swapchain functions.
    functions: ash::khr::swapchain::Device,
    format: vk::SurfaceFormatKHR,
    /// `format`, as the RHI names it.
    rhi_format: Format,
    present_mode: vk::PresentModeKHR,
    usage: vk::ImageUsageFlags,
}

impl Swapchain {
    /// Prepares a swapchain for `surface`, choosing the first of the
    /// description's formats the surface supports, or any representable one,
    /// and its present mode, or FIFO if unsupported. Images exist once
    /// configured.
    pub(crate) fn new(
        device: &Arc<Device>,
        surface: Surface,
        desc: &SwapchainDesc<'_>,
    ) -> Result<(Self, Format)> {
        let invalid = |reason: String| Error::invalid("swapchain", desc.label, reason);
        if !Arc::ptr_eq(&surface.device, device) {
            return Err(invalid("the surface belongs to another device".to_owned()));
        }
        let swapchains = device.swapchains.clone().ok_or_else(|| {
            invalid(
                "presentation is disabled; create the RHI with `RhiCreateInfo::display`".to_owned(),
            )
        })?;
        let usage = desc.usage;
        let functions = &surface.functions;
        // SAFETY: the surface belongs to this device's instance and adapter.
        let (capabilities, supported_formats, modes) = unsafe {
            (
                functions
                    .get_physical_device_surface_capabilities(device.physical, surface.raw)
                    .check("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?,
                functions
                    .get_physical_device_surface_formats(device.physical, surface.raw)
                    .check("vkGetPhysicalDeviceSurfaceFormatsKHR")?,
                functions
                    .get_physical_device_surface_present_modes(device.physical, surface.raw)
                    .check("vkGetPhysicalDeviceSurfacePresentModesKHR")?,
            )
        };
        if !capabilities.supported_usage_flags.contains(usage.vk()) {
            return Err(invalid(format!(
                "the surface does not support {usage:?} images"
            )));
        }
        let srgb = |format: &&vk::SurfaceFormatKHR| {
            format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        };
        let (format, chosen) = desc
            .formats
            .iter()
            .find_map(|&wanted| {
                supported_formats
                    .iter()
                    .filter(srgb)
                    .find(|format| format.format == wanted.vk())
                    .map(|format| (*format, wanted))
            })
            .or_else(|| {
                supported_formats
                    .iter()
                    .filter(srgb)
                    .find_map(|format| Some((*format, Format::from_vk(format.format)?)))
            })
            .ok_or_else(|| {
                invalid(format!(
                    "the surface supports none of the RHI's formats: {supported_formats:?}"
                ))
            })?;
        let present_mode = Some(desc.present_mode.vk())
            .filter(|mode| modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let swapchain = Self {
            device: device.clone(),
            generation: None,
            surface,
            functions: swapchains,
            format,
            rhi_format: chosen,
            present_mode,
            usage: usage.vk(),
        };
        Ok((swapchain, chosen))
    }

    /// Recreates the swapchain at `extent`, or the surface's own extent,
    /// retiring the previous generation once the GPU is done with it.
    /// Returns its images, labelled `label`, and their extent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SwapchainOutOfDate`] while the surface has no area,
    /// such as a minimized window.
    pub(crate) fn configure(
        &mut self,
        label: &str,
        extent: (u32, u32),
    ) -> Result<(Vec<Image>, (u32, u32))> {
        // SAFETY: the surface belongs to this device's instance and adapter.
        let capabilities = unsafe {
            self.surface
                .functions
                .get_physical_device_surface_capabilities(self.device.physical, self.surface.raw)
        }
        .check("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?;
        let extent = Self::extent(&capabilities, extent).ok_or(Error::SwapchainOutOfDate)?;
        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|mode| capabilities.supported_composite_alpha.contains(*mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::INHERIT);
        let old = self.generation.take();
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface.raw)
            .min_image_count(image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(self.usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(
                old.as_ref()
                    .map_or_else(vk::SwapchainKHR::null, |old| old.raw),
            );
        // SAFETY: `info` is valid for the surface, and the old swapchain is
        // only retired, not destroyed, until its work completes.
        let created = unsafe { self.functions.create_swapchain(&info, None) };
        // Creating retires the old swapchain even if it fails.
        if let Some(old) = old {
            self.device.bury(old.into_garbage());
        }
        let mut generation = Generation {
            raw: created.check("vkCreateSwapchainKHR")?,
            acquired: [vk::Semaphore::null(); FRAMES_IN_FLIGHT],
            rendered: Vec::new(),
        };
        match self.fill(&mut generation, label) {
            Ok(images) => {
                self.generation = Some(generation);
                Ok((images, (extent.width, extent.height)))
            }
            Err(error) => {
                self.device.bury(generation.into_garbage());
                Err(error)
            }
        }
    }

    /// Creates `generation`'s semaphores and wraps its images.
    fn fill(&self, generation: &mut Generation, label: &str) -> Result<Vec<Image>> {
        let semaphore = || {
            // SAFETY: the default semaphore info is valid.
            unsafe {
                self.device
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .check("vkCreateSemaphore")
        };
        for acquired in &mut generation.acquired {
            *acquired = semaphore()?;
        }
        // SAFETY: the swapchain was created from this device.
        let raw_images = unsafe { self.functions.get_swapchain_images(generation.raw) }
            .check("vkGetSwapchainImagesKHR")?;
        raw_images
            .into_iter()
            .map(|raw| {
                generation.rendered.push(semaphore()?);
                Image::swapchain(&self.device, raw, self.rhi_format, label)
            })
            .collect()
    }

    /// Acquires an image with frame slot `slot`'s semaphore, blocking until
    /// one is available, or returns `None` if the swapchain is out of date.
    pub(crate) fn acquire(&mut self, slot: usize) -> Result<Option<Acquired>> {
        let Some(generation) = &self.generation else {
            return Ok(None);
        };
        // SAFETY: `&mut self` synchronizes the swapchain; the slot's
        // semaphore is unsignaled since its frame's work waited for it.
        let acquired = unsafe {
            self.functions.acquire_next_image(
                generation.raw,
                u64::MAX,
                generation.acquired[slot],
                vk::Fence::null(),
            )
        };
        match acquired {
            Ok((index, suboptimal)) => Ok(Some(Acquired {
                index,
                suboptimal,
                image: None,
            })),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(error) => Err(error).check("vkAcquireNextImageKHR"),
        }
    }

    /// The semaphore the acquisition with frame slot `slot` signals, which
    /// work using the image waits for; `None` while the swapchain is not
    /// configured, when nothing was acquired.
    pub(crate) fn acquired(&self, slot: usize) -> Option<Semaphore> {
        Some(self.generation.as_ref()?.acquired[slot])
    }

    /// The semaphore presenting image `index` waits for, which the
    /// submission before it signals; `None` while the swapchain is not
    /// configured.
    pub(crate) fn rendered(&self, index: u32) -> Option<Semaphore> {
        Some(self.generation.as_ref()?.rendered[index as usize])
    }

    /// The extent to create images with: the surface's, or `wanted` clamped
    /// to its bounds when it lets the swapchain decide. `None` while the
    /// surface has no area.
    fn extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        wanted: (u32, u32),
    ) -> Option<vk::Extent2D> {
        let extent = if capabilities.current_extent.width == u32::MAX {
            let (min, max) = (capabilities.min_image_extent, capabilities.max_image_extent);
            vk::Extent2D {
                width: wanted.0.clamp(min.width, max.width),
                height: wanted.1.clamp(min.height, max.height),
            }
        } else {
            capabilities.current_extent
        };
        (extent.width > 0 && extent.height > 0).then_some(extent)
    }
}

impl Device {
    /// Presents image `index` of each swapchain once the graphics work
    /// submitted before completes, which signaled their rendered semaphores.
    /// Returns, per swapchain, whether it should be recreated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a swapchain was reconfigured without
    /// success since its image was acquired.
    pub(crate) fn present(&self, images: &mut [(&mut Swapchain, u32)]) -> Result<Vec<bool>> {
        let Some((first, _)) = images.first() else {
            return Ok(Vec::new());
        };
        let functions = &first.functions;
        let (waits, swapchains): (Vec<_>, Vec<_>) = images
            .iter()
            .map(|(swapchain, index)| {
                let generation = swapchain.generation.as_ref().ok_or_else(|| {
                    Error::invalid("presentation", "", "a swapchain lost its configuration")
                })?;
                Ok((generation.rendered[*index as usize], generation.raw))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        let indices: Vec<_> = images.iter().map(|&(_, index)| index).collect();
        let mut results = vec![vk::Result::SUCCESS; images.len()];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices)
            .results(&mut results);
        let queue = self.lock_queue(QueueType::Graphics);
        // SAFETY: the queue is locked, the images were acquired and the
        // submission before signaled `waits`.
        match unsafe { functions.queue_present(*queue, &info) } {
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
            Err(error) => return Err(error).check("vkQueuePresentKHR"),
        }
        results
            .into_iter()
            .map(|result| match result {
                vk::Result::SUCCESS => Ok(false),
                vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => Ok(true),
                error => Err(error).check("vkQueuePresentKHR"),
            })
            .collect()
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        if let Some(generation) = self.generation.take() {
            self.device.bury(generation.into_garbage());
        }
    }
}

impl Generation {
    /// The swapchain and its semaphores, for destruction once the GPU is
    /// done; null semaphores of a partial generation are ignored.
    fn into_garbage(self) -> Garbage {
        Garbage::Swapchain(
            self.raw,
            self.acquired.into_iter().chain(self.rendered).collect(),
        )
    }
}

impl PresentMode {
    /// The equivalent Vulkan present mode.
    const fn vk(self) -> vk::PresentModeKHR {
        match self {
            Self::Fifo => vk::PresentModeKHR::FIFO,
            Self::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
            Self::Mailbox => vk::PresentModeKHR::MAILBOX,
            Self::Immediate => vk::PresentModeKHR::IMMEDIATE,
        }
    }
}

#[cfg(test)]
mod tests;
