//! Surfaces, swapchains and presentation.

use std::sync::Arc;

use ash::vk;

use super::{Check, CommandBuffer, Device, Image, device::Garbage};
use crate::{
    Error, FRAMES_IN_FLIGHT, Format, PresentMode, QueueType, Result, SurfaceTarget, SwapchainDesc,
};

/// Stages that may first access an acquired image: attachment writes and
/// copies. Submissions wait for the acquire there, and
/// [`ImageAccess::Present`](crate::ImageAccess::Present) barriers start there.
pub(super) const PRESENT_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
        | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
);

/// Binary semaphores a submission waits for or signals for presentation.
#[derive(Clone, Copy, Default)]
pub(super) struct Presentation {
    /// Signaled when the image a submission renders to is acquired.
    pub(super) acquired: Option<vk::Semaphore>,
    /// Signaled by the submission, for presentation to wait for.
    pub(super) rendered: Option<vk::Semaphore>,
}

/// A Vulkan surface and the window it presents to.
pub(crate) struct Surface {
    device: Arc<Device>,
    raw: vk::SurfaceKHR,
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
        // SAFETY: ash-window lists pointers to static C strings.
        if !required
            .iter()
            .all(|&name| enabled.contains(&unsafe { std::ffi::CStr::from_ptr(name) }))
        {
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

    /// Surface functions, which exist since the surface does.
    fn functions(&self) -> &ash::khr::surface::Instance {
        &self
            .device
            .instance
            .surface
            .as_ref()
            .expect("surfaces exist only with presentation")
            .0
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
    format: vk::SurfaceFormatKHR,
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
        let usage = desc.usage;
        let functions = surface.functions();
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
            format,
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
        let functions = self.surface.functions();
        // SAFETY: the surface belongs to this device's instance and adapter.
        let capabilities = unsafe {
            functions
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
        let swapchains = self.functions();
        // SAFETY: `info` is valid for the surface, and the old swapchain is
        // only retired, not destroyed, until its work completes.
        let created = unsafe { swapchains.create_swapchain(&info, None) };
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
        let raw_images = unsafe { self.functions().get_swapchain_images(generation.raw) }
            .check("vkGetSwapchainImagesKHR")?;
        let format =
            Format::from_vk(self.format.format).expect("swapchains use formats the RHI represents");
        raw_images
            .into_iter()
            .map(|raw| {
                generation.rendered.push(semaphore()?);
                Image::swapchain(&self.device, raw, format, label)
            })
            .collect()
    }

    /// Acquires an image with frame slot `slot`'s semaphore, blocking until
    /// one is available. Returns its index and whether the swapchain should
    /// be recreated, or `None` if it is out of date.
    pub(crate) fn acquire(&mut self, slot: usize) -> Result<Option<(u32, bool)>> {
        let Some(generation) = &self.generation else {
            return Ok(None);
        };
        // SAFETY: `&mut self` synchronizes the swapchain; the slot's
        // semaphore is unsignaled since its frame's work waited for it.
        let acquired = unsafe {
            self.functions().acquire_next_image(
                generation.raw,
                u64::MAX,
                generation.acquired[slot],
                vk::Fence::null(),
            )
        };
        match acquired {
            Ok(acquired) => Ok(Some(acquired)),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
            Err(error) => Err(error).check("vkAcquireNextImageKHR"),
        }
    }

    /// Submits `commands`, rendering to image `index` acquired with frame
    /// slot `slot`, to the graphics queue after the GPU reaches `waits`.
    pub(crate) fn submit(
        &self,
        commands: CommandBuffer,
        waits: [u64; QueueType::COUNT],
        slot: usize,
        index: u32,
    ) -> Result<u64> {
        let generation = self.current();
        self.device.submit_batch(
            QueueType::Graphics,
            Some(commands),
            waits,
            Presentation {
                acquired: Some(generation.acquired[slot]),
                rendered: Some(generation.rendered[index as usize]),
            },
        )
    }

    /// Consumes the acquisition of frame slot `slot` without rendering, so
    /// its semaphore can be destroyed. The image stays acquired until the
    /// swapchain is recreated.
    pub(crate) fn release(&self, slot: usize) -> Result<()> {
        let presentation = Presentation {
            acquired: Some(self.current().acquired[slot]),
            rendered: None,
        };
        self.device
            .submit_batch(
                QueueType::Graphics,
                None,
                [0; QueueType::COUNT],
                presentation,
            )
            .map(drop)
    }

    /// Presents image `index` once its rendering completes. Returns whether
    /// the swapchain should be recreated.
    pub(crate) fn present(&mut self, index: u32) -> Result<bool> {
        let generation = self.current();
        let waits = [generation.rendered[index as usize]];
        let swapchains = [generation.raw];
        let indices = [index];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let queue = self.device.lock_queue(QueueType::Graphics);
        // SAFETY: the queue is locked, the image was acquired and its
        // rendering submitted, signaling `waits`.
        match unsafe { self.functions().queue_present(*queue, &info) } {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Err(error) => Err(error).check("vkQueuePresentKHR"),
        }
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

    /// The configured generation; images are only acquired from one.
    fn current(&self) -> &Generation {
        self.generation
            .as_ref()
            .expect("images are acquired from a configured swapchain")
    }

    /// Swapchain functions, which exist since the surface does.
    fn functions(&self) -> &ash::khr::swapchain::Device {
        self.device
            .swapchains
            .as_ref()
            .expect("swapchains exist only with presentation")
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
mod tests {
    use ash::vk;

    use super::Swapchain;

    #[test]
    fn extents_follow_the_surface_or_clamp_to_it() {
        let capabilities = |current| vk::SurfaceCapabilitiesKHR {
            current_extent: current,
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        };
        let extent = |width, height| vk::Extent2D { width, height };
        // Fixed surfaces dictate the extent.
        let fixed = capabilities(extent(800, 600));
        assert_eq!(Swapchain::extent(&fixed, (1, 1)), Some(extent(800, 600)));
        // Others take the wanted one, clamped.
        let free = capabilities(extent(u32::MAX, u32::MAX));
        assert_eq!(
            Swapchain::extent(&free, (640, 9000)),
            Some(extent(640, 4096))
        );
        // Minimized windows have no extent to present at.
        let minimized = capabilities(extent(0, 0));
        assert_eq!(Swapchain::extent(&minimized, (640, 480)), None);
    }
}
