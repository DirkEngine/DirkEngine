use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Error, Extent3d, Format, Frame, Image, ImageAccess, ImageDimension, ImageInfo, ImageRange,
    ImageUsage, QueueType, Result, Rhi, SampleCount, ViewType, native,
};

/// A window surfaces present to: the platform's shared owner of a native
/// window, such as `dirk_platform::WindowSurfaceTarget`.
pub trait SurfaceTarget: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static {}

impl<T: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static> SurfaceTarget for T {}

/// Somewhere to present images: a window. Consumed by the swapchain
/// presenting to it.
///
/// The surface keeps its window alive until it is destroyed, once the GPU
/// is done with it, which happens when a later frame begins or the RHI
/// waits for the GPU or drops. Dropping the window's last owner can close
/// it, which on some platforms must happen on the main thread: if the
/// surface may hold the last clone, keep another one, or make sure frames
/// begin on the main thread.
pub struct Surface {
    raw: native::Surface,
}

/// How presented images are paced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PresentMode {
    /// Wait for vertical blank, queueing images. Always supported, and the
    /// fallback for the others.
    #[default]
    Fifo,
    /// Like `Fifo`, but present late images immediately, which may tear.
    FifoRelaxed,
    /// Wait for vertical blank, replacing the queued image with newer ones.
    Mailbox,
    /// Present immediately, which may tear.
    Immediate,
}

/// Describes a swapchain to create.
#[derive(Clone, Copy, Debug)]
pub struct SwapchainDesc<'a> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// Width and height of images, for surfaces that let the swapchain
    /// choose, such as Wayland windows. Others use the window's size.
    pub extent: (u32, u32),
    /// Image formats, most preferred first. Without a supported one, any
    /// format the surface supports in the sRGB color space is used.
    pub formats: &'a [Format],
    /// How presentation is paced.
    pub present_mode: PresentMode,
    /// How images are used: [`ImageUsage::COLOR_ATTACHMENT`],
    /// [`ImageUsage::COPY_SRC`] and [`ImageUsage::COPY_DST`] are allowed.
    pub usage: ImageUsage,
}

/// Images presented to a surface, recreated when the surface changes.
///
/// Acquire an image in a frame with [`Frame::acquire`], render to it with
/// any graphics work of the frame, and present it with [`Frame::present`],
/// which presents the images of any number of swapchains at once. The
/// frame's graphics submissions after an acquisition wait for it, and
/// presenting waits for every graphics submission before it.
///
/// Swapchain images rest in [`ImageAccess::Present`], like other images in
/// their resting access: render passes and copies move them from it and
/// back, and each acquisition hands one back with undefined contents.
///
/// Presenting consumes the acquired image, so it cannot be presented twice:
/// ```compile_fail
/// use dirk_rhi::{Frame, Result, Swapchain};
/// fn present(frame: &Frame<'_>, swapchain: &mut Swapchain) -> Result<()> {
///     let image = frame.acquire(swapchain)?;
///     frame.present([image])?;
///     frame.present([image])
/// }
/// ```
///
/// An acquired image borrows its swapchain and frame, so the swapchain
/// cannot be resized meanwhile:
/// ```compile_fail
/// use dirk_rhi::{Frame, Result, Swapchain};
/// fn resize(frame: &Frame<'_>, swapchain: &mut Swapchain) -> Result<()> {
///     let image = frame.acquire(swapchain)?;
///     swapchain.resize(1, 1);
///     frame.present([image])
/// }
/// ```
///
/// Nor can the image outlive the frame:
/// ```compile_fail
/// use dirk_rhi::{Result, Rhi, Swapchain};
/// fn escape(rhi: &mut Rhi, swapchain: &mut Swapchain) -> Result<()> {
///     let frame = rhi.begin_frame()?;
///     let image = frame.acquire(swapchain)?;
///     frame.finish();
///     let presented = image.image();
///     Ok(())
/// }
/// ```
///
/// Images abandoned before presentation are released by recreating the
/// swapchain on the next acquisition.
pub struct Swapchain {
    /// Dropped before the native swapchain retires their generation. On
    /// Metal, the one image, of the drawable acquired last.
    images: Vec<Image>,
    /// What the images are.
    info: ImageInfo,
    raw: native::Swapchain,
    label: Box<str>,
    /// The extent to recreate images with.
    extent: (u32, u32),
    /// Whether the swapchain must be recreated before acquiring.
    stale: bool,
    /// The frame that acquired an image last.
    acquired_in: u64,
    /// Whether an image was acquired and not presented.
    unpresented: bool,
}

/// An image a native swapchain acquired.
pub(crate) struct Acquired {
    pub(crate) index: u32,
    /// Whether the swapchain no longer matches its surface exactly.
    pub(crate) suboptimal: bool,
    /// The image at `index`, when the native swapchain brings one with each
    /// acquisition, as Metal's drawables do. It replaces the images.
    pub(crate) image: Option<native::Image>,
}

/// An image acquired from a swapchain, to render to and present with
/// [`Frame::present`].
#[must_use = "acquired images are presented"]
pub struct AcquiredImage<'a> {
    /// The frame that acquired it, the only one presenting it.
    frame: &'a Frame<'a>,
    swapchain: &'a mut Swapchain,
    index: u32,
    suboptimal: bool,
}

impl Rhi {
    /// Creates a surface presenting to `target`'s window, keeping `target`
    /// alive until the surface is destroyed; see [`Surface`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the RHI was created without a display,
    /// or for another kind of display, or the device cannot present to the
    /// window, or a native error if its handles are unavailable or the
    /// surface cannot be created.
    pub fn create_surface<T: SurfaceTarget>(&self, target: Arc<T>) -> Result<Surface> {
        Ok(Surface {
            raw: native::Surface::new(&self.device, &(target as Arc<dyn SurfaceTarget>))?,
        })
    }

    /// Creates a swapchain presenting to `surface`. Its images are created
    /// when the first is acquired.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the surface belongs to another device,
    /// the usage is empty or not allowed, or the surface supports neither
    /// the usage nor any format the RHI represents, or a native error.
    pub fn create_swapchain(
        &self,
        surface: Surface,
        desc: &SwapchainDesc<'_>,
    ) -> Result<Swapchain> {
        let allowed = ImageUsage::COLOR_ATTACHMENT | ImageUsage::COPY_SRC | ImageUsage::COPY_DST;
        if desc.usage.is_empty() || !allowed.contains(desc.usage) {
            return Err(Error::invalid(
                "swapchain",
                desc.label,
                format!("usage {:?} is not within {allowed:?}", desc.usage),
            ));
        }
        let (raw, format) = native::Swapchain::new(&self.device, surface.raw, desc)?;
        Ok(Swapchain {
            images: Vec::new(),
            info: ImageInfo {
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(desc.extent.0, desc.extent.1),
                format,
                usage: desc.usage,
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::X1,
                rest: ImageAccess::Present,
            },
            raw,
            label: desc.label.into(),
            extent: desc.extent,
            stale: true,
            acquired_in: 0,
            unpresented: false,
        })
    }
}

impl Swapchain {
    /// Format of the images.
    #[must_use]
    pub fn format(&self) -> Format {
        self.info.format
    }

    /// Recreates the images at `width` by `height` before the next
    /// acquisition, for surfaces that let the swapchain choose its extent.
    /// Others follow their window's size by themselves.
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.extent != (width, height) {
            self.extent = (width, height);
            self.stale = true;
        }
    }

    /// Recreates the images, dropping the old ones before their swapchain.
    fn recreate(&mut self) -> Result<()> {
        self.images.clear();
        let (images, (width, height)) = self.raw.configure(&self.label, self.extent)?;
        self.info.extent = Extent3d::new_2d(width, height);
        self.images = images.into_iter().map(|raw| self.wrap(raw)).collect();
        self.stale = false;
        Ok(())
    }

    /// Wraps one of the native swapchain's images.
    fn wrap(&self, raw: native::Image) -> Image {
        Image::from_native(raw, self.info, ViewType::D2, &self.label)
    }
}

impl Frame<'_> {
    /// Acquires the next image of `swapchain`, recreating it first if it is
    /// out of date, was resized or still holds an image never presented.
    /// Blocks until an image is available. The frame's next graphics
    /// submission waits for the acquisition.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SwapchainOutOfDate`] while the surface cannot be
    /// presented to, such as a minimized window: skip presenting and try
    /// again next frame. Returns [`Error::Invalid`] if the swapchain belongs
    /// to another device or already acquired an image this frame, or a
    /// native error.
    pub fn acquire<'a>(&'a self, swapchain: &'a mut Swapchain) -> Result<AcquiredImage<'a>> {
        let invalid = |reason: &str| Err(Error::invalid("swapchain", &swapchain.label, reason));
        if !swapchain.raw.is_from(&self.rhi().device) {
            return invalid("belongs to another device");
        }
        if swapchain.acquired_in == self.number() {
            return invalid("one image is acquired per swapchain and frame");
        }
        // Recreating releases an image acquired and never presented.
        swapchain.stale |= std::mem::take(&mut swapchain.unpresented);
        let slot = self.slot();
        let mut retried = false;
        let Acquired {
            index,
            suboptimal,
            image,
        } = loop {
            if swapchain.stale {
                swapchain.recreate()?;
            }
            match swapchain.raw.acquire(slot)? {
                Some(acquired) => break acquired,
                // Out of date even when just recreated: try again later.
                None if retried => return Err(Error::SwapchainOutOfDate),
                None => (swapchain.stale, retried) = (true, true),
            }
        };
        if let Some(raw) = image {
            let image = swapchain.wrap(raw);
            swapchain.images.clear();
            swapchain.images.push(image);
        }
        self.wait_for_acquisition(swapchain.raw.acquired(slot));
        // Its contents are undefined until the frame's next graphics
        // submission moves it to rest, after waiting for the acquisition.
        let image = &swapchain.images[index as usize];
        self.rhi().device.initialize(image.transition(
            ImageRange::ALL,
            ImageAccess::Undefined,
            ImageAccess::Present,
        ));
        swapchain.acquired_in = self.number();
        swapchain.unpresented = true;
        // Still presentable; recreated on the next acquisition.
        swapchain.stale |= suboptimal;
        Ok(AcquiredImage {
            frame: self,
            swapchain,
            index,
            suboptimal,
        })
    }

    /// Presents `images`, of different swapchains, once the graphics work
    /// submitted before completes: one submission orders them after it,
    /// then one presentation presents them all. Swapchains that no longer
    /// match their surface are recreated on their next acquisition.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`], presenting nothing, if an image was
    /// acquired by another frame, such as one of another RHI, whose
    /// submissions this frame's do not order; or a native error such as
    /// [`Error::DeviceLost`] or a lost surface.
    pub fn present<'a>(&self, images: impl IntoIterator<Item = AcquiredImage<'a>>) -> Result<()> {
        let mut images: Vec<_> = images.into_iter().collect();
        if let Some(image) = images.iter().find(|image| !std::ptr::eq(image.frame, self)) {
            return Err(Error::invalid(
                "presentation",
                &image.swapchain.label,
                format!(
                    "the image was acquired by another frame, not frame {}",
                    self.number()
                ),
            ));
        }
        if images.is_empty() {
            return Ok(());
        }
        let rendered: Vec<_> = images
            .iter()
            .filter_map(|image| image.swapchain.raw.rendered(image.index))
            .collect();
        if !rendered.is_empty() {
            self.submit_commands(QueueType::Graphics, None, [0; QueueType::COUNT], &rendered)?;
        }
        let mut targets: Vec<_> = images
            .iter_mut()
            .map(|image| (&mut image.swapchain.raw, image.index))
            .collect();
        let outdated = self.rhi().device.present(&mut targets)?;
        for (image, outdated) in images.into_iter().zip(outdated) {
            image.swapchain.unpresented = false;
            image.swapchain.stale |= outdated;
        }
        Ok(())
    }
}

impl AcquiredImage<'_> {
    /// The image, to render to or copy into, resting in
    /// [`ImageAccess::Present`] with undefined contents until something
    /// writes it.
    #[must_use]
    pub fn image(&self) -> &Image {
        &self.swapchain.images[self.index as usize]
    }

    /// Whether the swapchain no longer matches its surface exactly. The
    /// image is still presented; the swapchain is recreated next time.
    #[must_use]
    pub fn suboptimal(&self) -> bool {
        self.suboptimal
    }
}
