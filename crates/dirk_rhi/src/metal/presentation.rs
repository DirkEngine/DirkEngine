//! Surfaces, swapchains and presentation, through `CAMetalLayer`.

use std::sync::Arc;

use objc2::{MainThreadMarker, rc::autoreleasepool};
use objc2_core_foundation::CGSize;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::RawWindowHandle;
use raw_window_metal::Layer;

use super::{Device, Image, Object, device::Garbage};
use crate::{
    Error, Format, ImageUsage, PresentMode, Result, SurfaceTarget, SwapchainDesc,
    presentation::Acquired,
};

/// A semaphore ordering presentation, which Metal never needs: none exists.
pub(crate) enum Semaphore {}

/// Semaphores a submission waits for or signals for presentation: always
/// empty on Metal, which presents drawables after the work committed before.
#[derive(Clone, Copy, Default)]
#[expect(
    dead_code,
    reason = "Metal has no semaphores; the fields keep the type shared with Vulkan"
)]
pub(crate) struct Presentation<'a> {
    /// Signaled when images the submission may render to are acquired.
    pub(crate) acquired: &'a [Semaphore],
    /// Signaled by the submission, for presentation to wait for.
    pub(crate) rendered: &'a [Semaphore],
}

/// Formats a `CAMetalLayer` presents that the RHI represents, in the sRGB
/// color space.
const FORMATS: [Format; 2] = [Format::Bgra8Srgb, Format::Bgra8Unorm];

/// A Metal layer drawing into a window, and the window, kept alive until the
/// layer is released.
pub(crate) struct Surface {
    device: Arc<Device>,
    layer: Layer,
    target: Arc<dyn SurfaceTarget>,
}

impl Surface {
    /// Creates a layer drawing into `target`'s view. `AppKit` and `UIKit`
    /// views are only accessed on the main thread.
    pub(crate) fn new(device: &Arc<Device>, target: &Arc<dyn SurfaceTarget>) -> Result<Self> {
        let invalid = |reason: &str| Error::invalid("surface", "", reason);
        if !device.presentation {
            return Err(invalid(
                "presentation is disabled; create the RHI with `RhiCreateInfo::display`",
            ));
        }
        let window = target
            .window_handle()
            .map_err(|error| Error::native("getting the window", error))?;
        if MainThreadMarker::new().is_none() {
            return Err(invalid("surfaces are created on the main thread"));
        }
        // SAFETY: the view is valid while `target` lives, which the surface
        // keeps alive, and this is the main thread.
        let layer = match window.as_raw() {
            RawWindowHandle::AppKit(handle) => unsafe { Layer::from_ns_view(handle.ns_view) },
            RawWindowHandle::UiKit(handle) => unsafe { Layer::from_ui_view(handle.ui_view) },
            _ => {
                return Err(invalid(
                    "the window is on another kind of display than the RHI was created for",
                ));
            }
        };
        Ok(Self {
            device: device.clone(),
            layer,
            target: target.clone(),
        })
    }

    /// The layer.
    fn layer(&self) -> &CAMetalLayer {
        // SAFETY: the pointer is the layer's, which `self` retains.
        unsafe { self.layer.as_ptr().cast().as_ref() }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.device
            .bury(Garbage::object((self.layer.clone(), self.target.clone())));
    }
}

/// A Metal layer's drawables, one acquired at a time.
pub(crate) struct Swapchain {
    pub(super) device: Arc<Device>,
    surface: Surface,
    /// The drawable acquired last, until it is presented or released.
    drawable: Option<Object<dyn CAMetalDrawable>>,
}

impl Swapchain {
    /// Configures `surface`'s layer with the first of the description's
    /// formats it supports, or an sRGB one, and the present mode, or FIFO if
    /// unsupported. Waiting for a drawable never times out, as acquisitions
    /// block until an image is available.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the surface belongs to another device.
    pub(crate) fn new(
        device: &Arc<Device>,
        surface: Surface,
        desc: &SwapchainDesc<'_>,
    ) -> Result<(Self, Format)> {
        if !Arc::ptr_eq(&surface.device, device) {
            return Err(Error::invalid(
                "swapchain",
                desc.label,
                "the surface belongs to another device",
            ));
        }
        let format = desc
            .formats
            .iter()
            .copied()
            .find(|format| FORMATS.contains(format))
            .unwrap_or(FORMATS[0]);
        let layer = surface.layer();
        layer.setDevice(Some(&device.raw));
        layer.setPixelFormat(format.mtl());
        layer.setFramebufferOnly(
            !desc
                .usage
                .intersects(ImageUsage::COPY_SRC | ImageUsage::COPY_DST),
        );
        layer.setAllowsNextDrawableTimeout(false);
        #[cfg(target_os = "macos")]
        layer.setDisplaySyncEnabled(desc.present_mode != PresentMode::Immediate);
        Ok((
            Self {
                device: device.clone(),
                surface,
                drawable: None,
            },
            format,
        ))
    }

    /// Sizes drawables to the layer, in pixels. Returns no images: each
    /// drawable brings its own.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SwapchainOutOfDate`] while the layer has no area,
    /// such as a minimized window.
    pub(crate) fn configure(
        &mut self,
        _label: &str,
        _extent: (u32, u32),
    ) -> Result<(Vec<super::Image>, (u32, u32))> {
        self.drawable = None;
        let layer = self.surface.layer();
        let (width, height) = Self::pixels(layer);
        if width == 0 || height == 0 {
            return Err(Error::SwapchainOutOfDate);
        }
        layer.setDrawableSize(CGSize {
            width: f64::from(width),
            height: f64::from(height),
        });
        Ok((Vec::new(), (width, height)))
    }

    /// The size of `layer` in pixels.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "layer sizes are small, non-negative pixel counts"
    )]
    fn pixels(layer: &CAMetalLayer) -> (u32, u32) {
        let size = layer.bounds().size;
        let scale = layer.contentsScale();
        (
            (size.width * scale).round() as u32,
            (size.height * scale).round() as u32,
        )
    }

    /// Acquires the next drawable, blocking until one is available. It is
    /// suboptimal once the layer's size no longer matches the drawables'.
    /// Returns `None` if the layer has no drawable to give, such as when it
    /// has no area.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "the signature is shared with Vulkan, where this can fail"
    )]
    pub(crate) fn acquire(&mut self, _slot: usize) -> Result<Option<Acquired>> {
        let layer = self.surface.layer();
        let Some(drawable) = autoreleasepool(|_| layer.nextDrawable()) else {
            return Ok(None);
        };
        let drawable = Object::from(drawable);
        let texture = drawable.texture().into();
        let size = layer.drawableSize();
        let (width, height) = Self::pixels(layer);
        let suboptimal = (f64::from(width), f64::from(height)) != (size.width, size.height);
        self.drawable = Some(drawable);
        Ok(Some(Acquired {
            index: 0,
            suboptimal,
            image: Some(Image::drawable(&self.device, texture)),
        }))
    }

    /// None: Metal acquisitions need no semaphore, since acquiring blocks
    /// until the drawable is available.
    #[expect(
        clippy::unused_self,
        reason = "the signature is shared with Vulkan, whose acquisitions signal one"
    )]
    pub(crate) fn acquired(&self, _slot: usize) -> Option<Semaphore> {
        None
    }

    /// None: drawables are presented after the work committed before them.
    #[expect(
        clippy::unused_self,
        reason = "the signature is shared with Vulkan, whose presentation waits for one"
    )]
    pub(crate) fn rendered(&self, _index: u32) -> Option<Semaphore> {
        None
    }
}

impl Device {
    /// Presents the drawable each swapchain acquired once the graphics work
    /// submitted before completes: one graphics command buffer presents them
    /// all, after waiting for it. Size changes are found when acquiring, so
    /// this never asks for recreation.
    pub(crate) fn present(&self, images: &mut [(&mut Swapchain, u32)]) -> Result<Vec<bool>> {
        self.present_drawables(images.iter_mut().map(|(swapchain, _)| {
            swapchain
                .drawable
                .take()
                .expect("images are presented after being acquired")
        }))?;
        Ok(vec![false; images.len()])
    }
}
