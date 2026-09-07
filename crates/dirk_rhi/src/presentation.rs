use std::{fmt, num::NonZeroU32, sync::Arc};

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};

use crate::{Api, Extent3d, ImageUsages, PresentMode, Result, SurfaceFormat, SurfaceStatus};

/// Owned provider of the native handles needed by a presentation surface.
///
/// The blanket implementation lets applications pass an `Arc` to their
/// platform window without exposing that platform's concrete type through the
/// RHI.
pub trait SurfaceTarget: HasDisplayHandle + HasWindowHandle + Send + Sync + 'static {}

impl<T> SurfaceTarget for T where T: HasDisplayHandle + HasWindowHandle + Send + Sync + 'static {}

/// Owned platform target used to create a graphics presentation surface.
///
/// Backends must retain [`Self::target`] for at least as long as the returned
/// surface so the borrowed raw handles cannot outlive their owner.
#[derive(Clone)]
pub struct SurfaceCreateInfo {
    target: Arc<dyn SurfaceTarget>,
}

impl SurfaceCreateInfo {
    /// Creates surface information that owns a shared reference to its window.
    #[must_use]
    pub fn new<T: SurfaceTarget>(target: Arc<T>) -> Self {
        Self { target }
    }

    /// Returns the platform display handle.
    ///
    /// # Errors
    ///
    /// Returns the platform provider's error when its display is unavailable.
    pub fn display_handle(&self) -> std::result::Result<DisplayHandle<'_>, HandleError> {
        self.target.display_handle()
    }

    /// Returns the platform window handle.
    ///
    /// # Errors
    ///
    /// Returns the platform provider's error when its window is unavailable.
    pub fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, HandleError> {
        self.target.window_handle()
    }

    /// Returns the owned target that a backend surface must retain.
    #[must_use]
    pub fn target(&self) -> &Arc<dyn SurfaceTarget> {
        &self.target
    }
}

impl fmt::Debug for SurfaceCreateInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SurfaceCreateInfo")
            .finish_non_exhaustive()
    }
}

/// Presentation swapchain description.
pub struct SwapchainDesc<'a, B: Api> {
    /// Debug label.
    pub label: &'a str,
    /// Presentation surface.
    pub surface: &'a B::Surface,
    /// Requested image dimensions.
    pub width: NonZeroU32,
    /// Requested image dimensions.
    pub height: NonZeroU32,
    /// Required swapchain image uses.
    pub usage: ImageUsages,
    /// Preferred image formats and color spaces, most preferred first.
    ///
    /// The backend selects the first supported entry, or its own default when
    /// no entry is supported.
    pub preferred_formats: &'a [SurfaceFormat],
    /// Preferred number of images. The backend may select a different count
    /// when constrained by the surface.
    pub desired_image_count: Option<NonZeroU32>,
    /// Preferred presentation mode. The backend may fall back when it is not
    /// supported by the surface.
    pub present_mode: PresentMode,
}

/// Reconfigurable presentation chain and its acquired frames.
///
/// Frames follow a strict lifecycle: [`Self::acquire`] yields a frame, a
/// submission lists that frame in [`Submission::surface_frames`](crate::Submission::surface_frames) to order
/// rendering against presentation, and the frame ends when
/// [`Self::present`](Swapchain::present) returns it to the display or
/// [`Self::discard`](Swapchain::discard) safely abandons it. Implementations
/// must reject duplicate submission and presentation without an intervening
/// submission.
///
/// # Safety
/// Implementations must preserve native object lifetimes and obey the shared
/// [`crate::Backend`] contract. Native operations are called with validated inputs.
pub unsafe trait Swapchain<B: Api> {
    /// Format selected for images in the current swapchain generation.
    fn format(&self) -> SurfaceFormat;
    /// Dimensions selected for images in the current swapchain generation.
    fn extent(&self) -> Extent3d;
    /// Number of images in the current swapchain generation.
    fn image_count(&self) -> NonZeroU32;
    /// Acquires the next presentation frame.
    ///
    /// This call may block the calling thread until an image is available,
    /// including when the presentation queue is full.
    ///
    /// The caller owns frame pacing: hold only a fixed in-flight budget of
    /// acquired frames (tracked with fences, typically two or three), and
    /// present each frame promptly after rendering to it. Holding every
    /// frame deadlocks future acquisitions; treat blocking in this call as a
    /// queue-drain backstop rather than as the renderer's frame limiter.
    ///
    /// # Errors
    ///
    /// Returns an error when the surface is unavailable or must be recreated.
    /// Returns [`crate::Error::Timeout`] when `timeout_ns` expires before an
    /// image becomes available. `u64::MAX` requests an indefinite wait.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn acquire(&mut self, timeout_ns: u64) -> Result<B::SurfaceFrame>;
    /// Releases a frame acquired from this swapchain without presenting it.
    ///
    /// A backend without native acquired-image release may recreate or
    /// invalidate the swapchain rather than presenting stale contents. The
    /// caller must no longer submit or present the frame after this returns.
    ///
    /// # Errors
    ///
    /// Returns an error when the frame is foreign, already submitted, or
    /// cannot be safely abandoned.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn discard(&mut self, frame: B::SurfaceFrame) -> Result<()>;
    /// Reconfigures this swapchain for a new extent.
    ///
    /// All frames previously acquired from this swapchain must have been
    /// presented with [`Self::present`](Swapchain::present) or released with
    /// [`Self::discard`](Swapchain::discard) before resizing.
    ///
    /// # Errors
    ///
    /// Returns an error when recreation fails.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn resize(&mut self, width: NonZeroU32, height: NonZeroU32) -> Result<()>;
    /// Presents a frame previously acquired from this swapchain.
    ///
    /// The frame's rendering submission must have listed the frame in
    /// [`Submission::surface_frames`](crate::Submission::surface_frames).
    ///
    /// Returns [`SurfaceStatus::Suboptimal`] when the frame was presented but
    /// the swapchain should be recreated.
    ///
    /// # Errors
    ///
    /// Returns [`crate::InvalidResourceKind::BadState`] when the frame was not
    /// submitted exactly once, or another error when presentation fails.
    ///
    /// # Safety
    /// The caller must uphold the native [`crate::Backend`] contract for this
    /// operation, including resource lifetime, valid state, and host synchronization.
    unsafe fn present(&mut self, frame: B::SurfaceFrame) -> Result<SurfaceStatus>;
}

/// A presentation image acquired from a backend swapchain.
///
/// Implementations must safely abandon an unconsumed acquisition when a frame
/// is dropped. This may mark its swapchain for recreation. Image and view
/// handles exposed by the frame must retain an immutable acquisition token;
/// recording or submitting them after the frame lifecycle ends must return
/// [`crate::InvalidResourceKind::BadState`].
#[must_use = "an acquired surface frame must be submitted and presented or explicitly discarded"]
///
/// # Safety
/// Implementations must preserve native object lifetimes and obey the shared
/// [`crate::Backend`] contract. Native operations are called with validated inputs.
pub unsafe trait SurfaceFrame<B: Api> {
    /// Image backing this frame.
    fn image(&self) -> &B::Image;
    /// Default view of [`Self::image`].
    fn view(&self) -> &B::ImageView;
    /// Image format.
    fn format(&self) -> SurfaceFormat;
    /// Image dimensions.
    fn extent(&self) -> Extent3d;
    /// Whether the frame is usable but should trigger recreation.
    fn status(&self) -> SurfaceStatus;
}
