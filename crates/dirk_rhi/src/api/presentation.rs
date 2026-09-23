use super::{
    Arc, AtomicUsize, Backend, GpuImage, GpuImageView, Mutex, Object, Ordering, Rhi, ViewMetadata,
};
use crate::{
    Extent3d, ImageDimension, ImageInfo, ImageUsages, InvalidResourceKind as Ir,
    NativeSurfaceFrame as _, NativeSwapchain as _, Result, SampleCount, SurfaceFormat,
    SurfaceStatus, SwapchainDesc,
};
use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicBool, AtomicU8},
};

struct Chain<B: Backend> {
    raw: Mutex<B::Swapchain>,
    acquired: AtomicUsize,
    invalid: AtomicBool,
    usage: ImageUsages,
}
/// Reconfigurable chain. Resizing requires all acquisitions to be consumed.
pub struct GpuSwapchain<B: Backend> {
    format: SurfaceFormat,
    extent: Extent3d,
    image_count: NonZeroU32,
    device: Arc<super::device::Device<B>>,
    chain: Arc<Chain<B>>,
}
/// Owned acquisition. Image and view access borrow this frame.
pub struct GpuSurfaceFrame<B: Backend> {
    backend: Arc<B>,
    gate: Arc<Mutex<()>>,
    chain: Arc<Chain<B>>,
    pub(super) native: Mutex<Option<B::SurfaceFrame>>,
    pub(super) phase: AtomicU8,
    image: GpuImage<B>,
    view: GpuImageView<B>,
    format: SurfaceFormat,
    status: SurfaceStatus,
}
impl<B: Backend> Rhi<B> {
    /// Creates a presentation chain, retaining its platform target through the native surface.
    pub fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<GpuSwapchain<B>> {
        let _gate = self.device.gate.lock();
        let raw = unsafe {
            self.device.backend.create_swapchain(&SwapchainDesc::<B> {
                label: desc.label,
                surface: desc.surface.raw(),
                width: desc.width,
                height: desc.height,
                usage: desc.usage,
                preferred_formats: desc.preferred_formats,
                desired_image_count: desc.desired_image_count,
                present_mode: desc.present_mode,
            })?
        };
        Ok(GpuSwapchain {
            format: raw.format(),
            extent: raw.extent(),
            image_count: raw.image_count(),
            device: self.device.clone(),
            chain: Arc::new(Chain {
                raw: Mutex::new(raw),
                acquired: AtomicUsize::new(0),
                invalid: AtomicBool::new(false),
                usage: desc.usage,
            }),
        })
    }
}
impl<B: Backend> GpuSwapchain<B> {
    /// Selected surface format.
    pub fn format(&self) -> SurfaceFormat {
        self.format
    }
    /// Selected surface extent.
    pub fn extent(&self) -> Extent3d {
        self.extent
    }
    /// Actual native image count.
    pub fn image_count(&self) -> NonZeroU32 {
        self.image_count
    }
    /// Acquires a frame, respecting the caller's timeout and in-flight budget.
    pub fn acquire(&mut self, timeout_ns: u64) -> Result<GpuSurfaceFrame<B>> {
        if self.chain.invalid.load(Ordering::Acquire) {
            return Err(crate::Error::SwapchainOutOfDate);
        }
        let _gate = self.device.gate.lock();
        let mut chain = self.chain.raw.lock();
        let mut raw = unsafe { chain.acquire(timeout_ns)? };
        let format = raw.format();
        let extent = raw.extent();
        let status = raw.status();
        let allocation = ImageInfo {
            dimension: ImageDimension::TwoD,
            extent,
            format: format.texture,
            usage: self.chain.usage,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::One,
        };
        let (native_image, native_view) = raw.take_resources()?;
        let image = Object {
            raw: native_image,
            metadata: allocation,
            marker: std::marker::PhantomData,
        };
        let view = Object {
            raw: native_view,
            metadata: ViewMetadata {
                image: allocation,
                range: crate::ImageSubresourceRange::WHOLE.resolve(&allocation)?,
            },
            marker: std::marker::PhantomData,
        };
        self.chain.acquired.fetch_add(1, Ordering::Release);
        Ok(GpuSurfaceFrame {
            backend: self.device.backend.clone(),
            gate: self.device.gate.clone(),
            chain: self.chain.clone(),
            native: Mutex::new(Some(raw)),
            phase: AtomicU8::new(0),
            image,
            view,
            format,
            status,
        })
    }
    /// Safely abandons an unsubmitted acquisition. Failure invalidates the chain.
    pub fn discard(&mut self, frame: GpuSurfaceFrame<B>) -> Result<()> {
        self.require_frame(&frame)?;
        if frame.phase.load(Ordering::Acquire) != 0 {
            return Err(Ir::BadState.into());
        }
        let result = frame.release(false).map(|_| ());
        drop(frame);
        result
    }
    /// Presents a submitted frame. The acquisition ends even when presentation fails.
    pub fn present(&mut self, frame: GpuSurfaceFrame<B>) -> Result<SurfaceStatus> {
        self.require_frame(&frame)?;
        if frame.phase.load(Ordering::Acquire) != 1 {
            return Err(Ir::BadState.into());
        }
        let result = frame.release(true);
        drop(frame);
        result
    }
    fn require_frame(&self, frame: &GpuSurfaceFrame<B>) -> Result<()> {
        if !Arc::ptr_eq(&self.chain, &frame.chain) {
            return Err(Ir::ForeignInstance.into());
        }
        Ok(())
    }
    /// Waits for device work and recreates the chain after all frames have ended.
    pub fn resize(&mut self, width: NonZeroU32, height: NonZeroU32) -> Result<()> {
        if self.chain.acquired.load(Ordering::Acquire) != 0 {
            return Err(Ir::BadState
                .with_detail("consume acquired frames before resize")
                .into());
        }
        self.device.wait_idle()?;
        let _gate = self.device.gate.lock();
        let result = unsafe { self.chain.raw.lock().resize(width, height) };
        self.chain.invalid.store(result.is_err(), Ordering::Release);
        result?;
        let raw = self.chain.raw.lock();
        self.format = raw.format();
        self.extent = raw.extent();
        self.image_count = raw.image_count();
        Ok(())
    }
}
impl<B: Backend> GpuSurfaceFrame<B> {
    /// Acquired image, valid through presentation/discard.
    #[must_use]
    pub fn image(&self) -> &GpuImage<B> {
        &self.image
    }
    /// Default acquired image view.
    #[must_use]
    pub fn view(&self) -> &GpuImageView<B> {
        &self.view
    }
    /// Actual acquired format.
    #[must_use]
    pub fn format(&self) -> SurfaceFormat {
        self.format
    }
    /// Actual acquired extent.
    #[must_use]
    pub fn extent(&self) -> Extent3d {
        self.image.description().extent
    }
    /// Whether recreation is recommended after presentation.
    #[must_use]
    pub fn status(&self) -> SurfaceStatus {
        self.status
    }
    pub(super) fn require_device(&self, backend: &Arc<B>) -> Result<()> {
        if !Arc::ptr_eq(&self.backend, backend) {
            return Err(Ir::ForeignInstance.into());
        }
        if self.phase.load(Ordering::Acquire) != 0 {
            return Err(Ir::BadState.into());
        }
        Ok(())
    }
    pub(super) fn poison(&self, native: &mut Option<B::SurfaceFrame>) {
        // A failed native submission may have consumed acquisition synchronization.
        // Keep uncertain native ownership alive and make this chain unusable.
        if let Some(frame) = native.take() {
            std::mem::forget(frame);
        }
        self.phase.store(2, Ordering::Release);
        self.chain.acquired.fetch_sub(1, Ordering::Release);
        self.chain.invalid.store(true, Ordering::Release);
    }
    fn release(&self, present: bool) -> Result<SurfaceStatus> {
        let _gate = self.gate.lock();
        let mut native = self.native.lock();
        let frame = native.take().ok_or(Ir::BadState)?;
        let result = if present {
            unsafe { self.chain.raw.lock().present(frame) }
        } else {
            unsafe { self.chain.raw.lock().discard(frame) }.map(|()| SurfaceStatus::Optimal)
        };
        self.phase.store(2, Ordering::Release);
        self.chain.acquired.fetch_sub(1, Ordering::Release);
        if result.is_err() {
            self.chain.invalid.store(true, Ordering::Release);
        }
        result
    }
}
impl<B: Backend> Drop for GpuSurfaceFrame<B> {
    fn drop(&mut self) {
        // Abandoned submitted work is presented using its recorded dependencies;
        // unsubmitted acquisitions are discarded. Errors force recreation.
        if self.phase.load(Ordering::Acquire) < 2
            && self
                .release(self.phase.load(Ordering::Acquire) == 1)
                .is_err()
        {
            self.chain.invalid.store(true, Ordering::Release);
        }
    }
}
