use super::{
    Arc, AtomicUsize, Backend, GpuImage, GpuImageView, ImageMetadata, Mutex, Ordering, Rhi,
    ViewMetadata, identity, lock,
};
use crate::{
    Extent3d, ImageDimension, ImageInfo, ImageUsages, InvalidResourceKind as Ir, Result,
    SampleCount, SurfaceFormat, SurfaceFrame as _, SurfaceStatus, Swapchain as _, SwapchainDesc,
};
use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicBool, AtomicU8},
};

#[derive(Debug)]
pub(super) struct FrameState {
    pub(super) id: u64,
    pub(super) phase: AtomicU8,
}
struct Chain<B: Backend> {
    raw: Mutex<B::Swapchain>,
    id: u64,
    acquired: AtomicUsize,
    invalid: AtomicBool,
    usage: ImageUsages,
}
/// Reconfigurable chain. Resizing requires all acquisitions to be consumed.
pub struct GpuSwapchain<B: Backend> {
    device: Rhi<B>,
    chain: Arc<Chain<B>>,
}
/// Owned acquisition. Images retain its lifetime token and reject stale use.
pub struct GpuSurfaceFrame<B: Backend> {
    backend: Arc<B>,
    gate: Arc<Mutex<()>>,
    chain: Arc<Chain<B>>,
    pub(super) native: Mutex<Option<B::SurfaceFrame>>,
    pub(super) state: Arc<FrameState>,
    image: GpuImage<B>,
    view: GpuImageView<B>,
    format: SurfaceFormat,
    status: SurfaceStatus,
}
impl<B: Backend> Rhi<B> {
    /// Creates a presentation chain, retaining its platform target through the native surface.
    pub fn create_swapchain(&self, desc: &SwapchainDesc<'_, Self>) -> Result<GpuSwapchain<B>> {
        desc.surface.require_device(&self.0.backend)?;
        let _gate = lock(&self.0.gate)?;
        let raw = unsafe {
            self.0.backend.create_swapchain(&SwapchainDesc::<B> {
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
            device: self.clone(),
            chain: Arc::new(Chain {
                raw: Mutex::new(raw),
                id: identity(),
                acquired: AtomicUsize::new(0),
                invalid: AtomicBool::new(false),
                usage: desc.usage,
            }),
        })
    }
}
impl<B: Backend> GpuSwapchain<B> {
    /// Selected surface format.
    pub fn format(&self) -> Result<SurfaceFormat> {
        Ok(lock(&self.chain.raw)?.format())
    }
    /// Selected surface extent.
    pub fn extent(&self) -> Result<Extent3d> {
        Ok(lock(&self.chain.raw)?.extent())
    }
    /// Actual native image count.
    pub fn image_count(&self) -> Result<NonZeroU32> {
        Ok(lock(&self.chain.raw)?.image_count())
    }
    /// Acquires a frame, respecting the caller's timeout and in-flight budget.
    pub fn acquire(&mut self, timeout_ns: u64) -> Result<GpuSurfaceFrame<B>> {
        if self.chain.invalid.load(Ordering::Acquire) {
            return Err(crate::Error::SwapchainOutOfDate);
        }
        let _gate = lock(&self.device.0.gate)?;
        let mut chain = lock(&self.chain.raw)?;
        let raw = unsafe { chain.acquire(timeout_ns)? };
        let format = raw.format();
        let extent = raw.extent();
        let status = raw.status();
        let state = Arc::new(FrameState {
            id: identity(),
            phase: AtomicU8::new(0),
        });
        let allocation = ImageInfo {
            dimension: ImageDimension::TwoD,
            extent,
            format: format.texture,
            usage: self.chain.usage,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::One,
        };
        let image = self.device.object(
            raw.image().clone(),
            ImageMetadata {
                allocation,
                frame: Some(state.clone()),
            },
        );
        let view = self.device.object(
            raw.view().clone(),
            ViewMetadata {
                image: image.clone(),
                range: crate::ImageSubresourceRange::WHOLE.resolve(&allocation)?,
            },
        );
        self.chain.acquired.fetch_add(1, Ordering::Release);
        Ok(GpuSurfaceFrame {
            backend: self.device.0.backend.clone(),
            gate: self.device.0.gate.clone(),
            chain: self.chain.clone(),
            native: Mutex::new(Some(raw)),
            state,
            image,
            view,
            format,
            status,
        })
    }
    /// Safely abandons an unsubmitted acquisition. Failure invalidates the chain.
    pub fn discard(&mut self, frame: GpuSurfaceFrame<B>) -> Result<()> {
        self.require_frame(&frame)?;
        if frame.state.phase.load(Ordering::Acquire) != 0 {
            return Err(Ir::BadState.into());
        }
        let result = frame.release(false).map(|_| ());
        drop(frame);
        result
    }
    /// Presents a submitted frame. The acquisition ends even when presentation fails.
    pub fn present(&mut self, frame: GpuSurfaceFrame<B>) -> Result<SurfaceStatus> {
        self.require_frame(&frame)?;
        if frame.state.phase.load(Ordering::Acquire) != 1 {
            return Err(Ir::BadState.into());
        }
        let result = frame.release(true);
        drop(frame);
        result
    }
    fn require_frame(&self, frame: &GpuSurfaceFrame<B>) -> Result<()> {
        if self.chain.id != frame.chain.id {
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
        let _gate = lock(&self.device.0.gate)?;
        let result = unsafe { lock(&self.chain.raw)?.resize(width, height) };
        self.chain.invalid.store(result.is_err(), Ordering::Release);
        result
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
        if self.state.phase.load(Ordering::Acquire) != 0 {
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
        self.state.phase.store(2, Ordering::Release);
        self.chain.acquired.fetch_sub(1, Ordering::Release);
        self.chain.invalid.store(true, Ordering::Release);
    }
    fn release(&self, present: bool) -> Result<SurfaceStatus> {
        let _gate = lock(&self.gate)?;
        let mut native = lock(&self.native)?;
        let frame = native.take().ok_or(Ir::BadState)?;
        let result = if present {
            unsafe { lock(&self.chain.raw)?.present(frame) }
        } else {
            unsafe { lock(&self.chain.raw)?.discard(frame) }.map(|()| SurfaceStatus::Optimal)
        };
        self.state.phase.store(2, Ordering::Release);
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
        if self.state.phase.load(Ordering::Acquire) < 2
            && self
                .release(self.state.phase.load(Ordering::Acquire) == 1)
                .is_err()
        {
            self.chain.invalid.store(true, Ordering::Release);
        }
    }
}
