use std::{
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use core_graphics_types::geometry::CGSize;
use dirk_rhi::{
    ColorSpace, Extent3d, ImageUsages, InvalidResourceKind as Ir, Result, SurfaceCreateInfo,
    SurfaceFormat, SurfaceFrame, SurfaceStatus, Swapchain, SwapchainDesc, TextureFormat,
};
use metal::foreign_types::ForeignType;
use metal::{MTLPixelFormat, MetalDrawable, MetalLayer};
use raw_window_handle::RawWindowHandle;

use crate::{
    MetalBackend,
    backend::Context,
    resource::{MetalImage, MetalImageView, require_context},
};

struct SurfaceInner {
    context: Arc<Context>,
    layer: MetalLayer,
    _target: Arc<dyn dirk_rhi::SurfaceTarget>,
}

/// Metal presentation surface backed by a `CAMetalLayer`.
#[derive(Clone)]
pub struct MetalSurface(Arc<SurfaceInner>);

impl MetalSurface {
    pub(crate) fn create(context: &Arc<Context>, info: &SurfaceCreateInfo) -> Result<Self> {
        let window = info.window_handle().map_err(|error| {
            dirk_rhi::Error::Backend(anyhow::anyhow!("window handle is unavailable: {error:?}"))
        })?;
        let raw_layer = match window.as_raw() {
            RawWindowHandle::AppKit(handle) => {
                // SAFETY: `SurfaceCreateInfo` carries a borrowed native window
                // handle supplied by the platform window implementation.
                unsafe { raw_window_metal::Layer::from_ns_view(handle.ns_view) }
            }
            RawWindowHandle::UiKit(handle) => {
                // SAFETY: As above, this pointer is a valid UIView for the
                // lifetime managed by the platform window.
                unsafe { raw_window_metal::Layer::from_ui_view(handle.ui_view) }
            }
            _ => {
                return Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
                    "Metal surfaces require an AppKit or UIKit window"
                )));
            }
        };
        let pointer = raw_layer.into_raw().as_ptr().cast::<metal::CAMetalLayer>();
        // SAFETY: `into_raw` transfers a +1 retained CAMetalLayer reference.
        let layer = unsafe { MetalLayer::from_ptr(pointer) };
        layer.set_device(&context.device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
        layer.set_framebuffer_only(false);
        Ok(Self(Arc::new(SurfaceInner {
            context: context.clone(),
            layer,
            _target: info.target().clone(),
        })))
    }
}

/// Reconfigurable Metal layer presentation state.
pub struct MetalSwapchain {
    pub(crate) context: Arc<Context>,
    surface: MetalSurface,
    format: SurfaceFormat,
    extent: Extent3d,
    image_count: NonZeroU32,
}

impl MetalSwapchain {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &SwapchainDesc<'_, MetalBackend>,
    ) -> Result<Self> {
        require_context(context, &desc.surface.0.context)?;
        let supported = [
            SurfaceFormat {
                texture: TextureFormat::Bgra8Srgb,
                color_space: ColorSpace::Srgb,
            },
            SurfaceFormat {
                texture: TextureFormat::Bgra8Unorm,
                color_space: ColorSpace::Srgb,
            },
        ];
        let format = desc
            .preferred_formats
            .iter()
            .find(|preferred| supported.contains(preferred))
            .copied()
            .unwrap_or(supported[0]);
        desc.surface
            .0
            .layer
            .set_pixel_format(crate::convert::format(format.texture));
        desc.surface.0.layer.set_framebuffer_only(
            !desc.usage.contains(ImageUsages::COPY_SRC)
                && !desc.usage.contains(ImageUsages::COPY_DST)
                && !desc.usage.contains(ImageUsages::SAMPLED)
                && !desc.usage.contains(ImageUsages::STORAGE),
        );
        set_extent(&desc.surface.0.layer, desc.width.get(), desc.height.get());
        let image_count = desc.desired_image_count.unwrap_or(NonZeroU32::MIN);
        desc.surface
            .0
            .layer
            .set_maximum_drawable_count(u64::from(image_count.get()));
        Ok(Self {
            context: context.clone(),
            surface: desc.surface.clone(),
            format,
            extent: Extent3d::new_2d(desc.width.get(), desc.height.get()),
            image_count,
        })
    }
}

impl Swapchain<MetalBackend> for MetalSwapchain {
    fn format(&self) -> SurfaceFormat {
        self.format
    }

    fn extent(&self) -> Extent3d {
        self.extent
    }

    fn image_count(&self) -> NonZeroU32 {
        self.image_count
    }

    fn acquire(&mut self, timeout_ns: u64) -> Result<MetalSurfaceFrame> {
        if timeout_ns == 0 {
            return Err(dirk_rhi::Error::Timeout);
        }
        let drawable = self
            .surface
            .0
            .layer
            .next_drawable()
            .ok_or(dirk_rhi::Error::SwapchainOutOfDate)?
            .to_owned();
        let texture = drawable.texture().to_owned();
        let image = MetalImage::surface(&self.context, texture.clone(), self.format.texture);
        let view = MetalImageView::surface(&self.context, texture);
        Ok(MetalSurfaceFrame {
            context: self.context.clone(),
            drawable,
            image,
            view,
            format: self.format,
            extent: self.extent,
            submitted: AtomicBool::new(false),
        })
    }

    fn discard(&mut self, frame: MetalSurfaceFrame) -> Result<()> {
        require_context(&self.context, &frame.context)?;
        if frame.was_submitted() {
            return Err(Ir::BadState.into());
        }
        Ok(())
    }

    fn resize(&mut self, width: NonZeroU32, height: NonZeroU32) -> Result<()> {
        set_extent(&self.surface.0.layer, width.get(), height.get());
        self.extent = Extent3d::new_2d(width.get(), height.get());
        Ok(())
    }

    fn present(&mut self, frame: MetalSurfaceFrame) -> Result<SurfaceStatus> {
        require_context(&self.context, &frame.context)?;
        if frame.was_submitted() {
            // `CAMetalLayer` presents at vertical blank and never reports a
            // suboptimal chain.
            Ok(SurfaceStatus::Optimal)
        } else {
            Err(Ir::BadState.into())
        }
    }
}

/// One drawable acquired from a Metal layer.
pub struct MetalSurfaceFrame {
    pub(crate) context: Arc<Context>,
    pub(crate) drawable: MetalDrawable,
    image: MetalImage,
    view: MetalImageView,
    format: SurfaceFormat,
    extent: Extent3d,
    submitted: AtomicBool,
}

impl MetalSurfaceFrame {
    pub(crate) fn mark_submitted(&self) -> Result<()> {
        self.submitted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Ir::BadState.into())
    }

    pub(crate) fn was_submitted(&self) -> bool {
        self.submitted.load(Ordering::Acquire)
    }
}

impl SurfaceFrame<MetalBackend> for MetalSurfaceFrame {
    fn image(&self) -> &MetalImage {
        &self.image
    }

    fn view(&self) -> &MetalImageView {
        &self.view
    }

    fn format(&self) -> SurfaceFormat {
        self.format
    }

    fn extent(&self) -> Extent3d {
        self.extent
    }

    fn status(&self) -> SurfaceStatus {
        SurfaceStatus::Optimal
    }
}

fn set_extent(layer: &metal::MetalLayerRef, width: u32, height: u32) {
    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
}
