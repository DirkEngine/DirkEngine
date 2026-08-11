use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use core_graphics_types::geometry::CGSize;
use dirk_rhi::{
    Extent3d, Format, ImageUsages, Result, SurfaceCreateInfo, SurfaceFrame, SurfaceStatus,
    Swapchain, SwapchainDesc,
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
}

/// Metal presentation surface backed by a `CAMetalLayer`.
#[derive(Clone)]
pub struct MetalSurface(Arc<SurfaceInner>);

impl MetalSurface {
    pub(crate) fn create(context: &Arc<Context>, info: SurfaceCreateInfo) -> Result<Self> {
        let raw_layer = match info.window {
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
                return Err(dirk_rhi::Error::Unsupported(
                    "Metal surfaces require an AppKit or UIKit window",
                ));
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
        })))
    }
}

/// Reconfigurable Metal layer presentation state.
pub struct MetalSwapchain {
    pub(crate) context: Arc<Context>,
    surface: MetalSurface,
    format: Format,
    extent: Extent3d,
}

impl MetalSwapchain {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &SwapchainDesc<'_, MetalBackend>,
    ) -> Result<Self> {
        require_context(context, &desc.surface.0.context)?;
        validate_extent(desc.width, desc.height)?;
        let format = Format::Bgra8Srgb;
        desc.surface
            .0
            .layer
            .set_pixel_format(crate::convert::format(format));
        desc.surface.0.layer.set_framebuffer_only(
            !desc.usage.contains(ImageUsages::COPY_SRC)
                && !desc.usage.contains(ImageUsages::COPY_DST)
                && !desc.usage.contains(ImageUsages::SAMPLED)
                && !desc.usage.contains(ImageUsages::STORAGE),
        );
        set_extent(&desc.surface.0.layer, desc.width, desc.height);
        Ok(Self {
            context: context.clone(),
            surface: desc.surface.clone(),
            format,
            extent: Extent3d::new_2d(desc.width, desc.height),
        })
    }
}

impl Swapchain<MetalBackend> for MetalSwapchain {
    fn format(&self) -> Format {
        self.format
    }

    fn extent(&self) -> Extent3d {
        self.extent
    }

    fn acquire(&mut self) -> Result<MetalSurfaceFrame> {
        let drawable = self
            .surface
            .0
            .layer
            .next_drawable()
            .ok_or(dirk_rhi::Error::SurfaceOutOfDate)?
            .to_owned();
        let texture = drawable.texture().to_owned();
        let image = MetalImage::surface(&self.context, texture.clone(), self.format);
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

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        validate_extent(width, height)?;
        set_extent(&self.surface.0.layer, width, height);
        self.extent = Extent3d::new_2d(width, height);
        Ok(())
    }

    fn present(&mut self, frame: MetalSurfaceFrame) -> Result<()> {
        require_context(&self.context, &frame.context)?;
        if frame.was_submitted() {
            Ok(())
        } else {
            Err(dirk_rhi::Error::InvalidResource(
                "Metal surface frame must be submitted before presentation",
            ))
        }
    }
}

/// One drawable acquired from a Metal layer.
pub struct MetalSurfaceFrame {
    pub(crate) context: Arc<Context>,
    pub(crate) drawable: MetalDrawable,
    image: MetalImage,
    view: MetalImageView,
    format: Format,
    extent: Extent3d,
    submitted: AtomicBool,
}

impl MetalSurfaceFrame {
    pub(crate) fn mark_submitted(&self) {
        self.submitted.store(true, Ordering::Release);
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

    fn format(&self) -> Format {
        self.format
    }

    fn extent(&self) -> Extent3d {
        self.extent
    }

    fn status(&self) -> SurfaceStatus {
        SurfaceStatus::Optimal
    }
}

fn validate_extent(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        Err(dirk_rhi::Error::InvalidResource(
            "swapchain extent must be non-zero",
        ))
    } else {
        Ok(())
    }
}

fn set_extent(layer: &metal::MetalLayerRef, width: u32, height: u32) {
    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
}
