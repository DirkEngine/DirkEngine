use dirk_platform::WindowId;
use dirk_render_utils::{Handle, RenderGraph};
use dirk_rhi::{
    AcquiredImage, ColorTarget, CullMode, Format, Frame, FrontFace, GraphicsPipeline,
    GraphicsPipelineDesc, Image, ImageUsage, LoadOp, PresentMode, Rhi, SampleCount, Sampler,
    SamplerDesc, ShaderModule, ShaderStages, Swapchain, SwapchainDesc, Topology,
};
use dirk_shaders::types::PresentPush;

use crate::{
    Result, Sampled,
    shaders::{PresentEncodeFs, PresentFs, PresentVs},
};

/// The renderer's side of a platform window: its swapchain and the pipeline
/// presenting to it.
pub struct Window {
    id: WindowId,
    swapchain: Swapchain,
    pipeline: GraphicsPipeline<PresentPush>,
    /// Size in pixels, as last reported by the platform.
    extent: (u32, u32),
    /// Whether the window is hidden, such as minimized.
    occluded: bool,
}

/// A window's image to present this frame, and the pipeline presenting to
/// it.
pub struct Target<'a> {
    pub window: WindowId,
    pub image: AcquiredImage<'a>,
    pub pipeline: &'a GraphicsPipeline<PresentPush>,
}

impl Window {
    /// Creates the swapchain of `window`, preferring sRGB formats.
    pub fn new(rhi: &Rhi, presenter: &Presenter, window: &dirk_platform::Window) -> Result<Self> {
        let size = window.size();
        let extent = (size.width, size.height);
        let surface = rhi.create_surface(window.surface_target())?;
        let swapchain = rhi.create_swapchain(
            surface,
            &SwapchainDesc {
                label: "window",
                extent,
                formats: &[Format::Bgra8Srgb, Format::Rgba8Srgb],
                present_mode: PresentMode::Fifo,
                usage: ImageUsage::COLOR_ATTACHMENT,
            },
        )?;
        Ok(Self {
            id: window.id(),
            pipeline: presenter.pipeline(rhi, swapchain.format())?,
            swapchain,
            extent,
            occluded: false,
        })
    }

    pub fn id(&self) -> WindowId {
        self.id
    }
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.extent = (width, height);
        self.swapchain.resize(width, height);
    }
    pub fn set_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }

    /// Acquires the window's next image in `frame`, or returns `None` when
    /// the window shows nothing: hidden, empty, or not presentable yet.
    pub fn acquire<'a>(&'a mut self, frame: &'a Frame<'_>) -> Result<Option<Target<'a>>> {
        if self.occluded || self.extent.0 == 0 || self.extent.1 == 0 {
            return Ok(None);
        }
        match frame.acquire(&mut self.swapchain) {
            Ok(image) => Ok(Some(Target {
                window: self.id,
                image,
                pipeline: &self.pipeline,
            })),
            Err(dirk_rhi::Error::SwapchainOutOfDate) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }
}

/// Shows images of linear colors in windows, encoding them exactly once:
/// windows with an sRGB format encode in hardware, others in the shader.
pub struct Presenter {
    vertex: ShaderModule<PresentPush>,
    /// For sRGB targets.
    fragment: ShaderModule<PresentPush>,
    /// For other targets, encoding colors itself.
    encode: ShaderModule<PresentPush>,
    sampler: Sampler,
}

impl Presenter {
    pub fn new(rhi: &Rhi) -> Result<Self> {
        Ok(Self {
            vertex: rhi.create_shader::<PresentVs>()?.with_push(),
            fragment: rhi.create_shader::<PresentFs>()?,
            encode: rhi.create_shader::<PresentEncodeFs>()?,
            sampler: rhi.create_sampler(&SamplerDesc {
                label: "presentation",
                ..SamplerDesc::default()
            })?,
        })
    }

    /// The pipeline presenting to targets of `format`.
    pub fn pipeline(&self, rhi: &Rhi, format: Format) -> Result<GraphicsPipeline<PresentPush>> {
        let fragment = if format.is_srgb() {
            &self.fragment
        } else {
            &self.encode
        };
        Ok(rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: "presentation",
            vertex: &self.vertex,
            fragment: Some(fragment),
            topology: Topology::TriangleList,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            color_targets: &[ColorTarget::new(format)],
            depth: None,
            samples: SampleCount::X1,
        })?)
    }

    /// Adds the pass covering `target` with `source`, stretched to fit, or
    /// clearing it to black without a source. `pipeline` presents to
    /// `target`'s format.
    pub fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        pipeline: &'a GraphicsPipeline<PresentPush>,
        target: Handle<Image>,
        source: Option<Handle<Image>>,
    ) -> Result<()> {
        let pass = graph.pass("present");
        let Some(source) = source else {
            pass.color(target, LoadOp::Clear([0.0, 0.0, 0.0, 1.0]))
                .record(|_| Ok(()))?;
            return Ok(());
        };
        let sampler = self.sampler.index();
        pass.sampled(source, ShaderStages::FRAGMENT)
            .color(target, LoadOp::Discard)
            .record(move |pass| {
                let push = PresentPush {
                    source: pass.image(source)?.texture()?,
                    sampler,
                };
                pass.draw(pipeline, &push, 0..3, 0..1)
            })?;
        Ok(())
    }
}
