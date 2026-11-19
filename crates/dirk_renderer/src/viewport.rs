use dirk_render_utils::{Handle, RenderGraph};
use dirk_rhi::{
    Extent3d, Frame, Image, ImageAccess, ImageDesc, ImageDimension, ImageUsage, LoadOp, Rhi,
    SampleCount, ShaderStages, TextureIndex,
};
use dirk_universe::{Entity, WorldId};

use crate::{
    COLOR_FORMAT, Result, Sampled,
    models::ModelRegistry,
    proxy::scene::{Projection, SceneManager, SceneTargets},
};

/// How viewport outputs are left for the passes that show them.
pub(crate) const SHOWN: ImageAccess = ImageAccess::Sampled(ShaderStages::FRAGMENT);

/// A player's view of a world, rendered to its own output image, which
/// windows and editor panels show.
pub(crate) struct Viewport {
    pub camera: Option<Entity>,
    pub world: Option<WorldId>,
    settings: ViewportSettings,
    targets: SceneTargets,
    images: ViewportImages,
}

/// The images a viewport renders to. One set per viewport: frames in
/// flight render to it in turn, ordered by the graph's barriers.
struct ViewportImages {
    /// Linear colors, sampled by whatever shows the viewport.
    output: Image,
    depth: Image,
    /// Multisampled color, resolved into `output`.
    multisampled: Option<Image>,
    /// Whether a submitted frame rendered to the images, so they are in
    /// their attachment and [`SHOWN`] accesses rather than undefined.
    rendered: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct ViewportSettings {
    pub extent: (u32, u32),
    /// Linear RGBA.
    pub clear_color: [f32; 4],
    pub projection: Projection,
}

impl Viewport {
    pub fn new(rhi: &Rhi, targets: SceneTargets) -> Result<Self> {
        let settings = ViewportSettings {
            extent: (1, 1),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            projection: Projection {
                fov_y_radians: 45_f32.to_radians(),
                near: 0.1,
                far: 100_000.0,
            },
        };
        Ok(Self {
            camera: None,
            world: None,
            images: ViewportImages::new(rhi, settings.extent, targets)?,
            settings,
            targets,
        })
    }

    /// Size of the images, in pixels.
    #[cfg_attr(
        not(feature = "editor"),
        expect(dead_code, reason = "the editor sizes panels")
    )]
    pub fn extent(&self) -> (u32, u32) {
        self.settings.extent
    }
    #[cfg_attr(
        not(feature = "editor"),
        expect(dead_code, reason = "the editor tells which viewports are ready")
    )]
    pub fn is_renderable(&self) -> bool {
        self.world.is_some() && self.camera.is_some()
    }
    /// Whether the output holds a rendered image.
    #[cfg_attr(
        not(feature = "editor"),
        expect(dead_code, reason = "the editor tells which viewports are ready")
    )]
    pub fn has_rendered(&self) -> bool {
        self.images.rendered
    }
    /// The output, to show once it [`has_rendered`](Self::has_rendered).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the renderer shows outputs it records")
    )]
    pub fn output(&self) -> &Image {
        &self.images.output
    }
    /// The output's index in the bindless heap.
    #[cfg_attr(
        not(feature = "editor"),
        expect(dead_code, reason = "the editor shows outputs")
    )]
    pub fn texture(&self) -> dirk_rhi::Result<TextureIndex> {
        self.images.output.texture()
    }

    /// Recreates the images at `extent`, at least one pixel wide and high,
    /// if it differs. Do so while a frame is open: frames already recorded
    /// may still show the old output.
    pub fn resize(&mut self, rhi: &Rhi, extent: (u32, u32)) -> Result<()> {
        let extent = (extent.0.max(1), extent.1.max(1));
        if self.settings.extent != extent {
            self.settings.extent = extent;
            self.images = ViewportImages::new(rhi, extent, self.targets)?;
        }
        Ok(())
    }

    /// Adds the pass rendering the camera's view of its world to `graph`,
    /// leaving the output for fragment shaders to sample, and returns the
    /// output to show. Without a world and camera, the output shows what it
    /// rendered last, if anything. Call
    /// [`mark_rendered`](Self::mark_rendered) once it is submitted.
    pub fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame: &Frame<'_>,
        scene: &'a SceneManager,
        models: &'a ModelRegistry,
    ) -> Result<Option<Handle<Image>>> {
        let images = &self.images;
        let (Some(world), Some(camera)) = (self.world, self.camera) else {
            let shown = images
                .rendered
                .then(|| graph.import_image(&images.output, SHOWN, SHOWN));
            return Ok(shown.transpose()?);
        };
        let (width, height) = self.settings.extent;
        // Extents are far below 2^24, where `f32` loses integers.
        #[expect(clippy::cast_precision_loss, reason = "extents are small")]
        let aspect = width as f32 / height as f32;
        let view = scene.view(world, camera, aspect, self.settings.projection)?;
        let objects = scene.objects(frame);

        let before = |access| {
            if images.rendered {
                access
            } else {
                ImageAccess::Undefined
            }
        };
        let output = graph.import_image(&images.output, before(SHOWN), SHOWN)?;
        let attachment = ImageAccess::DepthStencilAttachment;
        let depth = graph.import_image(&images.depth, before(attachment), attachment)?;
        let multisampled = images
            .multisampled
            .as_ref()
            .map(|image| {
                let attachment = ImageAccess::ColorAttachment;
                graph.import_image(image, before(attachment), attachment)
            })
            .transpose()?;
        let clear = LoadOp::Clear(self.settings.clear_color);
        let pass = graph.pass("scene");
        let pass = match multisampled {
            Some(multisampled) => pass.color_resolved(multisampled, clear, output),
            None => pass.color(output, clear),
        };
        pass.depth(depth, LoadOp::Clear(1.0))
            .record(move |pass| scene.draw(pass, objects, models, &view))?;
        Ok(Some(output))
    }

    /// Records that a frame rendered to the images and was submitted.
    pub fn mark_rendered(&mut self) {
        self.images.rendered = true;
    }
}

impl ViewportImages {
    fn new(rhi: &Rhi, (width, height): (u32, u32), targets: SceneTargets) -> Result<Self> {
        let image = |label, format, usage, samples| {
            rhi.create_image(&ImageDesc {
                label,
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(width, height),
                format,
                usage,
                mip_levels: 1,
                array_layers: 1,
                samples,
            })
        };
        let multisampled = targets.samples != SampleCount::X1;
        Ok(Self {
            output: image(
                "viewport output",
                COLOR_FORMAT,
                ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                SampleCount::X1,
            )?,
            depth: image(
                "viewport depth",
                targets.depth,
                ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                targets.samples,
            )?,
            multisampled: multisampled
                .then(|| {
                    image(
                        "viewport multisampled color",
                        COLOR_FORMAT,
                        ImageUsage::COLOR_ATTACHMENT,
                        targets.samples,
                    )
                })
                .transpose()?,
            rendered: false,
        })
    }
}
