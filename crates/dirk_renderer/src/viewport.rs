use dirk_player::PlayerId;
use dirk_rhi::{Extent3d, Format, ImageAspects, ImageUsages, SampleCount};
use dirk_universe::{Entity, WorldId};

use crate::{
    Result,
    frame_graph::ImportedTexture,
    resources::{
        ActiveTimelineSemaphore,
        device::RenderDevice,
        image::{Image, ImageCreateInfo},
        sync::TimelineSemaphore,
    },
};

#[derive(Clone, Copy)]
pub(crate) struct TextureState {
    state: dirk_rhi::ImageState,
}

pub(crate) struct Viewport {
    player: PlayerId,
    pub camera: Option<Entity>,
    pub world: Option<WorldId>,
    settings: ViewportSettings,
    output: Image,
    output_state: TextureState,
    render_semaphore: TimelineSemaphore,
    last_render_value: u64,
    output_has_rendered: bool,
}

impl Viewport {
    pub fn new(
        device: &RenderDevice,
        player: PlayerId,
        settings: ViewportSettings,
    ) -> Result<Self> {
        let settings = settings.clamped();

        Ok(Self {
            player,
            camera: None,
            world: None,
            settings,
            output: Self::create_output(device, &settings)?,
            output_state: Viewport::undefined_state(),
            render_semaphore: TimelineSemaphore::create(&device.rhi, 0)?,
            last_render_value: 0,
            output_has_rendered: false,
        })
    }

    pub fn player(&self) -> PlayerId {
        self.player
    }
    pub fn settings(&self) -> &ViewportSettings {
        &self.settings
    }

    #[cfg(feature = "editor")]
    pub fn output_rhi_view(&self) -> &crate::resources::ActiveImageView {
        self.output.rhi_view()
    }
    pub fn is_renderable(&self) -> bool {
        self.world.is_some() && self.camera.is_some()
    }
    pub fn has_rendered(&self) -> bool {
        self.output_has_rendered
    }

    pub fn resize(&mut self, device: &RenderDevice, extent: Extent3d) -> Result<()> {
        self.reconfigure(
            device,
            ViewportSettings {
                extent,
                ..self.settings
            },
        )
    }
    pub fn reconfigure(&mut self, device: &RenderDevice, settings: ViewportSettings) -> Result<()> {
        let settings = settings.clamped();
        if self.settings == settings {
            return Ok(());
        }

        self.settings = settings;
        self.output = Self::create_output(device, &self.settings)?;
        self.output_state = Self::undefined_state();
        self.output_has_rendered = false;
        Ok(())
    }

    pub fn import(&self) -> ImportedTexture {
        ImportedTexture {
            image: self.output.rhi_image().clone(),
            view: self.output.rhi_view().clone(),
            aspects: self.output.rhi_aspects(),
            initial_state: self.output_state.state,
            final_state: Self::shader_read_state().state,
        }
    }

    #[cfg(not(feature = "editor"))]
    pub fn import_after_render(&self) -> ImportedTexture {
        let mut import = self.import();
        import.initial_state = Self::shader_read_state().state;
        import
    }

    pub fn next_render_value(&self) -> u64 {
        self.last_render_value + 1
    }

    pub fn render_semaphore(&self) -> &ActiveTimelineSemaphore {
        self.render_semaphore.rhi()
    }

    pub fn mark_render_submitted(&mut self, value: u64) {
        self.last_render_value = value;
        self.output_state = Self::shader_read_state();
        self.output_has_rendered = true;
    }

    fn undefined_state() -> TextureState {
        TextureState {
            state: dirk_rhi::ImageState::Undefined,
        }
    }

    fn shader_read_state() -> TextureState {
        TextureState {
            state: dirk_rhi::ImageState::ShaderRead,
        }
    }

    fn create_output(device: &RenderDevice, settings: &ViewportSettings) -> Result<Image> {
        Image::create_image(
            device,
            &ImageCreateInfo {
                extent: settings.extent,
                format: settings.format,
                usage: ImageUsages::COLOR_ATTACHMENT | ImageUsages::SAMPLED | ImageUsages::COPY_SRC,
                mip_levels: 1,
                samples: SampleCount::One,
                aspects: ImageAspects::COLOR,
            },
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct ViewportSettings {
    pub extent: Extent3d,
    pub format: Format,
    pub clear_color: [f32; 4],
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl ViewportSettings {
    pub(crate) fn new(extent: Extent3d, format: Format) -> Self {
        Self {
            extent,
            format,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            fov_y_radians: 45_f32.to_radians(),
            near: 0.1,
            far: 100_000.0,
        }
    }

    fn clamped(self) -> Self {
        Self {
            extent: Extent3d::new_2d(self.extent.width.max(1), self.extent.height.max(1)),
            ..self
        }
    }
}
