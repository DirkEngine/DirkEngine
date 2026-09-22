use crate::resources::{
    buffer::UniformBuffer,
    descriptors::{BindingLayout, DescriptorSet, sets::SceneSet},
};
use dirk_player::PlayerId;
use dirk_rhi::{Extent3d, ImageUsages, SampleCount, TextureFormat};
use dirk_shaders::types::SceneUbo;
use dirk_universe::{Entity, WorldId};

use crate::{
    Result,
    frame_graph::ImportedTexture,
    resources::{
        Rhi,
        image::{Image, ImageCreateInfo},
    },
};

pub(crate) struct Viewport {
    player: PlayerId,
    camera_ubo: [UniformBuffer<SceneUbo>; crate::MAX_FRAMES_IN_FLIGHT],
    camera_sets: [DescriptorSet<SceneSet>; crate::MAX_FRAMES_IN_FLIGHT],
    pub camera: Option<Entity>,
    pub world: Option<WorldId>,
    settings: ViewportSettings,
    output: Image,
    output_state: dirk_rhi::ImageState,
    output_has_rendered: bool,
}

impl Viewport {
    pub fn new(device: &Rhi, player: PlayerId, settings: ViewportSettings) -> Result<Self> {
        let settings = settings.clamped();

        let camera_ubo = [UniformBuffer::new(device)?, UniformBuffer::new(device)?];
        let allocator = BindingLayout::<SceneSet>::new(device)?;
        let camera_sets = [
            allocator.uniform_buffer(
                device,
                0,
                camera_ubo[0].buffer(),
                size_of::<SceneUbo>() as u64,
            )?,
            allocator.uniform_buffer(
                device,
                0,
                camera_ubo[1].buffer(),
                size_of::<SceneUbo>() as u64,
            )?,
        ];
        Ok(Self {
            camera_ubo,
            camera_sets,
            player,
            camera: None,
            world: None,
            settings,
            output: Self::create_output(device, &settings)?,
            output_state: Viewport::undefined_state(),
            output_has_rendered: false,
        })
    }

    pub fn camera_buffer(&self, frame: usize) -> &dirk_rhi::Buffer {
        self.camera_ubo[frame].buffer()
    }
    pub fn camera_set(&self, frame: usize) -> &DescriptorSet<SceneSet> {
        &self.camera_sets[frame]
    }
    pub fn prepare_camera(&mut self, frame: usize, view: glam::Mat4) -> Result<()> {
        #[allow(clippy::cast_precision_loss)]
        let aspect = self.settings.extent.width as f32 / self.settings.extent.height as f32;
        let proj = glam::camera::lh::proj::vulkan::perspective(
            self.settings.fov_y_radians,
            aspect,
            self.settings.near,
            self.settings.far,
        );
        // SAFETY: the renderer waited for this frame slot before preparing any view.
        unsafe {
            self.camera_ubo[frame].write(&SceneUbo { view, proj })?;
        }
        Ok(())
    }
    pub fn player(&self) -> PlayerId {
        self.player
    }
    pub fn settings(&self) -> &ViewportSettings {
        &self.settings
    }

    #[cfg(feature = "editor")]
    pub fn output_rhi_view(&self) -> &crate::resources::ImageView {
        self.output.rhi_view()
    }
    pub fn is_renderable(&self) -> bool {
        self.world.is_some() && self.camera.is_some()
    }
    pub fn has_rendered(&self) -> bool {
        self.output_has_rendered
    }

    pub fn resize(&mut self, device: &Rhi, extent: Extent3d) -> Result<()> {
        self.reconfigure(
            device,
            ViewportSettings {
                extent,
                ..self.settings
            },
        )
    }
    pub fn reconfigure(&mut self, device: &Rhi, settings: ViewportSettings) -> Result<()> {
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

    pub fn import(&self) -> ImportedTexture<'_> {
        ImportedTexture {
            image: self.output.rhi_image(),
            view: self.output.rhi_view(),
            initial_state: self.output_state,
            final_state: Self::shader_read_state(),
        }
    }

    pub fn import_after_render(&self) -> ImportedTexture<'_> {
        let mut import = self.import();
        import.initial_state = Self::shader_read_state();
        import
    }

    pub fn mark_render_submitted(&mut self) {
        self.output_state = Self::shader_read_state();
        self.output_has_rendered = true;
    }
    pub fn invalidate(&mut self) {
        self.output_has_rendered = false;
    }
    fn undefined_state() -> dirk_rhi::ImageState {
        dirk_rhi::ImageState::Undefined
    }
    fn shader_read_state() -> dirk_rhi::ImageState {
        dirk_rhi::ImageState::ShaderRead(dirk_rhi::ShaderStages::FRAGMENT)
    }

    fn create_output(device: &Rhi, settings: &ViewportSettings) -> Result<Image> {
        Image::create_image(
            device,
            &ImageCreateInfo {
                extent: settings.extent,
                format: settings.format,
                usage: ImageUsages::COLOR_ATTACHMENT | ImageUsages::SAMPLED | ImageUsages::COPY_SRC,
                mip_levels: 1,
                samples: SampleCount::One,
            },
        )
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct ViewportSettings {
    pub extent: Extent3d,
    pub format: TextureFormat,
    pub clear_color: [f32; 4],
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl ViewportSettings {
    pub(crate) fn new(extent: Extent3d, format: TextureFormat) -> Self {
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
