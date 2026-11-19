#![doc = include_str!("../README.md")]

use std::collections::HashMap;

use anyhow::Context as _;
#[cfg(feature = "editor")]
use dirk_platform::WindowInputEvent;
use dirk_platform::{PlatformEvent, PlatformWindows, WindowEvent, WindowId};
use dirk_player::PlayerId;
#[cfg(feature = "editor")]
use dirk_player::PlayerInputSender;
#[cfg(not(feature = "editor"))]
use dirk_player::PlayerPresentationAssignments;
use dirk_render_utils::{RenderGraph, Transients, Uploader};
use dirk_rhi::{
    Buffer, BufferIndex, Format, Graphics, Image, ImageAccess, Rhi, RhiCreateInfo, TextureIndex,
};
use dirk_universe::{Entity, Universe, WorldId};
use raw_window_handle::HasDisplayHandle;
use tracing::{debug, info};

mod errors;
#[cfg(feature = "editor")]
pub use egui;
pub use errors::{Error, Result};

#[cfg(feature = "editor")]
mod egui_integration;
#[cfg(feature = "editor")]
use egui_integration::{EguiFrameInput, EguiState};

mod window;
use window::{Presenter, Window};

mod proxy;
use proxy::{scene::SceneManager, systems::RendererSystem};

mod render_commands;
use render_commands::RenderCommandReceiver;

mod models;

mod viewport;
use viewport::Viewport;

#[cfg(feature = "editor")]
mod viewport_editor;
#[cfg(feature = "editor")]
use viewport_editor::ViewportEditor;

#[cfg(test)]
mod tests;

/// The shaders, built from `shaders/` by the engine's shader build.
#[expect(
    unsafe_code,
    reason = "the shader build vouches for the SPIR-V it generates `Shader` implementations for"
)]
#[cfg_attr(
    not(feature = "editor"),
    expect(dead_code, reason = "the egui shaders are only used by the editor")
)]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

/// Format of the renderer's own color images: viewport outputs, the UI and
/// egui's textures. They hold linear colors, sRGB-encoded to keep dark
/// tones precise in 8 bits; sampling decodes them.
const COLOR_FORMAT: Format = Format::Rgba8Srgb;

/// Initial size of each frame's staging buffer for uploads, in bytes.
const STAGING_SIZE: u64 = 1 << 22;

/// Registers renderer integration with the engine.
pub struct RendererPlugin;

impl dirk_engine::EnginePlugin for RendererPlugin {
    fn name(&self) -> &'static str {
        "renderer"
    }

    fn build(&self, builder: &mut dirk_engine::EngineBuilder) -> anyhow::Result<()> {
        builder.with_plugin(dirk_platform::PlatformPlugin)?;
        builder.with_plugin(dirk_assets::AssetsPlugin)?;
        builder.with_plugin(dirk_player::PlayerPlugin)?;

        builder.add_subsystem(|ctx| {
            let platform_windows = ctx.resource::<dirk_platform::PlatformWindows>()?;
            #[cfg(feature = "editor")]
            let editor = ctx.resource::<dirk_engine::editor::EditorServices>()?;
            #[cfg(not(feature = "editor"))]
            let presentation_assignments = ctx.resource::<PlayerPresentationAssignments>()?;
            #[cfg(feature = "editor")]
            let player_input_sender = ctx.resource::<PlayerInputSender>()?;

            let main_window = platform_windows.main_window();
            let mut renderer = Renderer::init(
                ctx.handle().metadata().app_name(),
                &main_window,
                ctx.events(),
                platform_windows.clone(),
                #[cfg(not(feature = "editor"))]
                presentation_assignments,
                #[cfg(feature = "editor")]
                player_input_sender,
                #[cfg(feature = "editor")]
                editor,
            )?;

            let system = renderer.system();
            ctx.configure_universe(|universe| universe.with_system(system));
            Ok(renderer)
        });
        Ok(())
    }
}

/// The bindless index of an image the renderer samples.
trait Sampled {
    /// The image's index in the heap's texture array.
    ///
    /// # Errors
    ///
    /// Returns [`dirk_rhi::Error::Invalid`] if the image is not sampled.
    fn texture(&self) -> dirk_rhi::Result<TextureIndex>;
}

impl Sampled for Image {
    fn texture(&self) -> dirk_rhi::Result<TextureIndex> {
        self.view()
            .and_then(dirk_rhi::ImageView::index)
            .ok_or_else(|| dirk_rhi::Error::Invalid {
                object: "image",
                label: self.label().to_owned(),
                reason: "is not sampled, so shaders cannot read it".to_owned(),
            })
    }
}

/// The bindless index of a buffer the renderer's shaders read.
trait StorageBuffer {
    /// The buffer's index in the heap's storage buffer array, read as `T`s.
    ///
    /// # Errors
    ///
    /// Returns [`dirk_rhi::Error::Invalid`] if the buffer is not a storage
    /// buffer.
    fn storage<T>(&self) -> dirk_rhi::Result<BufferIndex<T>>;
}

impl StorageBuffer for Buffer {
    fn storage<T>(&self) -> dirk_rhi::Result<BufferIndex<T>> {
        self.index().ok_or_else(|| dirk_rhi::Error::Invalid {
            object: "buffer",
            label: self.label().to_owned(),
            reason: "is not a storage buffer, so shaders cannot read it".to_owned(),
        })
    }
}

/// Holds all render state and renders every engine tick.
struct Renderer {
    /// Every platform window's swapchain.
    windows: HashMap<WindowId, Window>,
    window_order: Vec<WindowId>,
    platform_windows: PlatformWindows,
    #[cfg(not(feature = "editor"))]
    presentation_assignments: PlayerPresentationAssignments,
    /// The renderer's proxy of every world.
    scene_manager: SceneManager,
    models: models::ModelRegistry,
    presenter: Presenter,
    /// Immediate-mode UI rendering state.
    #[cfg(feature = "editor")]
    egui: EguiState,
    #[cfg(feature = "editor")]
    egui_window: Option<WindowId>,
    #[cfg(feature = "editor")]
    egui_input_consumer: dirk_events::Consumer<WindowInputEvent>,
    /// Editor window registry rendered through egui.
    #[cfg(feature = "editor")]
    editor: dirk_engine::editor::EditorServices,
    #[cfg(feature = "editor")]
    viewport_editor: ViewportEditor,
    /// Player-owned internal scene render outputs.
    viewports: HashMap<PlayerId, Viewport>,
    uploader: Uploader,
    transients: Transients,

    window_consumer: dirk_events::Consumer<dirk_platform::WindowEvent>,
    platform_consumer: dirk_events::Consumer<dirk_platform::PlatformEvent>,
    player_spawn_consumer: dirk_events::Consumer<dirk_player::PlayerSpawned>,
    player_despawn_consumer: dirk_events::Consumer<dirk_player::PlayerDespawned>,

    /// These receive all the commands from the game thread.
    receivers: Vec<RenderCommandReceiver>,

    /// Dropped last, on the thread dropping the renderer: it waits for the
    /// GPU, then destroys what the other fields dropped, surfaces included,
    /// which releases their native windows.
    rhi: Rhi,
}

fn sort_window_ids(ids: &mut [WindowId]) {
    ids.sort_unstable_by_key(|id| id.into_raw());
}

impl dirk_engine::Subsystem for Renderer {
    fn name(&self) -> &'static str {
        "renderer"
    }

    fn tick(
        &mut self,
        delta_time: f64,
        handle: &dirk_engine::EngineHandle,
        universe: &Universe,
    ) -> anyhow::Result<()> {
        self.tick()?;
        #[cfg(feature = "editor")]
        {
            let input = self.egui_frame_input();
            self.egui_window = Some(input.window_id);
            let frame = dirk_engine::editor::EditorRenderContext::new(delta_time, handle);

            self.viewport_editor.sync_ready_state(&self.viewports);
            let ui_state = self
                .egui
                .run_frame(&input, &self.editor, &frame, universe)?;
            self.platform_windows
                .apply_ui_state(input.window_id, ui_state);
        }

        #[cfg(not(feature = "editor"))]
        let _ = (delta_time, handle, universe);

        self.render().context("rendering")?;

        Ok(())
    }
}

impl Renderer {
    /// Creates the RHI for `window`'s display and everything rendering
    /// needs, with a swapchain for `window`.
    fn init(
        application_name: &str,
        window: &dirk_platform::Window,
        event_manager: &dirk_events::EventManager,
        platform_windows: PlatformWindows,
        #[cfg(not(feature = "editor"))] presentation_assignments: PlayerPresentationAssignments,
        #[cfg(feature = "editor")] player_input_sender: PlayerInputSender,
        #[cfg(feature = "editor")] editor: dirk_engine::editor::EditorServices,
    ) -> Result<Self> {
        let mut rhi = Rhi::new(&RhiCreateInfo {
            application_name,
            validation: cfg!(debug_assertions),
            display: Some(window.display_handle()?),
        })?;
        info!(device = rhi.capabilities().name, "renderer initialized");

        let mut uploader = Uploader::new(&rhi, STAGING_SIZE)?;
        let models = {
            let mut frame = rhi.begin_frame()?;
            let mut uploads = uploader.begin(&mut frame);
            let models = models::ModelRegistry::new(&mut uploads, event_manager)?;
            uploads.submit()?;
            models
        };
        let scene_manager = SceneManager::new(&rhi)?;
        let presenter = Presenter::new(&rhi)?;
        let window = Window::new(&rhi, &presenter, window)?;
        let window_order = vec![window.id()];

        Ok(Self {
            windows: HashMap::from([(window.id(), window)]),
            window_order,
            platform_windows,
            #[cfg(not(feature = "editor"))]
            presentation_assignments,
            scene_manager,
            models,
            presenter,
            #[cfg(feature = "editor")]
            egui: EguiState::new(&rhi)?,
            #[cfg(feature = "editor")]
            egui_window: None,
            #[cfg(feature = "editor")]
            egui_input_consumer: event_manager.subscribe(),
            #[cfg(feature = "editor")]
            editor,
            #[cfg(feature = "editor")]
            viewport_editor: ViewportEditor::new(player_input_sender),
            viewports: HashMap::new(),
            uploader,
            transients: Transients::default(),

            window_consumer: event_manager.subscribe(),
            platform_consumer: event_manager.subscribe(),
            player_spawn_consumer: event_manager.subscribe(),
            player_despawn_consumer: event_manager.subscribe(),
            receivers: Vec::new(),

            rhi,
        })
    }

    #[cfg(feature = "editor")]
    fn primary_window_id(&self) -> Option<WindowId> {
        self.window_order.first().copied()
    }

    #[cfg(feature = "editor")]
    #[expect(clippy::cast_possible_truncation, reason = "scale factors are small")]
    fn egui_frame_input(&mut self) -> EguiFrameInput {
        let window_id = self
            .primary_window_id()
            .unwrap_or_else(|| WindowId::from_raw(0));
        let extent = self.windows.get(&window_id).map_or((1, 1), Window::extent);
        let events = self
            .egui_input_consumer
            .consume_all()
            .filter(|event| event.window == window_id)
            .collect();

        let (native_pixels_per_point, focused, theme) = {
            let windows = self.platform_windows.windows();
            windows.get(&window_id).map_or((1.0, true, None), |window| {
                (
                    window.scale_factor() as f32,
                    window.focused(),
                    Some(window.theme()),
                )
            })
        };

        EguiFrameInput {
            window_id,
            extent,
            native_pixels_per_point,
            focused,
            theme,
            events,
        }
    }

    /// Returns the system mirroring universe changes into this [`Renderer`].
    fn system(&mut self) -> RendererSystem {
        let (sender, receiver) = render_commands::channel();
        self.receivers.push(receiver);
        RendererSystem::new(sender)
    }

    /// Updates the renderer's state from the events since the last tick:
    /// players, universe changes, windows.
    fn tick(&mut self) -> Result<()> {
        for event in self.player_spawn_consumer.consume_all() {
            #[cfg(feature = "editor")]
            self.viewport_editor.remove_viewport(event.id, &self.editor);

            let viewport = Viewport::new(&self.rhi, self.scene_manager.targets())?;
            #[cfg(feature = "editor")]
            self.viewport_editor
                .add_viewport(event.id, &viewport, &self.editor)?;
            self.viewports.insert(event.id, viewport);
        }

        for event in self.player_despawn_consumer.consume_all() {
            #[cfg(feature = "editor")]
            self.viewport_editor.remove_viewport(event.id, &self.editor);
            self.viewports.remove(&event.id);
        }

        let mut commands = Vec::new();
        for receiver in &self.receivers {
            commands.append(&mut receiver.collect());
        }
        for command in commands {
            command(self)?;
        }

        let platform_events: Vec<_> = self.platform_consumer.consume_all().collect();
        for event in platform_events {
            match event {
                PlatformEvent::WindowCreated { id } => {
                    let windows = self.platform_windows.windows();
                    let Some(platform_window) = windows.get(&id) else {
                        continue;
                    };
                    let window = Window::new(&self.rhi, &self.presenter, platform_window)?;
                    self.windows.insert(id, window);
                    self.window_order.push(id);
                    sort_window_ids(&mut self.window_order);
                    debug!("created renderer window with id {}", id.into_raw());
                }
                // In case the window was not destroyed by a close request.
                PlatformEvent::WindowDestroyed { id }
                | PlatformEvent::WindowCloseRequested { id } => {
                    self.windows.remove(&id);
                    self.window_order.retain(|window| *window != id);
                }
            }
        }

        for event in self.window_consumer.consume_all() {
            match event {
                WindowEvent::Resized { id, width, height } => {
                    if let Some(window) = self.windows.get_mut(&id) {
                        window.resize(width, height);
                    }
                }
                WindowEvent::Occluded { id, occluded } => {
                    if let Some(window) = self.windows.get_mut(&id) {
                        window.set_occluded(occluded);
                    }
                }
                WindowEvent::FocusChanged { .. } | WindowEvent::ThemeChanged { .. } => {}
            }
        }
        Ok(())
    }

    /// Renders one frame: uploads new assets, then renders the viewports
    /// shown in windows and every visible window in one graph and one
    /// submission, which presents the windows. Submits nothing when no
    /// window is visible, apart from uploads.
    fn render(&mut self) -> Result<()> {
        #[cfg(not(feature = "editor"))]
        self.update_presentation()?;

        let mut frame = self.rhi.begin_frame()?;
        let mut uploads = self.uploader.begin(&mut frame);
        self.models.update(uploads.rhi(), &mut uploads)?;
        #[cfg(feature = "editor")]
        self.egui.update_textures(uploads.rhi(), &mut uploads)?;
        uploads.submit()?;
        #[cfg(feature = "editor")]
        self.viewport_editor
            .apply_resize_requests(frame.rhi(), &mut self.viewports)?;

        let mut targets = Vec::new();
        for window in self.windows.values_mut() {
            targets.extend(window.acquire(&frame)?);
        }

        // The viewports some acquired window shows.
        #[cfg(feature = "editor")]
        let shown: Vec<PlayerId> = if targets
            .iter()
            .any(|target| Some(target.window) == self.egui_window)
        {
            self.viewports.keys().copied().collect()
        } else {
            Vec::new()
        };
        #[cfg(not(feature = "editor"))]
        let shown: Vec<PlayerId> = targets
            .iter()
            .filter_map(|target| {
                self.presentation_assignments
                    .player_for_window(target.window)
            })
            .filter(|player| self.viewports.contains_key(player))
            .collect();

        if !targets.is_empty() {
            let mut graph = RenderGraph::new();
            // The output each shown viewport shows.
            let mut outputs = HashMap::new();
            if !shown.is_empty() {
                self.scene_manager.write_objects(&frame)?;
                for player in &shown {
                    if let Some(viewport) = self.viewports.get(player)
                        && let Some(output) = viewport.record(
                            &mut graph,
                            &frame,
                            &self.scene_manager,
                            &self.models,
                        )?
                    {
                        outputs.insert(*player, output);
                    }
                }
            }

            #[cfg(feature = "editor")]
            let ui = targets
                .iter()
                .find(|target| Some(target.window) == self.egui_window)
                .map(|target| {
                    let extent = target.image.image().info().extent;
                    let extent = (extent.width, extent.height);
                    self.egui
                        .record(&mut graph, &frame, extent, outputs.values().copied())
                })
                .transpose()?;
            for target in &targets {
                let image = target.image.image();
                let image =
                    graph.import_image(image, ImageAccess::Present, ImageAccess::Present)?;
                #[cfg(feature = "editor")]
                let source = ui.filter(|_| Some(target.window) == self.egui_window);
                #[cfg(not(feature = "editor"))]
                let source = self
                    .presentation_assignments
                    .player_for_window(target.window)
                    .and_then(|player| outputs.get(&player).copied());
                self.presenter
                    .record(&mut graph, target.pipeline, image, source)?;
            }

            let mut encoder = frame.encoder::<Graphics>("frame")?;
            graph.execute(&frame, &mut encoder, &mut self.transients)?;
            let images = targets.into_iter().map(|target| target.image);
            for presentable in frame.submit_with(encoder, images)? {
                frame.present(presentable)?;
            }
            for player in outputs.keys() {
                if let Some(viewport) = self.viewports.get_mut(player) {
                    viewport.mark_rendered();
                }
            }
        }

        #[cfg(feature = "editor")]
        self.egui.finish_frame();
        Ok(())
    }

    /// Assigns each window a player's viewport to show, by order, and sizes
    /// each viewport like its window. Resizing outside a frame is safe here:
    /// nothing recorded later reads the old outputs.
    #[cfg(not(feature = "editor"))]
    fn update_presentation(&mut self) -> Result<()> {
        let mut players = self.viewports.keys().copied().collect::<Vec<_>>();
        players.sort_unstable();
        let assignments: Vec<_> = self
            .window_order
            .iter()
            .copied()
            .filter(|window| self.windows.contains_key(window))
            .zip(players)
            .collect();
        self.presentation_assignments.set(assignments.clone());

        for (window, player) in assignments {
            // Minimized windows keep their viewport's size.
            let extent = self.windows.get(&window).map(Window::extent);
            if let (Some(extent @ (1.., 1..)), Some(viewport)) =
                (extent, self.viewports.get_mut(&player))
            {
                viewport.resize(&self.rhi, extent)?;
            }
        }
        Ok(())
    }

    fn bind_viewport_to_entity(&mut self, player: PlayerId, entity: Entity) {
        let world = self.scene_manager.entity_world(entity);
        if let Some(viewport) = self.viewports.get_mut(&player) {
            viewport.camera = Some(entity);
            viewport.world = world;
        }
    }
    fn update_viewport_world_for_camera(&mut self, camera: Entity, world: WorldId) {
        for viewport in self.viewports.values_mut() {
            if viewport.camera == Some(camera) {
                viewport.world = Some(world);
            }
        }
    }
    fn clear_viewports_for_camera(&mut self, camera: Entity) {
        for viewport in self.viewports.values_mut() {
            if viewport.camera == Some(camera) {
                viewport.camera = None;
                viewport.world = None;
            }
        }
    }
}
