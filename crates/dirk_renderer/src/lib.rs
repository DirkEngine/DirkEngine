#![doc = include_str!("../README.md")]

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::Context;
#[cfg(validation)]
use ash::ext::debug_utils;
use ash::{Entry, khr::swapchain, vk};

#[cfg(feature = "editor")]
use dirk_platform::WindowInputEvent;
use dirk_platform::{PlatformEvent, WindowEvent, WindowId};
use dirk_player::PlayerId;
#[cfg(feature = "editor")]
use dirk_player::PlayerInputSender;
#[cfg(not(feature = "editor"))]
use dirk_player::PlayerPresentationAssignments;

use dirk_universe::{Entity, Universe, UniverseBuilder, WorldId};
use dirk_utils::Version;
use raw_window_handle::HasDisplayHandle;
use tracing::{debug, info};

use dirk_platform::PlatformWindows;

mod utils;
use utils::{Frame, RendererProperties, make_version};

mod errors;
#[cfg(feature = "editor")]
pub use egui;
pub use errors::{Error, Result};

#[cfg(feature = "editor")]
mod egui_integration;
#[cfg(feature = "editor")]
use egui_integration::{EguiFrameInput, EguiState};

mod window;
use window::Window;

mod resources;
use resources::{
    command_pool::CommandBuffer,
    device::{FrameCounters, RenderDevice},
    queues::QueueType,
    swapchain::RenderImage,
};

mod proxy;
use proxy::{
    scene::{SceneManager, SceneRenderSettings},
    systems::RendererSystem,
};

mod render_commands;
use render_commands::RenderCommandReceiver;

mod init;
mod models;
mod physical_device;
mod pipeline;
mod shaders;

mod viewport;
use viewport::{Viewport, ViewportSettings};

#[cfg(feature = "editor")]
mod viewport_editor;
#[cfg(feature = "editor")]
use viewport_editor::ViewportEditor;

mod frame_graph;
use frame_graph::{RenderGraph, TextureDesc};

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

            let create_info = RendererCreateInfo::from_engine_metadata(ctx.handle().metadata())?;

            let main_window = platform_windows.main_window();
            let mut renderer = Renderer::init(
                &create_info,
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

            ctx.extend_universe(renderer.universe_builder());
            Ok(renderer)
        });
        Ok(())
    }
}

#[cfg(validation)]
mod debug;

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEVICE_EXTENSIONS: &[&str] = &[
    unsafe { std::str::from_utf8_unchecked(swapchain::NAME.to_bytes()) },
    #[cfg(platform_macos)]
    unsafe {
        std::str::from_utf8_unchecked(ash::khr::portability_subset::NAME.to_bytes())
    },
];
#[cfg(validation)]
const VALIDATION_LAYERS: &[*const i8] = &[c"VK_LAYER_KHRONOS_validation".as_ptr()];

/// The information needed to create the renderer. This is primarily metadata
/// used for Vulkan initialisation.
pub struct RendererCreateInfo {
    /// The name of the engine. Used for vulkan initialisation.
    pub engine_name: CString,
    /// The version of the engine. Used for vulkan initialisation.
    pub engine_version: Version,
    /// The name of the application. Used for vulkan initialisation.
    pub app_name: CString,
    /// The version of the application. Used for vulkan initialisation.
    pub app_version: Version,
}

impl RendererCreateInfo {
    /// Creates renderer metadata from engine metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the metadata strings contains an interior NUL
    /// byte and cannot be passed to Vulkan.
    pub fn from_engine_metadata(metadata: &dirk_engine::EngineMetadata) -> anyhow::Result<Self> {
        Ok(Self {
            engine_name: CString::new(metadata.engine_name())?,
            engine_version: metadata.engine_version(),
            app_name: CString::new(metadata.app_name())?,
            app_version: metadata.app_version(),
        })
    }
}

/// The Renderer struct that holds all render state and is called upon to handle
/// all rendering operations
struct Renderer {
    // Heavy renderer state:
    /// All of the [`window::Window`]s constructed from [`platform::Window`]s.
    windows: HashMap<WindowId, Window>,
    window_order: Vec<WindowId>,
    platform_windows: PlatformWindows,
    #[cfg(not(feature = "editor"))]
    presentation_assignments: PlayerPresentationAssignments,
    /// All of the internal [`world::World`] representations.
    scene_manager: SceneManager,
    /// The management for all the models.
    models: models::ModelRegistry,
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

    frames: [Frame; MAX_FRAMES_IN_FLIGHT],
    current_frame: Arc<AtomicUsize>,
    frame_count: Arc<AtomicUsize>,

    // Events
    window_consumer: dirk_events::Consumer<dirk_platform::WindowEvent>,
    platform_consumer: dirk_events::Consumer<dirk_platform::PlatformEvent>,
    player_spawn_consumer: dirk_events::Consumer<dirk_player::PlayerSpawned>,
    player_despawn_consumer: dirk_events::Consumer<dirk_player::PlayerDespawned>,

    /// These receive all the commands from the game thread.
    receivers: Vec<RenderCommandReceiver>,

    // last as should be dropped last
    render_device: RenderDevice,
}

struct PresentationTarget {
    window: WindowId,
    extent: vk::Extent2D,
    image: RenderImage,
}

struct ViewportRenderSubmission {
    command_buffer: CommandBuffer,
    rendered_players: Vec<PlayerId>,
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
        self.tick(delta_time)?;
        #[cfg(feature = "editor")]
        {
            let ctx = self.begin_frame();
            let frame = dirk_engine::editor::EditorRenderContext::new(delta_time, handle);

            self.viewport_editor.sync_ready_state(&self.viewports);
            self.editor.render_ui(&ctx, &frame, universe)?;
        }

        #[cfg(not(feature = "editor"))]
        {
            let _ = (handle, universe);
        }

        self.end_frame().context("rendering")?;

        Ok(())
    }
}

impl Renderer {
    /// Renderer initialisation. Creates all Vulkan & other renderer objects.
    ///
    /// # Errors
    ///
    /// Plenty of Vulkan & platform errors can occur during renderer intializing
    pub fn init(
        create_info: &RendererCreateInfo,
        window: &dirk_platform::Window,
        event_manager: &dirk_events::EventManager,
        platform_windows: PlatformWindows,
        #[cfg(not(feature = "editor"))] presentation_assignments: PlayerPresentationAssignments,
        #[cfg(feature = "editor")] player_input_sender: PlayerInputSender,
        #[cfg(feature = "editor")] editor: dirk_engine::editor::EditorServices,
    ) -> Result<Self> {
        info!("Intializing Vulkan...");

        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(create_info.app_name.as_c_str())
            .application_version(make_version(create_info.app_version))
            .engine_name(create_info.engine_name.as_c_str())
            .engine_version(make_version(create_info.engine_version))
            .api_version(vk::API_VERSION_1_3);

        let mut extensions = Self::required_instance_extensions(window)?;
        let mut instance_create_info =
            vk::InstanceCreateInfo::default().application_info(&app_info);

        #[cfg(platform_macos)]
        {
            instance_create_info =
                instance_create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        #[cfg(validation)]
        let mut debug_create_info = debug::debug_create_info();

        #[cfg(validation)]
        {
            info!(target: "vulkan::validation", "using validation layers");
            extensions.push(debug_utils::NAME.as_ptr());
            debug::validate_instance_layers(&entry, VALIDATION_LAYERS)?;

            instance_create_info = instance_create_info
                .enabled_layer_names(VALIDATION_LAYERS)
                .push_next(&mut debug_create_info);
        }

        Self::validate_instance_extensions(&entry, &extensions)?;
        instance_create_info = instance_create_info.enabled_extension_names(&extensions);

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        #[cfg(validation)]
        let debug_messenger = debug::create_debug_messenger(&entry, &instance, &debug_create_info)?;

        let (physical_device, properties) =
            Self::select_physical_device(&entry, &instance, window)?;
        let device = Self::create_device(&instance, physical_device, &properties)?;

        let current_frame = Arc::new(AtomicUsize::new(0));
        let frame_count = Arc::new(AtomicUsize::new(0));

        let render_device = RenderDevice::new(
            entry.clone(),
            instance.clone(),
            device.clone(),
            physical_device,
            properties,
            FrameCounters {
                current_frame: current_frame.clone(),
                frame_count: frame_count.clone(),
            },
            #[cfg(validation)]
            debug_messenger,
        )?;

        let frames = Self::build_frames(&device, &render_device)?;

        let models = models::ModelRegistry::new(&render_device, event_manager)?;
        let scene_manager = SceneManager::init(&render_device)?;
        #[cfg(feature = "editor")]
        let egui = EguiState::new(&render_device)?;
        #[cfg(feature = "editor")]
        let viewport_editor = ViewportEditor::new(&render_device, player_input_sender)?;

        let windows = {
            let window = Window::build(&render_device, window)?;
            let mut windows = HashMap::new();
            windows.insert(window.id(), window);
            windows
        };
        let mut window_order = windows.keys().copied().collect::<Vec<_>>();
        sort_window_ids(&mut window_order);

        Ok(Self {
            render_device,

            windows,
            window_order,
            platform_windows,
            #[cfg(not(feature = "editor"))]
            presentation_assignments,
            scene_manager,
            viewports: HashMap::new(),
            models,
            #[cfg(feature = "editor")]
            egui,
            #[cfg(feature = "editor")]
            egui_window: None,
            #[cfg(feature = "editor")]
            editor,
            #[cfg(feature = "editor")]
            viewport_editor,
            #[cfg(feature = "editor")]
            egui_input_consumer: event_manager.subscribe(),

            frames,
            current_frame,
            frame_count,

            window_consumer: event_manager.subscribe(),
            platform_consumer: event_manager.subscribe(),
            player_spawn_consumer: event_manager.subscribe(),
            player_despawn_consumer: event_manager.subscribe(),
            receivers: Vec::new(),
        })
    }

    /// Begins a frame.
    ///
    /// Returns an [`egui::Context`] for rendering.
    #[cfg(feature = "editor")]
    pub fn begin_frame(&mut self) -> egui::Context {
        let input = self.egui_frame_input();
        self.egui_window = Some(input.window_id);
        self.egui.begin_frame(&input)
    }

    // TODO: shouldn't be necessary
    #[cfg(feature = "editor")]
    fn primary_extent(&self) -> vk::Extent2D {
        self.primary_window_id()
            .and_then(|id| self.windows.get(&id))
            .map_or(
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                Window::extent,
            )
    }

    #[cfg(feature = "editor")]
    fn primary_window_id(&self) -> Option<WindowId> {
        self.window_order.first().copied()
    }

    #[cfg(feature = "editor")]
    #[allow(clippy::cast_possible_truncation)]
    fn egui_frame_input(&mut self) -> EguiFrameInput {
        let window_id = self
            .primary_window_id()
            .unwrap_or_else(|| WindowId::from_raw(0));
        let extent = self.primary_extent();
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

    /// Returns a [`UniverseBuilder`] that is populated with [`Renderer`] systems.
    fn universe_builder(&mut self) -> UniverseBuilder {
        let (sender, receiver) = render_commands::channel();
        self.receivers.push(receiver);
        Universe::builder().with_system(RendererSystem::new(sender))
    }

    /// Ticks the renderer. Used to improve the various internal representations
    /// of external engine objects.
    /// The renderer listens to events to properly sync windows, scenes, ...
    ///
    /// # Errors
    ///
    /// Errors can occur when updating the scene & world (if one is missing for example)
    /// Some platform errors can also occur when handling windows
    ///
    /// # Panics
    ///
    /// Will panic if the scene object does not exist for the specified
    /// world (unless in [`WorldEvent::Created`] or [`WorldEvent::Destroyed`].
    fn tick(&mut self, _delta_time: f64) -> Result<()> {
        for event in self.player_spawn_consumer.consume_all() {
            #[cfg(feature = "editor")]
            self.viewport_editor.remove_viewport(event.id, &self.editor);

            let viewport = Viewport::new(
                &self.render_device,
                event.id,
                ViewportSettings::new(
                    vk::Extent2D {
                        width: 1,
                        height: 1,
                    },
                    self.render_device.properties.surface_format.format,
                ),
            )?;
            #[cfg(feature = "editor")]
            self.viewport_editor
                .add_viewport(event.id, &viewport, &self.editor, &mut self.egui)?;
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
                    let Some(plat_window) = windows.get(&id) else {
                        continue;
                    };

                    let window = window::Window::build(&self.render_device, plat_window)?;
                    self.windows.insert(window.id(), window);
                    self.window_order.push(id);
                    sort_window_ids(&mut self.window_order);

                    debug!("created renderer window with id {}", id.into_raw());
                }
                PlatformEvent::WindowDestroyed { id } => {
                    // in case the window was not destroyed by WindowCloseRequested
                    self.windows.remove(&id);
                    self.window_order.retain(|window| *window != id);
                }
                PlatformEvent::WindowCloseRequested { id } => {
                    self.windows.remove(&id);
                    self.window_order.retain(|window| *window != id);
                }
            }
        }

        let window_events: Vec<_> = self.window_consumer.consume_all().collect();
        for event in window_events {
            match event {
                WindowEvent::Resized { id, width, height } => {
                    let Some(window) = self.windows.get_mut(&id) else {
                        continue;
                    };
                    window.resize(vk::Extent2D { width, height })?;
                }
                WindowEvent::Occluded { id, occluded } => {
                    let Some(window) = self.windows.get_mut(&id) else {
                        continue;
                    };
                    window.set_occluded(occluded);
                }
                // don't care about these
                WindowEvent::FocusChanged { .. } | WindowEvent::ThemeChanged { .. } => {}
            }
        }

        self.models.tick()?;

        Ok(())
    }

    /// The actual rendering. This records render commands & submits them to
    /// the GPU.
    ///
    /// # Errors
    ///
    /// Vulkan errors can occur during rendering
    fn end_frame(&mut self) -> Result<()> {
        #[cfg(feature = "editor")]
        self.egui.end_frame();

        let frame_index = self.current_frame();
        self.render_device.flush_deletions();
        #[cfg(feature = "editor")]
        {
            self.egui.free_textures_for_frame(frame_index)?;
            self.viewport_editor
                .release_retired_textures(&mut self.egui);
            self.viewport_editor
                .apply_resize_requests(&mut self.viewports, &mut self.egui)?;
        }
        #[cfg(not(feature = "editor"))]
        self.update_non_editor_presentation()?;

        self.frames[frame_index].fence.wait(u64::MAX)?;
        self.frames[frame_index].submitted_command_buffers.clear();

        let viewport_submission = self.record_viewport_graph(frame_index)?;
        let presentation_targets = self.acquire_presentation_targets()?;
        let presentation_cmd = self.record_presentation_graph(
            frame_index,
            &presentation_targets,
            viewport_submission.as_ref(),
        )?;

        self.frames[frame_index].fence.reset()?;
        self.submit_frame(
            frame_index,
            viewport_submission.as_ref(),
            presentation_cmd.as_ref(),
            &presentation_targets,
        )?;

        if let Some(submission) = viewport_submission {
            for player in submission.rendered_players {
                if let Some(viewport) = self.viewports.get_mut(&player) {
                    viewport.mark_render_submitted(viewport.next_render_value());
                }
            }
            self.frames[frame_index]
                .submitted_command_buffers
                .push(submission.command_buffer);
        }
        if let Some(cmd) = presentation_cmd {
            self.frames[frame_index].submitted_command_buffers.push(cmd);
        }

        for target in presentation_targets {
            target.image.present()?;
        }

        self.current_frame.store(
            (self.current_frame() + 1) % MAX_FRAMES_IN_FLIGHT,
            Ordering::Relaxed,
        );
        self.frame_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn record_viewport_graph(
        &self,
        frame_index: usize,
    ) -> Result<Option<ViewportRenderSubmission>> {
        if self.viewports.is_empty() {
            return Ok(None);
        }

        let mut graph = RenderGraph::new();
        let mut rendered_players = Vec::new();
        for viewport in self.viewports.values() {
            if !viewport.is_renderable() {
                continue;
            }
            let Some(world) = viewport.world else {
                continue;
            };
            let Some(camera) = viewport.camera else {
                continue;
            };
            rendered_players.push(viewport.player());

            let output = graph.import_texture(TextureDesc {
                width: viewport.settings().extent.width,
                height: viewport.settings().extent.height,
                format: viewport.settings().format,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                samples: vk::SampleCountFlags::TYPE_1,
                imported: Some(viewport.import()),
            });
            self.scene_manager.render(
                &mut graph,
                &self.models,
                world,
                camera,
                SceneRenderSettings {
                    extent: viewport.settings().extent,
                    format: viewport.settings().format,
                    clear_color: viewport.settings().clear_color,
                    fov_y_radians: viewport.settings().fov_y_radians,
                    near: viewport.settings().near,
                    far: viewport.settings().far,
                },
                output,
            );
        }

        if rendered_players.is_empty() {
            return Ok(None);
        }

        let cmd = self.frames[frame_index].command_pool.allocate_buffer()?;
        cmd.begin_command_buffer(&vk::CommandBufferBeginInfo::default())?;
        graph.run(&self.render_device, &cmd)?;
        cmd.end_command_buffer()?;

        Ok(Some(ViewportRenderSubmission {
            command_buffer: cmd,
            rendered_players,
        }))
    }

    fn acquire_presentation_targets(&mut self) -> Result<Vec<PresentationTarget>> {
        let window_ids = self.windows.keys().copied().collect::<Vec<_>>();
        let mut targets = Vec::new();

        for window_id in window_ids {
            let window = self
                .windows
                .get_mut(&window_id)
                .expect("window keys should come from the window map");
            targets.push(PresentationTarget {
                window: window_id,
                extent: window.extent(),
                image: window.next_image()?,
            });
        }

        Ok(targets)
    }

    fn record_presentation_graph(
        &mut self,
        frame_index: usize,
        targets: &[PresentationTarget],
        #[cfg_attr(feature = "editor", allow(unused_variables))] viewport_submission: Option<
            &ViewportRenderSubmission,
        >,
    ) -> Result<Option<resources::command_pool::CommandBuffer>> {
        if targets.is_empty() {
            return Ok(None);
        }

        let mut graph = RenderGraph::new();
        #[cfg(feature = "editor")]
        let mut egui_target = None;
        for target in targets {
            #[cfg(feature = "editor")]
            let swapchain_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
            #[cfg(not(feature = "editor"))]
            let swapchain_usage =
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;

            let swapchain = graph.import_texture(TextureDesc {
                width: target.extent.width,
                height: target.extent.height,
                format: self.render_device.properties.surface_format.format,
                usage: swapchain_usage,
                samples: vk::SampleCountFlags::TYPE_1,
                imported: Some(target.image.import()),
            });

            graph.add_pass("clear swapchain").write_color_attachment(
                swapchain,
                frame_graph::AttachmentInfo::clear_color(0.0, 0.0, 0.0, 1.0),
            );

            #[cfg(feature = "editor")]
            if Some(target.window) == self.egui_window {
                egui_target = Some((swapchain, target.extent));
            }

            #[cfg(not(feature = "editor"))]
            if let Some(viewport) = self.assigned_viewport_for_window(target.window) {
                let rendered_this_frame = viewport_submission.is_some_and(|submission| {
                    submission.rendered_players.contains(&viewport.player())
                });
                if viewport.has_rendered() || rendered_this_frame {
                    let viewport_extent = viewport.settings().extent;
                    let target_extent = target.extent;
                    let viewport_source = graph.import_texture(TextureDesc {
                        width: viewport_extent.width,
                        height: viewport_extent.height,
                        format: viewport.settings().format,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                        samples: vk::SampleCountFlags::TYPE_1,
                        imported: Some(if rendered_this_frame {
                            viewport.import_after_render()
                        } else {
                            viewport.import()
                        }),
                    });

                    let mut copy_pass = graph.add_pass("copy scene to swapchain");
                    copy_pass
                        .read_transfer_src(viewport_source)
                        .write_transfer_dst(swapchain);
                    copy_pass.execute(Box::new(move |_, cmd, images| {
                        let region = vk::ImageCopy::default()
                            .src_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .dst_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: 0,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .extent(vk::Extent3D {
                                width: target_extent.width,
                                height: target_extent.height,
                                depth: 1,
                            });

                        cmd.copy_image(
                            images[viewport_source.index()].image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            images[swapchain.index()].image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[region],
                        );
                        Ok(())
                    }));
                }
            }
        }

        #[cfg(feature = "editor")]
        if let Some((swapchain, extent)) = egui_target {
            let mut egui_pass = graph.add_pass("egui");
            egui_pass.write_color_attachment(swapchain, frame_graph::AttachmentInfo::load_store());
            let egui = &mut self.egui;
            egui_pass.execute(Box::new(move |device, cmd, _| {
                egui.render(device, cmd, extent, frame_index)
            }));
        }

        let cmd = self.frames[frame_index].command_pool.allocate_buffer()?;
        cmd.begin_command_buffer(&vk::CommandBufferBeginInfo::default())?;
        graph.run(&self.render_device, &cmd)?;
        cmd.end_command_buffer()?;

        Ok(Some(cmd))
    }

    #[cfg(not(feature = "editor"))]
    fn update_non_editor_presentation(&mut self) -> Result<()> {
        let assignments = self.current_non_editor_assignments();
        self.presentation_assignments.set(assignments.clone());

        for (window_id, player) in assignments {
            let Some(window) = self.windows.get(&window_id) else {
                continue;
            };
            let extent = window.extent();
            if let Some(viewport) = self.viewports.get_mut(&player) {
                viewport.resize(&self.render_device, extent)?;
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "editor"))]
    fn current_non_editor_assignments(&self) -> Vec<(WindowId, PlayerId)> {
        let mut players = self.viewports.keys().copied().collect::<Vec<_>>();
        players.sort_unstable();
        self.window_order
            .iter()
            .copied()
            .filter(|window| self.windows.contains_key(window))
            .zip(players)
            .collect()
    }

    #[cfg(not(feature = "editor"))]
    fn assigned_viewport_for_window(&self, window: WindowId) -> Option<&Viewport> {
        self.presentation_assignments
            .player_for_window(window)
            .and_then(|player| self.viewports.get(&player))
    }

    fn submit_frame(
        &self,
        frame_index: usize,
        viewport_submission: Option<&ViewportRenderSubmission>,
        presentation_cmd: Option<&resources::command_pool::CommandBuffer>,
        presentation_targets: &[PresentationTarget],
    ) -> Result<()> {
        let mut submits = Vec::new();

        // VIEWPORTS

        // all the viewports that were rendered too this frame
        let rendered_viewports = viewport_submission.map_or_else(Vec::new, |submission| {
            submission
                .rendered_players
                .iter()
                .filter_map(|player| self.viewports.get(player))
                .collect::<Vec<_>>()
        });

        // timeline semaphores to signal for viewports
        let timeline_signal_semaphores = rendered_viewports
            .iter()
            .map(|viewport| viewport.render_semaphore())
            .collect::<Vec<_>>();
        // values to signal said semaphores too
        let timeline_signal_values = rendered_viewports
            .iter()
            .map(|viewport| viewport.next_render_value())
            .collect::<Vec<_>>();
        // submit info for the viewport timeline semaphores
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .signal_semaphore_values(&timeline_signal_values);

        // actually create the submit info for the viewport rendering
        if let Some(submission) = viewport_submission {
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&submission.command_buffer))
                .signal_semaphores(&timeline_signal_semaphores);
            if !timeline_signal_semaphores.is_empty() {
                submit = submit.push_next(&mut timeline_info);
            }
            submits.push(submit);
        }

        // WINDOWS

        let image_available_semaphores = presentation_targets
            .iter()
            .map(|target| target.image.image_available_semaphore)
            .collect::<Vec<_>>();
        let render_finished_semaphores = presentation_targets
            .iter()
            .map(|target| target.image.render_finished_semaphore)
            .collect::<Vec<_>>();

        // wait on image available semaphores & viewport timeline semaphores
        let mut wait_semaphores = image_available_semaphores.clone();
        wait_semaphores.extend(timeline_signal_semaphores.iter().copied());

        // wait values: 0 for image vailable, timeline signal values otherwise
        let mut wait_values = vec![0; image_available_semaphores.len()];
        wait_values.extend(timeline_signal_values.iter().copied());

        // submit info for the window timeline semaphores
        let mut presentation_timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);

        // all presentation targets wait on COLOR_ATTACHMENT_OUTPUT
        let mut wait_stages = presentation_targets
            .iter()
            .map(|_| vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .collect::<Vec<_>>();

        // all viewports wait on FRAGMENT_SHADER or TRANSFER when not in editor
        #[cfg(feature = "editor")]
        let viewport_wait_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        #[cfg(not(feature = "editor"))]
        let viewport_wait_stage = vk::PipelineStageFlags::TRANSFER;
        wait_stages.extend(
            timeline_signal_semaphores
                .iter()
                .map(|_| viewport_wait_stage),
        );

        // actually create the submit info for window rendering
        if let Some(cmd) = presentation_cmd {
            let mut submit = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(cmd))
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&render_finished_semaphores);
            if !timeline_signal_semaphores.is_empty() {
                submit = submit.push_next(&mut presentation_timeline_info);
            }
            submits.push(submit);
        }

        if submits.is_empty() {
            submits.push(vk::SubmitInfo::default());
        }

        self.render_device.queues.submit(
            QueueType::Graphics,
            &submits,
            Some(&self.frames[frame_index].fence),
        )?;

        Ok(())
    }

    #[inline]
    fn current_frame(&self) -> usize {
        self.current_frame
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    // EXTRA UTILS

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
    fn create_sampler(device: &RenderDevice, mip_levels: u32) -> Result<vk::Sampler> {
        let props = unsafe {
            device
                .instance
                .get_physical_device_properties(device.physical_device)
        };
        let max_aniso = props.limits.max_sampler_anisotropy;

        // the max_lod cast loses precision, as there are only a
        // small number of mip_levels, there should be no real
        // precision loss.
        #[allow(clippy::cast_precision_loss)]
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0)
            .anisotropy_enable(true) // TODO: use the detected prpoerty
            .max_anisotropy(max_aniso) // use hardware maximum
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false);

        Ok(unsafe { device.device.create_sampler(&sampler_info, None)? })
    }

    fn required_instance_extensions(window: &dirk_platform::Window) -> Result<Vec<*const i8>> {
        let surface_target = window.surface_target();
        let display_handle = surface_target.display_handle()?.as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();

        #[cfg(platform_macos)]
        let extensions = {
            let mut extensions = extensions;
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions
        };

        Ok(extensions)
    }

    fn validate_instance_extensions(entry: &Entry, extensions: &[*const i8]) -> Result<()> {
        let available = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .unwrap_or_default()
        };

        for &required in extensions {
            let required = unsafe { CStr::from_ptr(required) };
            let found = available
                .iter()
                .any(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) } == required);

            if !found {
                return Err(Error::ExtensionNotFound(
                    required.to_string_lossy().into_owned(),
                ));
            }
        }

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.render_device.device.device_wait_idle().ok();
        }
        info!("cleaning up renderer");
    }
}
