//! Core front-facing engine API.
//!
//! This crate owns the engine runtime primitives and composition API. Optional
//! engine features such as rendering, assets, players, and editor tooling are
//! registered through [`EnginePlugin`] implementations instead of being
//! hard-wired into [`Engine`].
//!
//! # Composition model
//!
//! `DirkEngine` has three deliberately separate extension concepts:
//!
//! - **Plugins** are build-time extension points. A plugin receives an
//!   [`EngineBuilder`] and registers the pieces needed by one engine feature.
//!   Plugins are never started or ticked by the runtime.
//! - **Subsystems** are runtime lifecycle objects. They own mutable feature
//!   state and are started, ticked, and shut down by the engine loop.
//! - **Resources** are cloneable handles published while subsystems are being
//!   built. They let later subsystem factories discover capabilities created by
//!   earlier subsystem factories without transferring runtime ownership out of
//!   the subsystem that drives the behavior.
//!
//! Plugin registration is idempotent by concrete plugin type. This lets plugins
//! declare their dependencies directly:
//!
//! ```rust
//! # use dirk_engine::{EngineBuilder, EnginePlugin};
//! # struct AssetsPlugin;
//! # impl EnginePlugin for AssetsPlugin {
//! #     fn name(&self) -> &'static str { "assets" }
//! #     fn build(&self, _builder: &mut EngineBuilder) -> anyhow::Result<()> { Ok(()) }
//! # }
//! # struct RendererPlugin;
//! impl EnginePlugin for RendererPlugin {
//!     fn name(&self) -> &'static str {
//!         "renderer"
//!     }
//!
//!     fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
//!         builder.with_plugin(AssetsPlugin)?;
//!         Ok(())
//!     }
//! }
//! ```
//!
//! If multiple plugins request the same dependency type, only the first
//! successful registration runs that dependency plugin's `build` method. The
//! builder stores explicit order lists for plugins and subsystems, so runtime
//! behavior follows first successful registration rather than hash-map order.
//!
//! Resources are intended to be small handles. They should be cheap to clone and
//! should not become the primary owner of mutable runtime behavior. If a feature
//! needs mutable state, keep that state in the subsystem and publish a small
//! synchronized handle only when another subsystem factory needs to find it
//! during engine construction.
//!
//! Resource availability is order-dependent: a resource published by one
//! subsystem factory is immediately available to later subsystem factories, but
//! earlier factories cannot see resources that have not been published yet.
//! Looking up a missing resource returns [`Error::ResourceMissing`], and
//! registering a second resource of the same concrete type returns
//! [`Error::ResourceAlreadyRegistered`].

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, Sender},
    },
    time::Instant,
};

use dirk_events::EventManager;
use dirk_threads::WorkerPool;
use dirk_universe::Universe;
use parking_lot::RwLock;
use tracing::error;

#[cfg(feature = "editor")]
pub mod editor;
pub mod errors;
pub mod events;
pub mod subsystem;

mod builder;
mod signal;
pub use builder::{EngineBuildContext, EngineBuilder};

use errors::{Error, Result};
pub use subsystem::{EnginePlugin, EngineResource, Subsystem};

mod tests;

/// Immutable metadata describing this engine instance.
///
/// The engine constructs this once while building [`EngineHandle`]. Subsystems
/// should treat it as read-only configuration for integration points such as
/// renderer or platform initialisation.
#[derive(Debug)]
pub struct EngineMetadata {
    app_name: String,
    app_version: dirk_utils::Version,
    engine_name: String,
    engine_version: dirk_utils::Version,
}

impl EngineMetadata {
    /// Creates engine metadata.
    #[must_use]
    pub fn new(
        app_name: impl Into<String>,
        app_version: dirk_utils::Version,
        engine_name: impl Into<String>,
        engine_version: dirk_utils::Version,
    ) -> Self {
        Self {
            app_name: app_name.into(),
            app_version,
            engine_name: engine_name.into(),
            engine_version,
        }
    }

    /// Returns the application name.
    #[must_use]
    pub fn app_name(&self) -> &str {
        &self.app_name
    }

    /// Returns the application version.
    #[must_use]
    pub fn app_version(&self) -> dirk_utils::Version {
        self.app_version
    }

    /// Returns the engine name.
    #[must_use]
    pub fn engine_name(&self) -> &str {
        &self.engine_name
    }

    /// Returns the engine version.
    #[must_use]
    pub fn engine_version(&self) -> dirk_utils::Version {
        self.engine_version
    }
}

type ResourceStorage = Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>;

/// The main engine object.
///
/// Create an engine with [`Engine::builder`] to register plugins and
/// subsystems before the runtime starts.
pub struct Engine {
    logger: piquel_log::Logger,
    universe: Universe,
    subsystems: Vec<Box<dyn Subsystem>>,
    #[cfg(feature = "editor")]
    editor: editor::EditorRuntime,
    state: Arc<EngineState>,
    handle: EngineHandle,
    command_receiver: Receiver<EngineCommand>,
    signals: signal::OperatingSystemSignals,
    frame_dispatcher: dirk_events::Dispatcher<events::BeginFrame>,
    exiting_dispatcher: dirk_events::Dispatcher<events::Exiting>,
    last_tick: Instant,
    lifecycle: EngineLifecycle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineLifecycle {
    Ready,
    Starting,
    Running,
    Failed,
    Shutdown,
}

impl Engine {
    /// Returns a new engine builder.
    #[must_use]
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// Creates a new engine with default builder configuration.
    ///
    /// Operating-system signal handling is disabled by default. Applications
    /// that own process-wide signal policy can enable it with
    /// [`EngineBuilder::with_os_signals`].
    ///
    /// # Errors
    ///
    /// Returns an error if the engine cannot be built.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// Starts all subsystems.
    ///
    /// This is called automatically by [`Engine::run`] and [`Engine::tick`].
    ///
    /// # Errors
    ///
    /// Returns an error if a subsystem fails to start.
    pub fn start(&mut self) -> Result<()> {
        match self.lifecycle {
            EngineLifecycle::Running => return Ok(()),
            EngineLifecycle::Ready => {}
            EngineLifecycle::Starting | EngineLifecycle::Failed | EngineLifecycle::Shutdown => {
                return Err(Error::InvalidLifecycleState {
                    operation: "start",
                    status: self.status(),
                });
            }
        }

        self.lifecycle = EngineLifecycle::Starting;
        self.state.set_status(EngineStatus::Starting);

        let result = (|| {
            for subsystem in &mut self.subsystems {
                subsystem
                    .start(&self.handle)
                    .map_err(|source| Error::SubsystemFailedStart {
                        name: subsystem.name(),
                        source,
                    })?;
            }

            #[cfg(feature = "editor")]
            self.editor.start(&self.handle)?;
            Ok(())
        })();

        if result.is_err() {
            self.lifecycle = EngineLifecycle::Failed;
            self.state.set_status(EngineStatus::Error);
            return result;
        }

        self.last_tick = Instant::now();
        self.lifecycle = EngineLifecycle::Running;
        self.state.set_status(EngineStatus::Running);
        Ok(())
    }

    /// Runs the engine until exit is requested or an error occurs.
    ///
    /// # Errors
    ///
    /// Returns an error when startup, ticking, or shutdown fails.
    pub fn run(mut self) -> Result<()> {
        self.start().map_err(|err| Error::StartFailed(err.into()))?;

        while !self.is_exiting() {
            self.tick().map_err(|err| Error::TickFailed(err.into()))?;
        }

        let shutdown_result = self.shutdown();
        if let Some(error) = self.take_error() {
            let source = match shutdown_result {
                Ok(()) => error,
                Err(shutdown_error) => {
                    error.context(format!("engine shutdown also failed: {shutdown_error:#}"))
                }
            };
            return Err(Error::RuntimeFailed { source });
        }
        shutdown_result.map_err(|err| Error::ShutdownFailed(err.into()))
    }

    /// Starts the engine if needed, then advances it by one tick.
    ///
    /// # Errors
    ///
    /// Returns an error if a subsystem fails to start or tick.
    pub fn tick(&mut self) -> Result<EngineStatus> {
        if self.lifecycle == EngineLifecycle::Ready {
            self.start()?;
        }
        if matches!(
            self.lifecycle,
            EngineLifecycle::Failed | EngineLifecycle::Shutdown
        ) {
            return Err(Error::InvalidLifecycleState {
                operation: "tick",
                status: self.status(),
            });
        }

        self.process_commands();
        if self.is_exiting() {
            return Ok(self.status());
        }

        self.process_operating_system_signals();
        if self.is_exiting() {
            return Ok(self.status());
        }

        let frame = self.state.increment_frame();
        self.frame_dispatcher.dispatch(events::BeginFrame(frame));

        let delta_time = self.capture_delta_time();

        self.universe.tick(delta_time);

        for index in 0..self.subsystems.len() {
            let subsystem = &mut self.subsystems[index];
            let name = subsystem.name();
            subsystem
                .tick(delta_time, &self.handle, &self.universe)
                .map_err(|source| Error::SubsystemFailedTick { name, source })?;

            self.process_commands();
            if self.is_exiting() {
                break;
            }
        }

        #[cfg(feature = "editor")]
        self.editor.tick(delta_time, &self.handle)?;

        self.process_commands();
        Ok(self.status())
    }

    /// Returns whether the engine is exiting or has exited.
    #[must_use]
    fn is_exiting(&self) -> bool {
        self.status().is_exiting()
    }

    /// Returns the current lightweight engine status.
    #[must_use]
    fn status(&self) -> EngineStatus {
        self.state.status()
    }

    /// Shuts down the engine subsystems.
    ///
    /// Shutdown happens in reverse subsystem registration order.
    ///
    /// # Errors
    ///
    /// Returns an error if a subsystem fails to shut down.
    pub fn shutdown(&mut self) -> Result<()> {
        if self.lifecycle == EngineLifecycle::Shutdown {
            return Ok(());
        }

        if !self.is_exiting() {
            self.request_exit(None);
        }

        self.lifecycle = EngineLifecycle::Shutdown;

        let mut first_error = None;

        #[cfg(feature = "editor")]
        if let Err(err) = self.editor.shutdown(&self.handle) {
            first_error = Some(err);
        }

        while let Some(mut subsystem) = self.subsystems.pop() {
            if let Err(source) = subsystem.shutdown(&self.handle) {
                let err = Error::SubsystemFailedShutdown {
                    name: subsystem.name(),
                    source,
                };
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    error!("additional engine shutdown failure: {err:#}");
                }
            }
        }

        self.process_commands();
        self.signals.shutdown();
        self.state.set_status(EngineStatus::Exited);
        first_error.map_or(Ok(()), Err)
    }

    /// Takes a terminal error reported with [`EngineHandle::exit_with_error`].
    ///
    /// Call this after manually driving the engine with [`Engine::tick`] to
    /// inspect the original failure. [`Engine::run`] returns it automatically.
    pub fn take_error(&mut self) -> Option<anyhow::Error> {
        self.state.error.write().take()
    }

    fn request_exit(&mut self, error: Option<anyhow::Error>) {
        let was_exiting = self.is_exiting();

        match error {
            Some(error) => self.state.set_error(error),
            None if was_exiting => return,
            None => self.state.set_status(EngineStatus::ExitRequested),
        }

        if !was_exiting {
            self.exiting_dispatcher.dispatch(events::Exiting);
        }
    }

    fn process_commands(&mut self) {
        while let Ok(command) = self.command_receiver.try_recv() {
            match command {
                EngineCommand::RequestExit(error) => self.request_exit(error),
            }
        }
    }

    fn process_operating_system_signals(&mut self) {
        if self.signals.exit_requested() {
            self.request_exit(None);
        }
    }

    fn capture_delta_time(&mut self) -> f64 {
        let current_time = Instant::now();
        let delta_time = current_time.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = current_time;
        delta_time
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.lifecycle != EngineLifecycle::Shutdown
            && let Err(err) = self.shutdown()
        {
            error!("engine shutdown failed: {err:#}");
        }

        let _ = &self.logger;
    }
}

/// A lightweight copyable engine lifecycle status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineStatus {
    /// The engine has been built but has not started subsystems.
    Initializing,
    /// The engine is starting subsystems.
    Starting,
    /// The engine is in its normal tick loop.
    Running,
    /// A system has requested that the engine exit.
    ExitRequested,
    /// An error occurred in the engine.
    Error,
    /// The engine has shut down.
    Exited,
}

impl EngineStatus {
    /// Returns whether this status means the engine should stop ticking.
    #[must_use]
    pub fn is_exiting(self) -> bool {
        matches!(self, Self::ExitRequested | Self::Error | Self::Exited)
    }
}

/// A cheap handle for systems to communicate with the main engine object.
#[derive(Clone)]
pub struct EngineHandle {
    metadata: Arc<EngineMetadata>,
    state: Arc<EngineState>,
    events: EventManager,
    workers: WorkerPool,
    commands: Sender<EngineCommand>,
    resources: ResourceStorage,
}

impl EngineHandle {
    /// Returns immutable engine metadata.
    #[must_use]
    pub fn metadata(&self) -> &EngineMetadata {
        &self.metadata
    }

    /// Returns the shared event manager.
    #[must_use]
    pub fn events(&self) -> &EventManager {
        &self.events
    }

    /// Returns the worker pool.
    #[must_use]
    pub fn workers(&self) -> &WorkerPool {
        &self.workers
    }

    /// Requests engine exit without an error.
    pub fn exit(&self) {
        self.request_exit(None);
    }

    /// Requests engine exit with an error.
    pub fn exit_with_error(&self, error: anyhow::Error) {
        self.request_exit(Some(error));
    }

    /// Returns the current frame number.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.state.frame()
    }

    /// Returns the current lightweight engine status.
    #[must_use]
    pub fn status(&self) -> EngineStatus {
        self.state.status()
    }

    fn request_exit(&self, error: Option<anyhow::Error>) {
        if let Err(err) = self.commands.send(EngineCommand::RequestExit(error)) {
            error!("engine exit request failed: {err}");
        }
    }
}

enum EngineCommand {
    RequestExit(Option<anyhow::Error>),
}

struct EngineState {
    frame: AtomicU64,
    status: RwLock<EngineStatus>,
    error: RwLock<Option<anyhow::Error>>,
}

impl EngineState {
    fn new() -> Self {
        Self {
            frame: AtomicU64::new(0),
            status: RwLock::new(EngineStatus::Initializing),
            error: RwLock::new(None),
        }
    }

    fn frame(&self) -> u64 {
        self.frame.load(Ordering::Relaxed)
    }

    fn increment_frame(&self) -> u64 {
        self.frame.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn status(&self) -> EngineStatus {
        *self.status.read()
    }

    fn set_error(&self, err: anyhow::Error) {
        let mut reported_error = self.error.write();
        if reported_error.is_none() {
            *reported_error = Some(err);
        } else {
            error!("additional engine runtime failure: {err:#}");
        }
        *self.status.write() = EngineStatus::Error;
    }

    fn set_status(&self, status: EngineStatus) {
        *self.status.write() = status;
    }
}
