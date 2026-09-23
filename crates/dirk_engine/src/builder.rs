use std::{
    any::{TypeId, type_name},
    collections::{HashMap, HashSet},
    path::PathBuf,
    str::FromStr,
    sync::{Arc, mpsc},
    time::Instant,
};

use dirk_events::EventManager;
use dirk_threads::WorkerPool;
use dirk_universe::{Universe, UniverseBuilder};
use parking_lot::RwLock;
use tracing::info;

#[cfg(feature = "editor")]
use crate::editor::{self, EditorBuildContext, EditorServices, EditorSubsystem};
use crate::{
    Engine, EngineHandle, EngineMetadata, EnginePlugin, EngineResource, EngineState, Result,
    Subsystem, errors::Error, signal::OperatingSystemSignals,
};

type SubsystemFactory =
    Box<dyn FnOnce(&mut EngineBuildContext) -> anyhow::Result<Box<dyn Subsystem>> + 'static>;

/// Builds an [`Engine`] from core configuration and plugin registrations.
///
/// Register plugins with [`EngineBuilder::with_plugin`]. The method mutates the
/// builder and is idempotent by the concrete plugin type, which allows plugins
/// to call `with_plugin` for their dependencies without coordinating with the
/// application or with other plugins.
pub struct EngineBuilder {
    app_name: String,
    app_version: dirk_utils::Version,
    engine_name: String,
    engine_version: dirk_utils::Version,
    worker_name: String,
    log_level: piquel_log::LogLevel,
    handle_os_signals: bool,
    plugins: HashSet<TypeId>,
    plugins_in_progress: HashSet<TypeId>,
    failed_plugin: Option<&'static str>,
    subsystem_factories: HashMap<TypeId, SubsystemFactory>,
    subsystem_order: Vec<TypeId>,
    #[cfg(feature = "editor")]
    editor_subsystem_factories: HashMap<TypeId, editor::EditorSubsystemFactory>,
    #[cfg(feature = "editor")]
    editor_subsystem_order: Vec<TypeId>,
}

impl EngineBuilder {
    /// Creates an empty engine builder with default core configuration.
    ///
    /// # Panics
    ///
    /// Panics if this crate's package version is not a valid `DirkEngine`
    /// semantic version. This is a build-time configuration error.
    #[must_use]
    pub fn new() -> Self {
        let package_version = dirk_utils::Version::from_str(env!("CARGO_PKG_VERSION"))
            .expect("crate package version should be a valid DirkEngine version");

        Self {
            app_name: "DirkEngine".to_owned(),
            app_version: package_version,
            engine_name: "DirkEngine".to_owned(),
            engine_version: package_version,
            worker_name: "dirk-workers".to_owned(),
            log_level: piquel_log::LogLevel::Info,
            handle_os_signals: false,
            plugins: HashSet::new(),
            plugins_in_progress: HashSet::new(),
            failed_plugin: None,
            subsystem_factories: HashMap::new(),
            subsystem_order: Vec::new(),
            #[cfg(feature = "editor")]
            editor_subsystem_factories: HashMap::new(),
            #[cfg(feature = "editor")]
            editor_subsystem_order: Vec::new(),
        }
    }

    /// Sets the application name used for diagnostics.
    pub fn with_app_name(&mut self, app_name: impl Into<String>) -> &mut Self {
        self.app_name = app_name.into();
        self
    }

    /// Sets the application version used for diagnostics and subsystem metadata.
    pub fn with_app_version(&mut self, app_version: dirk_utils::Version) -> &mut Self {
        self.app_version = app_version;
        self
    }

    /// Sets the worker thread name prefix.
    pub fn with_worker_name(&mut self, worker_name: impl Into<String>) -> &mut Self {
        self.worker_name = worker_name.into();
        self
    }

    /// Sets the maximum log level configured by the engine.
    pub fn with_log_level(&mut self, level: piquel_log::LogLevel) -> &mut Self {
        self.log_level = level;
        self
    }

    /// Opts into process-wide SIGINT and termination handling.
    ///
    /// The host application should enable this only when the engine owns its
    /// signal policy for the process lifetime. `signal-hook` cannot restore the
    /// previous disposition after the engine is dropped. Window close events
    /// are handled separately and do not require this option.
    pub fn with_os_signals(&mut self, enabled: bool) -> &mut Self {
        self.handle_os_signals = enabled;
        self
    }

    /// Registers a plugin with the builder, unless this concrete plugin type
    /// has already been registered.
    ///
    /// Plugins may call this method from their own [`EnginePlugin::build`]
    /// implementation to declare dependencies. The builder uses
    /// [`TypeId::of::<P>`] for idempotency, so a second registration of the
    /// same concrete plugin type is skipped even if it was requested by a
    /// different plugin.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PluginBuildFailed`] if the plugin fails to register its
    /// build-time pieces. A failed registration makes the builder unusable:
    /// plugin callbacks may have already changed it, and retrying would retain
    /// those partial changes.
    pub fn with_plugin<P>(&mut self, plugin: P) -> Result<&mut Self>
    where
        P: EnginePlugin + 'static,
    {
        if let Some(name) = self.failed_plugin {
            return Err(Error::PluginBuilderPoisoned { name });
        }
        let type_id = TypeId::of::<P>();
        if self.plugins.contains(&type_id) {
            return Ok(self);
        }

        let name = plugin.name();
        if !self.plugins_in_progress.insert(type_id) {
            return Err(Error::PluginDependencyCycle {
                name,
                type_name: type_name::<P>(),
            });
        }

        let result = plugin.build(self);
        self.plugins_in_progress.remove(&type_id);

        if let Err(source) = result {
            self.failed_plugin.get_or_insert(name);
            return Err(Error::PluginBuildFailed { name, source });
        }
        self.plugins.insert(type_id);
        drop(plugin);
        Ok(self)
    }

    /// Adds a runtime subsystem factory.
    ///
    /// The factory runs during [`EngineBuilder::build`], after core engine
    /// services have been created. It receives an [`EngineBuildContext`] so it
    /// can use core services and publish or read typed resources.
    ///
    /// A subsystem factory should publish a resource when later subsystem
    /// factories need a stable handle to the subsystem's capability. The
    /// subsystem should still own its mutable runtime state; the resource should
    /// be an immutable, cheap-to-clone handle into that state.
    ///
    /// Duplicate subsystem types are skipped. The first registration controls
    /// runtime order.
    pub fn add_subsystem<F, S>(&mut self, factory: F) -> &mut Self
    where
        F: FnOnce(&mut EngineBuildContext) -> anyhow::Result<S> + 'static,
        S: Subsystem + 'static,
    {
        let type_id = TypeId::of::<S>();
        if self.subsystem_factories.contains_key(&type_id) {
            return self;
        }

        self.subsystem_factories.insert(
            type_id,
            Box::new(move |context| Ok(Box::new(factory(context)?) as Box<dyn Subsystem>)),
        );
        self.subsystem_order.push(type_id);
        self
    }

    /// Adds an editor subsystem factory.
    ///
    /// Duplicate editor subsystem types are skipped. The first registration
    /// controls editor lifecycle order.
    #[cfg(feature = "editor")]
    pub fn add_editor_subsystem<F, S>(&mut self, factory: F) -> &mut Self
    where
        F: FnOnce(&mut EditorBuildContext<'_>) -> anyhow::Result<S> + Send + 'static,
        S: EditorSubsystem + 'static,
    {
        let type_id = TypeId::of::<S>();
        if self.editor_subsystem_factories.contains_key(&type_id) {
            return self;
        }

        self.editor_subsystem_factories.insert(
            type_id,
            Box::new(move |context| {
                Ok(Box::new(factory(context).map_err(|source| {
                    Error::EditorSubsystemFailedInit {
                        type_name: type_name::<S>(),
                        source,
                    }
                })?) as Box<dyn EditorSubsystem>)
            }),
        );
        self.editor_subsystem_order.push(type_id);
        self
    }

    /// Builds a new engine.
    ///
    /// # Errors
    ///
    /// Returns an error if logging or subsystem initialization fails.
    pub fn build(mut self) -> Result<Engine> {
        if let Some(name) = self.failed_plugin {
            return Err(Error::PluginBuilderPoisoned { name });
        }
        let logger = piquel_log::Logger::new()
            .with_max_level(self.log_level)
            .with_log_bridge(true)
            .with_file(piquel_log::FileConfig::new(
                PathBuf::from(std::env!("SAVED_PATH")).join("logs"),
            ));

        if let Err(err) = logger.init() {
            match err {
                piquel_log::InitError::AlreadyInitialized
                | piquel_log::InitError::LogBridgeAlreadyInitialized => {}
                err @ piquel_log::InitError::Build(_) => return Err(err.into()),
            }
        }

        #[cfg(feature = "editor")]
        info!("starting editor");

        info!(app = self.app_name, "initialising engine");

        let metadata = Arc::new(EngineMetadata::new(
            self.app_name.clone(),
            self.app_version,
            self.engine_name,
            self.engine_version,
        ));
        let workers = WorkerPool::new(&self.worker_name);
        let events = EventManager::new(workers.clone());
        let state = Arc::new(EngineState::new());
        let (commands, command_receiver) = mpsc::channel();
        let resources = Arc::new(RwLock::new(HashMap::new()));
        let handle = EngineHandle {
            metadata,
            state: Arc::clone(&state),
            events: events.clone(),
            workers: workers.clone(),
            commands,
            resources,
        };

        let mut context = EngineBuildContext {
            handle: handle.clone(),
            builder: Universe::builder(),
        };
        context.add_resource(context.builder.handle())?;

        #[cfg(feature = "editor")]
        let editor_services = {
            let services = EditorServices::new();
            context.add_resource(services.clone())?;
            services
        };

        let signals = if self.handle_os_signals {
            OperatingSystemSignals::install().map_err(Error::SignalHandlerInitFailed)?
        } else {
            OperatingSystemSignals::disabled()
        };

        let mut subsystems = Vec::with_capacity(self.subsystem_factories.len());

        for type_id in self.subsystem_order {
            if let Some(factory) = self.subsystem_factories.remove(&type_id) {
                subsystems.push(factory(&mut context).map_err(Error::SubsystemFailedInit)?);
            }
        }

        #[cfg(feature = "editor")]
        let editor = {
            let mut editor_subsystems = Vec::with_capacity(self.editor_subsystem_factories.len());
            for type_id in self.editor_subsystem_order {
                let mut editor_context = EditorBuildContext {
                    engine: &mut context,
                    editor: &editor_services,
                };
                if let Some(factory) = self.editor_subsystem_factories.remove(&type_id) {
                    editor_subsystems.push(factory(&mut editor_context)?);
                }
            }
            editor::EditorRuntime::new(editor_services, editor_subsystems)
        };

        let universe = context.builder.build();

        Ok(Engine {
            logger,
            universe,
            subsystems,
            #[cfg(feature = "editor")]
            editor,
            state,
            handle,
            command_receiver,
            signals,
            frame_dispatcher: events.register(),
            exiting_dispatcher: events.register(),
            last_tick: Instant::now(),
            lifecycle: crate::EngineLifecycle::Ready,
        })
    }
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build-time access to core engine services and typed subsystem resources.
///
/// `EngineBuildContext` is passed to subsystem factories registered with
/// [`EngineBuilder::add_subsystem`]. It exposes the immutable core handles that
/// are safe to use while the engine is being assembled, plus a type-driven
/// resource map.
///
/// Resources are available immediately after they are added, but only to later
/// subsystem factories in the registration order. This makes dependencies
/// explicit without giving plugins a separate dependency solver:
///
/// ```rust
/// # use dirk_engine::{EngineBuildContext, EngineBuilder, EnginePlugin, Subsystem};
/// # #[derive(Clone)]
/// # struct SharedHandle;
/// # struct Provider;
/// # impl Subsystem for Provider { fn name(&self) -> &'static str { "provider" } }
/// # struct Consumer;
/// # impl Subsystem for Consumer { fn name(&self) -> &'static str { "consumer" } }
/// # struct ProviderPlugin;
/// # impl EnginePlugin for ProviderPlugin {
/// #     fn name(&self) -> &'static str { "provider" }
/// #     fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
/// builder.add_subsystem(|ctx: &mut EngineBuildContext| {
///     ctx.add_resource(SharedHandle)?;
///     Ok(Provider)
/// });
/// #         Ok(())
/// #     }
/// # }
/// # struct ConsumerPlugin;
/// impl EnginePlugin for ConsumerPlugin {
///     fn name(&self) -> &'static str {
///         "consumer"
///     }
///
///     fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
///         builder.with_plugin(ProviderPlugin)?;
///         builder.add_subsystem(|ctx| {
///             let _handle = ctx.resource::<SharedHandle>()?;
///             Ok(Consumer)
///         });
///         Ok(())
///     }
/// }
/// ```
pub struct EngineBuildContext {
    handle: EngineHandle,
    builder: UniverseBuilder,
}

impl EngineBuildContext {
    #[cfg(test)]
    pub(crate) fn new(handle: EngineHandle) -> Self {
        Self {
            handle,
            builder: Universe::builder(),
        }
    }
    /// Returns the engine handle being shared with runtime subsystems.
    #[must_use]
    pub fn handle(&self) -> &EngineHandle {
        &self.handle
    }

    /// Returns the shared event manager.
    #[must_use]
    pub fn events(&self) -> &EventManager {
        self.handle.events()
    }

    /// Returns the worker pool.
    #[must_use]
    pub fn workers(&self) -> &WorkerPool {
        self.handle.workers()
    }

    /// Extends the engine ECS builder with another prepared ECS builder.
    pub fn extend_universe(&mut self, builder: UniverseBuilder) -> &mut Self {
        let universe_builder = std::mem::take(&mut self.builder);
        self.builder = universe_builder.with_other(builder);
        self
    }

    /// Publishes a typed resource for later subsystem factories.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ResourceAlreadyRegistered`] if another resource of the
    /// same concrete type has already been published.
    pub fn add_resource<T: EngineResource>(&mut self, resource: T) -> Result<&mut Self> {
        let type_id = TypeId::of::<T>();
        {
            let mut resources = self.handle.resources.write();
            if resources.contains_key(&type_id) {
                return Err(Error::ResourceAlreadyRegistered {
                    type_name: type_name::<T>(),
                });
            }

            resources.insert(type_id, Box::new(resource));
        }
        Ok(self)
    }

    /// Clones a previously published resource handle by concrete type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ResourceMissing`] if no resource of this type has been
    /// published by an earlier subsystem factory.
    ///
    /// Returns [`Error::ResourceTypeMismatch`] if the stored value does not
    /// match the requested concrete type. This should not occur during normal
    /// engine use because resources are keyed by the same [`TypeId`] used for
    /// insertion.
    pub fn resource<T: EngineResource>(&self) -> Result<T> {
        let type_id = TypeId::of::<T>();
        let resources = self.handle.resources.read();
        let resource = resources
            .get(&type_id)
            .ok_or_else(|| Error::ResourceMissing {
                type_name: type_name::<T>(),
            })?;

        resource
            .downcast_ref::<T>()
            .cloned()
            .ok_or_else(|| Error::ResourceTypeMismatch {
                type_name: type_name::<T>(),
            })
    }
}
