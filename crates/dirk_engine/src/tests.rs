#![cfg(test)]

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

use parking_lot::{Mutex, RwLock};

use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
struct TestResource(&'static str);

struct CountingPlugin {
    builds: Arc<AtomicUsize>,
}

impl EnginePlugin for CountingPlugin {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn build(&self, _builder: &mut EngineBuilder) -> anyhow::Result<()> {
        self.builds.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

struct SharedDependencyPlugin {
    builds: Arc<AtomicUsize>,
}

impl EnginePlugin for SharedDependencyPlugin {
    fn name(&self) -> &'static str {
        "shared-dependency"
    }

    fn build(&self, _builder: &mut EngineBuilder) -> anyhow::Result<()> {
        self.builds.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

struct DependentPlugin {
    dependency_builds: Arc<AtomicUsize>,
}

impl EnginePlugin for DependentPlugin {
    fn name(&self) -> &'static str {
        "dependent"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.with_plugin(SharedDependencyPlugin {
            builds: Arc::clone(&self.dependency_builds),
        })?;
        Ok(())
    }
}

struct OtherDependentPlugin {
    dependency_builds: Arc<AtomicUsize>,
}

impl EnginePlugin for OtherDependentPlugin {
    fn name(&self) -> &'static str {
        "other-dependent"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.with_plugin(SharedDependencyPlugin {
            builds: Arc::clone(&self.dependency_builds),
        })?;
        Ok(())
    }
}

struct SelfDependentPlugin;

impl EnginePlugin for SelfDependentPlugin {
    fn name(&self) -> &'static str {
        "self-dependent"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.with_plugin(SelfDependentPlugin)?;
        Ok(())
    }
}

struct StartFailingSubsystem;

impl Subsystem for StartFailingSubsystem {
    fn name(&self) -> &'static str {
        "start-failing"
    }

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("start failed"))
    }
}

struct ShutdownFailingSubsystem;

impl Subsystem for ShutdownFailingSubsystem {
    fn name(&self) -> &'static str {
        "shutdown-failing"
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("shutdown failed"))
    }
}

struct RecordedSubsystem {
    name: &'static str,
    events: Arc<Mutex<Vec<&'static str>>>,
    fail_start: bool,
    fail_shutdown: bool,
}

impl Subsystem for RecordedSubsystem {
    fn name(&self) -> &'static str {
        self.name
    }

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push(self.name);
        anyhow::ensure!(!self.fail_start, "start failed");
        Ok(())
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push(self.name);
        anyhow::ensure!(!self.fail_shutdown, "shutdown failed");
        Ok(())
    }
}

struct RuntimeFailingSubsystem;

impl Subsystem for RuntimeFailingSubsystem {
    fn name(&self) -> &'static str {
        "runtime-failing"
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        handle.exit();
        handle.exit_with_error(anyhow::anyhow!("runtime fault"));
        Ok(())
    }
}

struct PartiallyFailingPlugin;

impl EnginePlugin for PartiallyFailingPlugin {
    fn name(&self) -> &'static str {
        "partially-failing"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.add_subsystem(|_| Ok(StartFailingSubsystem));
        anyhow::bail!("plugin setup failed")
    }
}

struct PublishingSubsystem {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl Subsystem for PublishingSubsystem {
    fn name(&self) -> &'static str {
        "publishing"
    }

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push("publishing-start");
        Ok(())
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        _handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        self.events.lock().push("publishing-tick");
        Ok(())
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push("publishing-shutdown");
        Ok(())
    }
}

struct ReadingSubsystem {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl Subsystem for ReadingSubsystem {
    fn name(&self) -> &'static str {
        "reading"
    }

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push("reading-start");
        Ok(())
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        self.events.lock().push("reading-tick");
        handle.exit();
        Ok(())
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        self.events.lock().push("reading-shutdown");
        Ok(())
    }
}

struct UniverseHandleSubsystem {
    handle: dirk_universe::UniverseHandle,
    world: Arc<Mutex<Option<dirk_universe::WorldId>>>,
}

impl Subsystem for UniverseHandleSubsystem {
    fn name(&self) -> &'static str {
        "universe-handle"
    }

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        let mut command_buffer = self.handle.command_buffer();
        let world = command_buffer.create_world(dirk_universe::World::builder("resource-world"));
        command_buffer.submit();
        *self.world.lock() = Some(world);
        Ok(())
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        handle.exit();
        Ok(())
    }
}

pub(crate) fn build_context() -> EngineBuildContext {
    let workers = WorkerPool::new("dirk-engine-test");
    let events = EventManager::new(workers.clone());
    let (commands, _command_receiver) = mpsc::channel();

    EngineBuildContext::new(EngineHandle {
        metadata: Arc::new(EngineMetadata::new(
            "test-app",
            dirk_utils::Version::ZERO,
            "test-engine",
            dirk_utils::Version::ZERO,
        )),
        state: Arc::new(EngineState::new()),
        events,
        workers,
        commands,
        resources: Arc::new(RwLock::new(HashMap::new())),
    })
}

pub(crate) fn engine_with_subsystems(subsystems: Vec<Box<dyn Subsystem>>) -> Engine {
    engine_with_subsystems_and_signals(
        subsystems,
        signal::OperatingSystemSignals::empty_for_tests(),
    )
}

fn engine_with_subsystems_and_signals(
    subsystems: Vec<Box<dyn Subsystem>>,
    signals: signal::OperatingSystemSignals,
) -> Engine {
    let workers = WorkerPool::new("dirk-engine-test");
    let events = EventManager::new(workers.clone());
    let state = Arc::new(EngineState::new());
    let (commands, command_receiver) = mpsc::channel();
    let resources = Arc::new(RwLock::new(HashMap::new()));
    let handle = EngineHandle {
        metadata: Arc::new(EngineMetadata::new(
            "test-app",
            dirk_utils::Version::ZERO,
            "test-engine",
            dirk_utils::Version::ZERO,
        )),
        state: Arc::clone(&state),
        events: events.clone(),
        workers,
        commands,
        resources,
    };

    Engine {
        logger: piquel_log::Logger::new(),
        universe: Universe::builder().build(),
        subsystems,
        #[cfg(feature = "editor")]
        editor: editor::EditorRuntime::empty_for_tests(),
        state,
        handle,
        command_receiver,
        signals,
        frame_dispatcher: events.register(),
        exiting_dispatcher: events.register(),
        last_tick: Instant::now(),
        lifecycle: EngineLifecycle::Ready,
    }
}

struct CountingTickSubsystem {
    ticks: Arc<AtomicUsize>,
}

impl Subsystem for CountingTickSubsystem {
    fn name(&self) -> &'static str {
        "counting-tick"
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        _handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        self.ticks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn registering_same_plugin_twice_only_builds_it_once() -> Result<()> {
    let builds = Arc::new(AtomicUsize::new(0));
    let mut builder = EngineBuilder::new();

    builder.with_plugin(CountingPlugin {
        builds: Arc::clone(&builds),
    })?;
    builder.with_plugin(CountingPlugin {
        builds: Arc::clone(&builds),
    })?;

    assert_eq!(builds.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn failed_plugin_registration_poison_builder() {
    let mut builder = EngineBuilder::new();
    assert!(builder.with_plugin(PartiallyFailingPlugin).is_err());
    assert!(matches!(
        builder.with_plugin(PartiallyFailingPlugin),
        Err(Error::PluginBuilderPoisoned { .. })
    ));
    assert!(matches!(
        builder.build(),
        Err(Error::PluginBuilderPoisoned { .. })
    ));
}

#[test]
fn runtime_error_is_returned_after_shutdown() {
    let engine = engine_with_subsystems(vec![Box::new(RuntimeFailingSubsystem)]);
    let err = engine.run().expect_err("runtime failure must reach caller");
    assert!(err.to_string().contains("runtime failure"));
    assert!(format!("{err:#}").contains("runtime fault"));
}

#[test]
fn manually_driven_engine_can_take_runtime_error() -> Result<()> {
    let mut engine = engine_with_subsystems(vec![Box::new(RuntimeFailingSubsystem)]);
    assert_eq!(engine.tick()?, EngineStatus::Error);
    let err = engine
        .take_error()
        .expect("runtime error must remain available");
    assert_eq!(err.to_string(), "runtime fault");
    engine.shutdown()?;
    Ok(())
}

#[test]
fn failed_start_cannot_repeat_subsystem_start() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = engine_with_subsystems(vec![Box::new(RecordedSubsystem {
        name: "start-once",
        events: Arc::clone(&events),
        fail_start: true,
        fail_shutdown: false,
    })]);
    assert!(engine.start().is_err());
    assert!(matches!(
        engine.start(),
        Err(Error::InvalidLifecycleState { .. })
    ));
    assert!(matches!(
        engine.tick(),
        Err(Error::InvalidLifecycleState { .. })
    ));
    assert_eq!(*events.lock(), vec!["start-once"]);
}

#[test]
fn shutdown_before_start_cannot_restart_engine() -> Result<()> {
    let mut engine = engine_with_subsystems(Vec::new());
    engine.shutdown()?;
    assert_eq!(engine.status(), EngineStatus::Exited);
    assert!(matches!(
        engine.tick(),
        Err(Error::InvalidLifecycleState { .. })
    ));
    Ok(())
}

#[test]
fn shutdown_attempts_every_subsystem_after_error() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let subsystems: Vec<Box<dyn Subsystem>> =
        [("first", false), ("second", true), ("third", false)]
            .into_iter()
            .map(|(name, fail_shutdown)| {
                Box::new(RecordedSubsystem {
                    name,
                    events: Arc::clone(&events),
                    fail_start: false,
                    fail_shutdown,
                }) as Box<dyn Subsystem>
            })
            .collect();
    let mut engine = engine_with_subsystems(subsystems);
    engine.start().expect("start succeeds");
    assert!(matches!(
        engine.shutdown(),
        Err(Error::SubsystemFailedShutdown { name: "second", .. })
    ));
    assert_eq!(
        *events.lock(),
        vec!["first", "second", "third", "third", "second", "first"]
    );
    assert_eq!(engine.status(), EngineStatus::Exited);
}

#[test]
fn plugin_can_register_another_plugin_as_a_dependency() -> Result<()> {
    let dependency_builds = Arc::new(AtomicUsize::new(0));
    let mut builder = EngineBuilder::new();

    builder.with_plugin(DependentPlugin {
        dependency_builds: Arc::clone(&dependency_builds),
    })?;

    assert_eq!(dependency_builds.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn shared_dependency_is_registered_once_for_multiple_dependents() -> Result<()> {
    let dependency_builds = Arc::new(AtomicUsize::new(0));
    let mut builder = EngineBuilder::new();

    builder.with_plugin(DependentPlugin {
        dependency_builds: Arc::clone(&dependency_builds),
    })?;
    builder.with_plugin(OtherDependentPlugin {
        dependency_builds: Arc::clone(&dependency_builds),
    })?;

    assert_eq!(dependency_builds.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn cyclic_plugin_dependency_reports_error() {
    let mut builder = EngineBuilder::new();

    let result = builder.with_plugin(SelfDependentPlugin);

    match result {
        Err(Error::PluginBuildFailed {
            name: "self-dependent",
            source,
        }) => assert!(matches!(
            source.downcast_ref::<Error>(),
            Some(Error::PluginDependencyCycle {
                name: "self-dependent",
                ..
            })
        )),
        _ => panic!("expected self-dependent plugin build to fail with a cycle"),
    }
}

#[test]
fn subsystem_start_failure_uses_start_error_variant() {
    let mut engine = engine_with_subsystems(vec![Box::new(StartFailingSubsystem)]);

    let result = engine.start();

    assert!(matches!(
        result,
        Err(Error::SubsystemFailedStart {
            name: "start-failing",
            ..
        })
    ));
}

#[test]
fn subsystem_shutdown_failure_uses_shutdown_error_variant() {
    let mut engine = engine_with_subsystems(vec![Box::new(ShutdownFailingSubsystem)]);

    let result = engine.shutdown();

    assert!(matches!(
        result,
        Err(Error::SubsystemFailedShutdown {
            name: "shutdown-failing",
            ..
        })
    ));
}

#[test]
fn operating_system_signals_are_processed_before_subsystem_ticks() -> Result<()> {
    let ticks = Arc::new(AtomicUsize::new(0));
    let signals =
        signal::OperatingSystemSignals::with_signal_for_tests(signal_hook::consts::SIGINT)
            .map_err(Error::SubsystemFailedInit)?;
    let mut engine = engine_with_subsystems_and_signals(
        vec![Box::new(CountingTickSubsystem {
            ticks: Arc::clone(&ticks),
        })],
        signals,
    );

    let status = engine.tick()?;

    assert_eq!(status, EngineStatus::ExitRequested);
    assert_eq!(ticks.load(Ordering::Relaxed), 0);
    engine.shutdown()?;
    Ok(())
}

#[test]
fn metadata_is_available_on_engine_handle() {
    let context = build_context();
    let metadata = context.handle().metadata();

    assert_eq!(metadata.app_name(), "test-app");
    assert_eq!(metadata.app_version(), dirk_utils::Version::ZERO);
    assert_eq!(metadata.engine_name(), "test-engine");
    assert_eq!(metadata.engine_version(), dirk_utils::Version::ZERO);
}

#[test]
fn resource_lookup_reports_missing_resources() {
    let context = build_context();

    let result = context.resource::<TestResource>();

    assert!(matches!(result, Err(Error::ResourceMissing { .. })));
}

#[test]
fn duplicate_resource_registration_reports_error() -> Result<()> {
    let mut context = build_context();

    context.add_resource(TestResource("first"))?;
    let result = context.add_resource(TestResource("second"));

    assert!(matches!(
        result,
        Err(Error::ResourceAlreadyRegistered { .. })
    ));
    Ok(())
}

#[test]
fn subsystem_factories_publish_and_read_resources_and_lifecycle_still_runs() -> Result<()> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut builder = EngineBuilder::new();

    {
        let events = Arc::clone(&events);
        builder.add_subsystem(move |ctx| {
            events.lock().push("publishing-factory");
            ctx.add_resource(TestResource("resource"))?;
            Ok(PublishingSubsystem { events })
        });
    }

    {
        let events = Arc::clone(&events);
        builder.add_subsystem(move |ctx| {
            let resource = ctx.resource::<TestResource>()?;
            assert_eq!(resource, TestResource("resource"));
            events.lock().push("reading-factory");
            Ok(ReadingSubsystem { events })
        });
    }

    let mut engine = builder.build()?;

    assert_eq!(&*events.lock(), &["publishing-factory", "reading-factory"]);

    let status = engine.tick()?;
    assert_eq!(status, EngineStatus::ExitRequested);
    engine.shutdown()?;

    assert_eq!(
        &*events.lock(),
        &[
            "publishing-factory",
            "reading-factory",
            "publishing-start",
            "reading-start",
            "publishing-tick",
            "reading-tick",
            "reading-shutdown",
            "publishing-shutdown",
        ]
    );

    Ok(())
}

#[test]
fn universe_handle_is_available_as_engine_resource() -> Result<()> {
    let world = Arc::new(Mutex::new(None));
    let mut builder = EngineBuilder::new();

    {
        let world = Arc::clone(&world);
        builder.add_subsystem(move |ctx| {
            Ok(UniverseHandleSubsystem {
                handle: ctx.resource::<dirk_universe::UniverseHandle>()?,
                world,
            })
        });
    }

    let mut engine = builder.build()?;
    let status = engine.tick()?;
    assert_eq!(status, EngineStatus::ExitRequested);

    let world = world.lock().expect("subsystem should record created world");
    assert_eq!(
        engine.universe.world(world).map(dirk_universe::World::name),
        Some("resource-world")
    );

    engine.shutdown()?;
    Ok(())
}
