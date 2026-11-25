//! Runtime subsystem lifecycle API. The engine owns subsystems once
//! [`EngineBuilder`] has built the core runtime, and its loop starts, ticks and
//! shuts them down; plugins only configure the builder.
//!
//! Subsystem factories receive an [`EngineBuildContext`]. A factory publishes
//! an [`EngineResource`], a cloneable build-time handle, when later factories
//! need to find something its subsystem creates; the subsystem still owns its
//! mutable state.
//!
//! [`EngineBuildContext`]: crate::EngineBuildContext

use dirk_universe::Universe;

use crate::{EngineBuilder, EngineHandle};

/// Builder-time extension point for engine features.
///
/// Plugins configure an [`EngineBuilder`]. They should register subsystem
/// factories, ECS systems, event registrations, editor panels, or other future
/// build-time integrations. They are not ticked by the engine.
///
/// Dependencies are declared by calling [`EngineBuilder::with_plugin`] inside
/// [`EnginePlugin::build`]. Because plugin registration is idempotent, two
/// plugins can request the same dependency without building it twice.
pub trait EnginePlugin {
    /// Returns the plugin name for diagnostics.
    fn name(&self) -> &'static str;

    /// Registers this plugin with the engine builder.
    ///
    /// # Errors
    ///
    /// Returns an error if plugin registration fails.
    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()>;
}

/// Marker trait for cloneable handles published during engine build.
///
/// Resources are type-driven and are stored by their concrete type. They should
/// be cheap to clone and safe to share across threads. A resource should not be
/// the primary owner of mutable runtime behavior; the subsystem that creates it
/// should keep that state and publish only the handle that other subsystem
/// factories need. Interior synchronization is acceptable when shared access is
/// necessary.
pub trait EngineResource: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> EngineResource for T {}

/// A runtime system owned and driven by the engine.
///
/// Subsystems are the lifecycle primitive for mutable runtime behavior. They
/// are created from factories during [`EngineBuilder::build`], started before
/// the first tick, ticked in registration order, and shut down in reverse
/// registration order.
pub trait Subsystem {
    /// The name of the subsystem.
    ///
    /// Used for diagnostics and error reporting.
    fn name(&self) -> &'static str;

    /// Starts the subsystem after all subsystems and the ECS have been built.
    ///
    /// The default implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if startup fails.
    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        Ok(())
    }

    /// Advances the subsystem by one engine tick.
    ///
    /// The default implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the subsystem cannot complete its tick.
    fn tick(
        &mut self,
        _delta_time: f64,
        _handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Shuts the subsystem down before the engine releases core services.
    ///
    /// Engine shutdown calls this even if startup failed or this subsystem's
    /// [`Self::start`] was never called. Clean up only resources acquired so far.
    ///
    /// The default implementation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails.
    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        Ok(())
    }
}
