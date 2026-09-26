//! This module contains all engine related errors.

use thiserror::Error;

/// A wrapper for the result type that has an [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// All errors that could be emitted by the engine.
#[derive(Error, Debug)]
pub enum Error {
    /// An operation was attempted after the engine entered a terminal state.
    #[error("cannot {operation} engine in {status:?} state")]
    InvalidLifecycleState {
        /// The requested operation.
        operation: &'static str,
        /// The current engine status.
        status: crate::EngineStatus,
    },
    /// A plugin failed earlier and its partial builder mutations cannot be retried safely.
    #[error("engine builder cannot be reused after plugin `{name}` failed")]
    PluginBuilderPoisoned {
        /// The plugin whose registration failed.
        name: &'static str,
    },
    /// A subsystem reported a terminal runtime failure through its engine handle.
    #[error("engine reported a runtime failure: {source}")]
    RuntimeFailed {
        /// The reported failure.
        #[source]
        source: anyhow::Error,
    },
    /// An error occurred when initializing the logger.
    #[error("failed to initialize logging")]
    LoggerFailure(#[from] piquel_log::InitError),
    /// When a plugin failed to register its build-time pieces.
    #[error("plugin `{name}` failed to build")]
    PluginBuildFailed {
        /// The plugin name.
        name: &'static str,
        /// The error that caused the build failure.
        #[source]
        source: anyhow::Error,
    },
    /// When plugin dependencies contain a cycle.
    #[error("plugin dependency cycle detected while building `{name}` ({type_name})")]
    PluginDependencyCycle {
        /// The plugin name.
        name: &'static str,
        /// The concrete plugin type name.
        type_name: &'static str,
    },
    /// When a requested typed engine resource has not been registered.
    #[error("resource `{type_name}` was not registered")]
    ResourceMissing {
        /// The requested resource type name.
        type_name: &'static str,
    },
    /// When a typed engine resource has already been registered.
    #[error("resource `{type_name}` was already registered")]
    ResourceAlreadyRegistered {
        /// The duplicate resource type name.
        type_name: &'static str,
    },
    /// When a stored resource does not match the requested concrete type.
    #[error("resource `{type_name}` had an unexpected concrete type")]
    ResourceTypeMismatch {
        /// The requested resource type name.
        type_name: &'static str,
    },
    /// When an engine subsystem failed to tick.
    #[error("subsystem {name} failed tick: {source}")]
    SubsystemFailedTick {
        /// The name of the subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an engine subsystem failed to start.
    #[error("subsystem {name} failed start: {source}")]
    SubsystemFailedStart {
        /// The name of the subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an engine subsystem failed to shut down.
    #[error("subsystem {name} failed shutdown: {source}")]
    SubsystemFailedShutdown {
        /// The name of the subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an engine subsystem failed to initialize.
    #[error("subsystem failed to initialize: {0}")]
    SubsystemFailedInit(#[source] anyhow::Error),
    /// When an editor subsystem failed to initialize.
    #[cfg(feature = "editor")]
    #[error("editor subsystem `{type_name}` failed to initialize: {source}")]
    EditorSubsystemFailedInit {
        /// The concrete editor subsystem type name.
        type_name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an editor subsystem failed to start.
    #[cfg(feature = "editor")]
    #[error("editor subsystem {name} failed start: {source}")]
    EditorSubsystemFailedStart {
        /// The name of the editor subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an editor subsystem failed to tick.
    #[cfg(feature = "editor")]
    #[error("editor subsystem {name} failed tick: {source}")]
    EditorSubsystemFailedTick {
        /// The name of the editor subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When an editor subsystem failed to shut down.
    #[cfg(feature = "editor")]
    #[error("editor subsystem {name} failed shutdown: {source}")]
    EditorSubsystemFailedShutdown {
        /// The name of the editor subsystem.
        name: &'static str,
        /// The error that caused the failure.
        #[source]
        source: anyhow::Error,
    },
    /// When operating system signal handlers failed to initialize.
    #[error("failed to initialize operating system signal handlers: {0}")]
    SignalHandlerInitFailed(#[source] anyhow::Error),
    /// An error occurred while starting.
    #[error("engine failed to start: {0}")]
    StartFailed(#[source] anyhow::Error),
    /// An error occurred while ticking.
    #[error("engine failed while ticking: {0}")]
    TickFailed(#[source] anyhow::Error),
    /// An error occurred while shutting down.
    #[error("engine failed while shutting down: {0}")]
    ShutdownFailed(#[source] anyhow::Error),
}
