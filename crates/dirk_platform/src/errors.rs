use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by out platform API
#[derive(Debug, Error)]
pub enum Error {
    /// The main native window could not be created during startup.
    #[error("Failed to create main window: {0}")]
    WindowCreation(#[source] anyhow::Error),
    /// Wrapper type for event loop errors. These are essentially
    /// platform errors.
    #[error("Error in winit event loop: {0}")]
    EventLoopError(#[from] winit::error::EventLoopError),
    /// Any errors to do with the window handle.
    #[error("Error fetching handle: {0}")]
    HandleError(#[from] winit::raw_window_handle::HandleError),
    /// Error returned if App exit is requested during initialisation.
    #[error("Application exited with code {0}")]
    AppExited(i32),
}
