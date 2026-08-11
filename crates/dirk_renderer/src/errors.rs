use raw_window_handle::HandleError;
use thiserror::Error;

/// An engine Result type to wrap the engine [enum@Error].
pub type Result<T> = std::result::Result<T, Error>;

/// All engine related errors
#[derive(Debug, Error)]
pub enum Error {
    /// An error returned by the temporary Vulkan editor adapter.
    #[cfg(feature = "editor")]
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),
    /// Error produced by the render hardware interface.
    #[error("RHI error: {0}")]
    Rhi(#[from] dirk_rhi::Error),

    /// An error loading Vulkan functions for the temporary editor adapter.
    #[cfg(feature = "editor")]
    #[error("Error loading Vulkan functions: {0}")]
    Loading(#[from] ash::LoadingError),
    /// Error produced by the [`platform`] crate.
    #[error("platform error: {0}")]
    Platform(#[from] dirk_platform::Error),
    /// Error produced by the [`assets`] crate.
    #[error("assets error: {0}")]
    AssetError(#[from] dirk_assets::Error),
    /// Error produced by the egui Vulkan renderer.
    #[cfg(feature = "editor")]
    #[error("egui renderer error: {0}")]
    EguiRenderer(#[from] egui_ash_renderer::RendererError),

    /// A glTF material references an image index that does not exist.
    #[error("glTF image index {0} is out of range")]
    TextureIndexOutOfRange(usize),
    /// The host descriptor set layout does not match the reflected shader layout.
    #[error("pipeline {pipeline} descriptor layout mismatch at set {set}")]
    PipelineDescriptorLayoutMismatch {
        /// Pipeline name.
        pipeline: &'static str,
        /// Descriptor set index.
        set: usize,
    },
    /// The host vertex input layout does not match the reflected shader layout.
    #[error("pipeline {pipeline} vertex input layout mismatch")]
    PipelineVertexInputMismatch {
        /// Pipeline name.
        pipeline: &'static str,
    },

    /// If there is no camera in the scene
    #[error("camera {0:?} does not exist")]
    CameraDoesNotExist(dirk_universe::Entity),
    /// If the requested world does not exist
    #[error("world {0} is not registered on renderer")]
    WorldDoesNotExist(dirk_universe::WorldId),
    /// If the requested entity does not exist
    #[error("entity {0:?} is not registered on renderer")]
    EntityDoesNotExist(dirk_universe::Entity),
    /// The requested window does not exist
    #[error("window {0:?} is not registered on renderer")]
    WindowDoesNotExist(dirk_platform::WindowId),
}

impl From<HandleError> for Error {
    fn from(value: HandleError) -> Self {
        Error::Platform(value.into())
    }
}
