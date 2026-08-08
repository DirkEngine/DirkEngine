use ash::vk;
use raw_window_handle::HandleError;
use thiserror::Error;

/// An engine Result type to wrap the engine [enum@Error].
pub type Result<T> = std::result::Result<T, Error>;

/// All engine related errors
#[derive(Debug, Error)]
pub enum Error {
    /// An error returned by Vulkan
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] ash::vk::Result),
    /// An error occuring during the allocation of a GPU object
    #[error("Allocation error: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),
    /// Error produced by the render hardware interface.
    #[error("RHI error: {0}")]
    Rhi(#[from] dirk_rhi::Error),

    /// An error loading Vulkan function
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

    /// If no physical device is found
    #[error("no suitable graphics device found")]
    NoDeviceFound,
    /// If no supported vulkan image format is found
    #[error("failed to find supported format")]
    NoSupportedFormat,

    /// The required Vulkan instance extension was not found
    #[error("instance extension {0} not found")]
    ExtensionNotFound(String),
    /// The required Vulkan validation layer was not found
    #[error("validation layer {0} not found")]
    ValidationLayerNotFound(String),

    /// Error during layout transition: the specified source layout
    /// is not currently supported by the engine. If it should be supported,
    /// add support in the transition function.
    #[error("the layout {0:?} is not supported as a source. implement it")]
    UnsupportedSourceLayout(vk::ImageLayout),
    /// Error during layout transition: the specified destination layout
    /// is not currently supported by the engine. If it should be supported,
    /// add support in the transition function.
    #[error("the layout {0:?} is not supported as a destination. implement it")]
    UnsupportedDesinationLayout(vk::ImageLayout),

    /// The Vulkan surface is suboptimal
    #[error("suboptimal surface")]
    SuboptimalSurface,
    /// The surface does not support the image usages required by the renderer.
    #[error(
        "surface does not support required swapchain usage {required:?}; supported usage is {supported:?}"
    )]
    UnsupportedSwapchainUsage {
        /// Required swapchain image usage flags.
        required: vk::ImageUsageFlags,
        /// Usage flags advertised by the surface capabilities.
        supported: vk::ImageUsageFlags,
    },
    /// The requested descriptor set allocation is too large for Vulkan.
    #[error("descriptor set allocation count {0} exceeds u32::MAX")]
    DescriptorSetCountTooLarge(usize),
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
