use raw_window_handle::HandleError;
use thiserror::Error;

/// A renderer result.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors the renderer reports.
#[derive(Debug, Error)]
pub enum Error {
    /// The render hardware interface failed.
    #[error(transparent)]
    Rhi(#[from] dirk_rhi::Error),
    /// The platform failed, such as to provide a window's handles.
    #[error("platform error: {0}")]
    Platform(#[from] dirk_platform::Error),
    /// An asset could not be read.
    #[error("assets error: {0}")]
    Asset(#[from] dirk_assets::Error),

    /// The device renders to none of the depth formats scenes use.
    #[error("the device renders to no depth format scenes use")]
    NoDepthFormat,
    /// egui updated part of a texture it never set.
    #[cfg(feature = "editor")]
    #[error("egui updated part of texture {0:?}, which it never set")]
    UnknownEguiTexture(egui::TextureId),

    /// The camera of a viewport is not a renderer entity, or has no view.
    #[error("camera {0:?} does not exist")]
    CameraDoesNotExist(dirk_universe::Entity),
    /// The requested world does not exist.
    #[error("world {0} is not registered on renderer")]
    WorldDoesNotExist(dirk_universe::WorldId),
    /// The requested entity does not exist.
    #[error("entity {0:?} is not registered on renderer")]
    EntityDoesNotExist(dirk_universe::Entity),
}

impl From<HandleError> for Error {
    fn from(value: HandleError) -> Self {
        Error::Platform(value.into())
    }
}
