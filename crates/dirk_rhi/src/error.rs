use thiserror::Error;

/// Result returned by backend-neutral RHI operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors common to all graphics backends.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The requested operation is unsupported by the selected backend.
    #[error("unsupported RHI operation: {0}")]
    Unsupported(#[from] UnsupportedOperation),
    /// No graphics device satisfies the requested capabilities.
    #[error("no suitable graphics device was found")]
    NoDevice,
    /// The graphics device was lost.
    #[error("the graphics device was lost")]
    DeviceLost,
    /// A timed wait on a synchronization primitive expired before the
    /// primitive reached its waited-for state.
    #[error("the synchronization wait timed out")]
    Timeout,
    /// The presentation swapchain must be recreated while retaining its
    /// surface.
    #[error("the presentation swapchain is out of date")]
    SwapchainOutOfDate,
    /// The presentation surface itself was lost and must be recreated.
    #[error("the presentation surface was lost")]
    SurfaceLost,
    /// Resource creation or use failed because the request is invalid.
    #[error("invalid RHI request: {0}")]
    InvalidResource(#[from] InvalidResource),
    /// A backend-specific operation failed.
    ///
    /// The payload retains the native error chain, so `Display` reports the
    /// most specific context recorded by the backend.
    #[error("graphics backend error: {0}")]
    Backend(#[from] anyhow::Error),
}

/// Operations a backend may classify as unsupported.
///
/// This enumeration covers only the portable operations the RHI itself
/// defines; failures outside that vocabulary belong in [`Error::Backend`].
/// New members require extending this crate so every backend documents the
/// same gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum UnsupportedOperation {
    /// Requested format is outside device support.
    #[error("unsupported texture format {0:?}")]
    TextureFormat(crate::TextureFormat),
    /// Optional capability is unavailable.
    #[error("unsupported capability: {0}")]
    Capability(&'static str),
    /// Filtered scaling between images is unavailable.
    #[error("image blits are not supported by this backend")]
    ImageBlit,
    /// The supplied shader source representation is not accepted natively.
    #[error("{0} shader source is not supported by this backend")]
    ShaderSource(#[source] ShaderLanguage),
}

/// Source representation of a shader program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum ShaderLanguage {
    /// SPIR-V words.
    #[error("SPIR-V")]
    SpirV,
    /// Metal Shading Language source text.
    #[error("Metal Shading Language")]
    Msl,
}

/// Invalid resource creation or use request with diagnostic context.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{kind}: {detail}")]
pub struct InvalidResource {
    kind: InvalidResourceKind,
    detail: String,
}

impl InvalidResource {
    /// Creates an invalid-resource error without discarding operation-specific
    /// diagnostic detail.
    #[must_use]
    pub fn new(kind: InvalidResourceKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    /// Returns the stable failure classification.
    #[must_use]
    pub const fn kind(&self) -> InvalidResourceKind {
        self.kind
    }

    /// Returns the operation-specific diagnostic detail.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Stable classifications for invalid resource requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum InvalidResourceKind {
    /// The resource belongs to another instance of this backend.
    #[error("resource belongs to a different RHI instance")]
    ForeignInstance,
    /// A requested size, offset, count, or range exceeds supported bounds.
    #[error("request exceeds a supported size, offset, or count limit")]
    OutOfRange,
    /// A required dimension, layer, level, or member count is zero.
    #[error("required dimension or count is zero")]
    Empty,
    /// The resource is not in a state permitting this operation.
    #[error("resource is not in a state permitting this operation")]
    BadState,
    /// The resource does not match its layout, format, or declared use.
    #[error("resource does not match its layout, format, or declared use")]
    Mismatch,
    /// The resource memory domain does not permit this host access.
    #[error("memory domain does not permit this host access")]
    NotHostAccessible,
    /// Input contains data the backend cannot represent natively.
    #[error("input contains data that cannot be represented")]
    Malformed,
}

impl InvalidResourceKind {
    /// Attaches diagnostic detail to this classification.
    #[must_use]
    pub fn with_detail(self, detail: impl Into<String>) -> InvalidResource {
        InvalidResource::new(self, detail)
    }
}

impl From<InvalidResourceKind> for Error {
    fn from(kind: InvalidResourceKind) -> Self {
        kind.with_detail("invalid portable RHI operation").into()
    }
}
