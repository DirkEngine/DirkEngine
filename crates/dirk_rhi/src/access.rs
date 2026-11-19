//! How resources are accessed, which barriers translate into precise native
//! synchronization.

use bitflags::bitflags;

use crate::{BufferUsage, ImageUsage};

bitflags! {
    /// Shader stages that access a resource.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct ShaderStages: u8 {
        /// Vertex shaders.
        const VERTEX = 1 << 0;
        /// Fragment shaders.
        const FRAGMENT = 1 << 1;
        /// Compute shaders.
        const COMPUTE = 1 << 2;
    }
}

/// How a buffer is accessed before or after a barrier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Not accessed: before the first access, or contents to be discarded.
    #[default]
    None,
    /// Read by copies.
    CopySource,
    /// Written by copies.
    CopyDestination,
    /// Read as an index buffer.
    Index,
    /// Read as a storage buffer by the given stages.
    ShaderRead(ShaderStages),
    /// Read and written as a storage buffer by the given stages.
    ShaderWrite(ShaderStages),
    /// Read by the host after the work completes.
    HostRead,
}

/// How an image is accessed before or after a barrier. Each access implies
/// the image layout it needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageAccess {
    /// Not accessed: before the first access, or contents to be discarded.
    /// Only valid before a barrier.
    #[default]
    Undefined,
    /// Read by copies and blits.
    CopySource,
    /// Written by copies and blits.
    CopyDestination,
    /// Sampled by the given stages.
    Sampled(ShaderStages),
    /// Read as a storage image by the given stages.
    StorageRead(ShaderStages),
    /// Read and written as a storage image by the given stages.
    StorageWrite(ShaderStages),
    /// Read and written as a color attachment.
    ColorAttachment,
    /// Read and written as a depth/stencil attachment.
    DepthStencilAttachment,
    /// Read as a depth/stencil attachment and sampled by the given stages,
    /// which may be empty.
    DepthStencilRead(ShaderStages),
    /// Owned by presentation: before a barrier, a swapchain image just
    /// acquired, whose contents are discarded; after it, ready to present.
    /// The first access after acquiring must be an attachment write or a
    /// copy.
    Present,
}

impl BufferAccess {
    /// Whether the copy queue can perform the access.
    pub(crate) const fn on_copy_queue(self) -> bool {
        matches!(
            self,
            Self::None | Self::CopySource | Self::CopyDestination | Self::HostRead
        )
    }

    /// Whether the access writes the buffer.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::CopyDestination | Self::ShaderWrite(_))
    }

    /// The stages of a shader access, which must name at least one.
    #[must_use]
    pub const fn shader_stages(self) -> Option<ShaderStages> {
        match self {
            Self::ShaderRead(stages) | Self::ShaderWrite(stages) => Some(stages),
            _ => None,
        }
    }

    /// The usage the buffer needs for this access.
    #[must_use]
    pub const fn usage(self) -> BufferUsage {
        match self {
            Self::None | Self::HostRead => BufferUsage::empty(),
            Self::CopySource => BufferUsage::COPY_SRC,
            Self::CopyDestination => BufferUsage::COPY_DST,
            Self::Index => BufferUsage::INDEX,
            Self::ShaderRead(_) | Self::ShaderWrite(_) => BufferUsage::STORAGE,
        }
    }
}

impl ImageAccess {
    /// Whether the copy queue can perform the access.
    pub(crate) const fn on_copy_queue(self) -> bool {
        matches!(
            self,
            Self::Undefined | Self::CopySource | Self::CopyDestination
        )
    }

    /// Whether the access writes the image.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(
            self,
            Self::CopyDestination
                | Self::StorageWrite(_)
                | Self::ColorAttachment
                | Self::DepthStencilAttachment
        )
    }

    /// The stages of a shader access, which must name at least one.
    /// [`ImageAccess::DepthStencilRead`] is not one: its stages may be empty.
    #[must_use]
    pub const fn shader_stages(self) -> Option<ShaderStages> {
        match self {
            Self::Sampled(stages) | Self::StorageRead(stages) | Self::StorageWrite(stages) => {
                Some(stages)
            }
            _ => None,
        }
    }

    /// The usage the image needs for this access.
    #[must_use]
    pub const fn usage(self) -> ImageUsage {
        match self {
            Self::Undefined | Self::Present => ImageUsage::empty(),
            Self::CopySource => ImageUsage::COPY_SRC,
            Self::CopyDestination => ImageUsage::COPY_DST,
            Self::Sampled(_) => ImageUsage::SAMPLED,
            Self::StorageRead(_) | Self::StorageWrite(_) => ImageUsage::STORAGE,
            Self::ColorAttachment => ImageUsage::COLOR_ATTACHMENT,
            Self::DepthStencilAttachment => ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            Self::DepthStencilRead(stages) if stages.is_empty() => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT
            }
            Self::DepthStencilRead(_) => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT.union(ImageUsage::SAMPLED)
            }
        }
    }
}
