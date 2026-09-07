//! Semantic access, ranges, portable limits, and shader binding agreement.
#![allow(
    clippy::missing_errors_doc,
    reason = "checked helpers return typed range, layout, or size errors"
)]
use crate::{
    BufferUsages, Extent3d, ImageAspects, ImageUsages, InvalidResourceKind as Ir, Result,
    ShaderStages, TextureFormat,
};
use std::num::NonZeroU32;

/// Portable device limits checked before native resource creation.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum buffer allocation in bytes.
    pub max_buffer_size: u64,
    /// Maximum width/height of a 2D image.
    pub max_image_dimension_2d: u32,
    /// Maximum dimension of a 3D image.
    pub max_image_dimension_3d: u32,
    /// Maximum image array layers.
    pub max_image_array_layers: u32,
    /// Maximum color attachments in a pass.
    pub max_color_attachments: u32,
    /// Maximum resource groups in a pipeline.
    pub max_bind_groups: u32,
    /// Maximum shader buffers per stage.
    pub max_shader_buffers: u32,
    /// Maximum shader textures per stage.
    pub max_shader_textures: u32,
    /// Maximum shader samplers per stage.
    pub max_shader_samplers: u32,
    /// Maximum vertex buffer slots.
    pub max_vertex_buffers: u32,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_buffer_size: 1 << 27,
            max_image_dimension_2d: 4096,
            max_image_dimension_3d: 256,
            max_image_array_layers: 256,
            max_color_attachments: 4,
            max_bind_groups: 4,
            max_shader_buffers: 16,
            max_shader_textures: 16,
            max_shader_samplers: 16,
            max_vertex_buffers: 8,
        }
    }
}

/// A checked byte range. The whole-remainder convention is shared by bindings and barriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRange {
    /// First byte.
    pub offset: u64,
    /// Byte count; `u64::MAX` selects the remainder.
    pub size: u64,
}
impl BufferRange {
    /// All bytes.
    pub const WHOLE: Self = Self {
        offset: 0,
        size: u64::MAX,
    };
    /// Resolves the remainder and rejects empty or out-of-bounds ranges.
    pub fn resolve(self, total: u64) -> Result<Self> {
        let size = if self.size == u64::MAX {
            total.checked_sub(self.offset)
        } else {
            Some(self.size)
        }
        .ok_or(Ir::OutOfRange)?;
        if size == 0 || self.offset.checked_add(size).is_none_or(|end| end > total) {
            return Err(Ir::OutOfRange.into());
        }
        Ok(Self {
            offset: self.offset,
            size,
        })
    }
}

/// Image aspects and mip/layer selection shared by views, barriers, and graph declarations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSubresourceRange {
    /// Selected aspects; empty selects all format aspects.
    pub aspects: ImageAspects,
    /// First mip.
    pub base_mip_level: u32,
    /// Mip count; `u32::MAX` selects the remainder.
    pub mip_level_count: u32,
    /// First layer.
    pub base_array_layer: u32,
    /// Layer count; `u32::MAX` selects the remainder.
    pub array_layer_count: u32,
}
impl ImageSubresourceRange {
    /// All subresources.
    pub const WHOLE: Self = Self {
        aspects: ImageAspects::NONE,
        base_mip_level: 0,
        mip_level_count: u32::MAX,
        base_array_layer: 0,
        array_layer_count: u32::MAX,
    };
    /// Selects one mip and all layers/aspects.
    #[must_use]
    pub const fn mip(level: u32) -> Self {
        Self {
            base_mip_level: level,
            mip_level_count: 1,
            ..Self::WHOLE
        }
    }
    /// Checks and resolves against allocation metadata.
    pub fn resolve(self, info: &ImageInfo) -> Result<Self> {
        let mips = BufferRange {
            offset: u64::from(self.base_mip_level),
            size: if self.mip_level_count == u32::MAX {
                u64::MAX
            } else {
                u64::from(self.mip_level_count)
            },
        }
        .resolve(u64::from(info.mip_levels))?;
        let layers = BufferRange {
            offset: u64::from(self.base_array_layer),
            size: if self.array_layer_count == u32::MAX {
                u64::MAX
            } else {
                u64::from(self.array_layer_count)
            },
        }
        .resolve(u64::from(info.array_layers))?;
        let aspects = if self.aspects.is_empty() {
            info.format.aspects()
        } else {
            self.aspects
        };
        if !info.format.aspects().contains(aspects) {
            return Err(Ir::Mismatch.into());
        }
        Ok(Self {
            aspects,
            mip_level_count: u32::try_from(mips.size).map_err(|_| Ir::OutOfRange)?,
            array_layer_count: u32::try_from(layers.size).map_err(|_| Ir::OutOfRange)?,
            ..self
        })
    }
}

/// Immutable image allocation metadata, also supplied for acquired surface images.
#[derive(Clone, Copy, Debug)]
pub struct ImageInfo {
    /// Dimensionality.
    pub dimension: crate::ImageDimension,
    /// Base mip dimensions.
    pub extent: Extent3d,
    /// Texel format.
    pub format: TextureFormat,
    /// Permitted uses.
    pub usage: ImageUsages,
    /// Number of mips.
    pub mip_levels: u32,
    /// Number of layers.
    pub array_layers: u32,
    /// Sample count.
    pub samples: crate::SampleCount,
}
impl From<&crate::ImageDesc<'_>> for ImageInfo {
    fn from(d: &crate::ImageDesc<'_>) -> Self {
        Self {
            dimension: d.dimension,
            extent: d.extent,
            format: d.format,
            usage: d.usage,
            mip_levels: d.mip_levels,
            array_layers: d.array_layers,
            samples: d.samples,
        }
    }
}
impl ImageInfo {
    /// Extent of an existing mip.
    pub fn mip_extent(self, mip: u32) -> Result<Extent3d> {
        if mip >= self.mip_levels {
            return Err(Ir::OutOfRange.into());
        }
        Ok(Extent3d {
            width: (self.extent.width >> mip).max(1),
            height: (self.extent.height >> mip).max(1),
            depth: (self.extent.depth >> mip).max(1),
        })
    }
}
impl TextureFormat {
    /// Aspects stored by this format.
    #[must_use]
    pub const fn aspects(self) -> ImageAspects {
        match self {
            Self::Depth16Unorm | Self::Depth32Float => ImageAspects::DEPTH,
            Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8 => {
                ImageAspects::DEPTH.union(ImageAspects::STENCIL)
            }
            _ => ImageAspects::COLOR,
        }
    }
}

/// Semantic access shared by images and buffers. Stages survive graph compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    /// No preserved contents.
    #[default]
    Undefined,
    /// Transfer source.
    CopySource,
    /// Transfer destination.
    CopyDestination,
    /// Vertex attribute fetch.
    Vertex,
    /// Index fetch.
    Index,
    /// Uniform buffer reads in the given shader stages.
    Uniform(ShaderStages),
    /// Sampled texture reads in the given stages.
    ShaderRead(ShaderStages),
    /// Read-only storage access.
    StorageRead(ShaderStages),
    /// Storage reads and writes.
    ShaderWrite(ShaderStages),
    /// Color attachment access.
    ColorAttachment,
    /// Writable depth/stencil access.
    DepthStencilAttachment,
    /// Read-only depth/stencil access, optionally also sampled by these stages.
    DepthStencilAttachmentReadOnly(ShaderStages),
    /// Display ownership.
    Present,
}
impl ResourceAccess {
    /// Whether this access may write contents.
    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(
            self,
            Self::CopyDestination
                | Self::ShaderWrite(_)
                | Self::ColorAttachment
                | Self::DepthStencilAttachment
        )
    }
    /// Image creation capabilities required by this access.
    #[must_use]
    pub fn image_usage(self) -> ImageUsages {
        match self {
            Self::CopySource => ImageUsages::COPY_SRC,
            Self::CopyDestination => ImageUsages::COPY_DST,
            Self::ShaderRead(_) => ImageUsages::SAMPLED,
            Self::StorageRead(_) | Self::ShaderWrite(_) => ImageUsages::STORAGE,
            Self::ColorAttachment => ImageUsages::COLOR_ATTACHMENT,
            Self::DepthStencilAttachment => ImageUsages::DEPTH_STENCIL_ATTACHMENT,
            Self::DepthStencilAttachmentReadOnly(stages) => {
                ImageUsages::DEPTH_STENCIL_ATTACHMENT
                    | if stages.is_empty() {
                        ImageUsages::NONE
                    } else {
                        ImageUsages::SAMPLED
                    }
            }
            Self::Present => ImageUsages::PRESENT,
            _ => ImageUsages::NONE,
        }
    }
    /// Buffer creation capabilities required by this access.
    #[must_use]
    pub const fn buffer_usage(self) -> BufferUsages {
        match self {
            Self::CopySource => BufferUsages::COPY_SRC,
            Self::CopyDestination => BufferUsages::COPY_DST,
            Self::Vertex => BufferUsages::VERTEX,
            Self::Index => BufferUsages::INDEX,
            Self::Uniform(_) => BufferUsages::UNIFORM,
            Self::StorageRead(_) | Self::ShaderWrite(_) => BufferUsages::STORAGE,
            _ => BufferUsages::NONE,
        }
    }
}

/// Aligned layout for tightly packed 2D upload pixels.
#[derive(Clone, Copy, Debug)]
pub struct UploadLayout {
    /// Unpadded row byte count.
    pub row_bytes: u32,
    /// Native-compatible row pitch.
    pub bytes_per_row: NonZeroU32,
    /// Number of rows.
    pub rows: NonZeroU32,
}
impl UploadLayout {
    /// Computes checked row padding using selected-device capabilities.
    pub fn new(
        width: u32,
        height: u32,
        format: TextureFormat,
        caps: crate::Capabilities,
    ) -> Result<Self> {
        let row_bytes = width
            .checked_mul(format.texel_size())
            .filter(|n| *n > 0)
            .ok_or(Ir::OutOfRange)?;
        let alignment = caps.buffer_copy_row_pitch_alignment.max(1);
        let pitch = row_bytes
            .checked_add(alignment - 1)
            .and_then(|n| (n / alignment).checked_mul(alignment))
            .and_then(NonZeroU32::new)
            .ok_or(Ir::OutOfRange)?;
        Ok(Self {
            row_bytes,
            bytes_per_row: pitch,
            rows: NonZeroU32::new(height).ok_or(Ir::Empty)?,
        })
    }
    /// Copies tightly packed rows into an aligned staging payload.
    pub fn pack(self, pixels: &[u8]) -> Result<Vec<u8>> {
        let packed = u64::from(self.row_bytes) * u64::from(self.rows.get());
        if u64::try_from(pixels.len()).map_err(|_| Ir::OutOfRange)? != packed {
            return Err(Ir::Mismatch.into());
        }
        let size =
            usize::try_from(u64::from(self.bytes_per_row.get()) * u64::from(self.rows.get()))
                .map_err(|_| Ir::OutOfRange)?;
        let mut output = vec![0; size];
        for (src, dst) in pixels
            .chunks_exact(self.row_bytes as usize)
            .zip(output.chunks_exact_mut(self.bytes_per_row.get() as usize))
        {
            dst[..src.len()].copy_from_slice(src);
        }
        Ok(output)
    }
}

/// Native shader slots for one portable group binding in one shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingSlots {
    /// Buffer slot, when this is a buffer binding.
    pub buffer: Option<u32>,
    /// Texture slot, when this is an image binding.
    pub texture: Option<u32>,
    /// Sampler slot, when this is a sampled-image pair.
    pub sampler: Option<u32>,
}
/// Authoritative per-stage binding map shared by shader translation and native layouts.
/// Different shader stages have independent native slot namespaces. Only bindings
/// visible to the requested stage contribute to its map, including sparse group indices.
#[derive(Clone, Debug)]
pub struct BindingMap {
    entries: std::collections::BTreeMap<(u32, u32), BindingSlots>,
}
impl BindingMap {
    /// Builds a deterministic map from sorted portable groups and stage visibility.
    pub fn new(
        groups: &[&[crate::BindGroupLayoutEntry]],
        stage: crate::ShaderStage,
    ) -> Result<Self> {
        let visible = match stage {
            crate::ShaderStage::Vertex => ShaderStages::VERTEX,
            crate::ShaderStage::Fragment => ShaderStages::FRAGMENT,
            crate::ShaderStage::Compute => ShaderStages::COMPUTE,
        };
        let mut entries = std::collections::BTreeMap::new();
        let mut buffers = 0u32;
        let mut textures = 0u32;
        let mut samplers = 0u32;
        for (group, bindings) in groups.iter().enumerate() {
            let mut sorted = bindings.to_vec();
            sorted.sort_by_key(|b| b.binding);
            for binding in sorted {
                if !binding.visibility.intersects(visible) {
                    continue;
                }
                let mut slots = BindingSlots {
                    buffer: None,
                    texture: None,
                    sampler: None,
                };
                match binding.ty {
                    crate::BindingType::UniformBuffer { .. }
                    | crate::BindingType::StorageBuffer { .. } => {
                        slots.buffer = Some(buffers);
                        buffers = buffers.checked_add(1).ok_or(Ir::OutOfRange)?;
                    }
                    crate::BindingType::SampledImage => {
                        slots.texture = Some(textures);
                        slots.sampler = Some(samplers);
                        textures = textures.checked_add(1).ok_or(Ir::OutOfRange)?;
                        samplers = samplers.checked_add(1).ok_or(Ir::OutOfRange)?;
                    }
                    crate::BindingType::StorageImage => {
                        slots.texture = Some(textures);
                        textures = textures.checked_add(1).ok_or(Ir::OutOfRange)?;
                    }
                }
                if entries
                    .insert(
                        (
                            u32::try_from(group).map_err(|_| Ir::OutOfRange)?,
                            binding.binding,
                        ),
                        slots,
                    )
                    .is_some()
                {
                    return Err(Ir::Mismatch.into());
                }
            }
        }
        Ok(Self { entries })
    }
    /// Gets slots for a visible binding. Invisible or undeclared bindings return `None`.
    #[must_use]
    pub fn get(&self, group: u32, binding: u32) -> Option<BindingSlots> {
        self.entries.get(&(group, binding)).copied()
    }
    /// Checks the map against the selected device's per-stage limits.
    pub fn validate(&self, limits: Limits) -> Result<()> {
        for slots in self.entries.values() {
            if slots.buffer.is_some_and(|n| n >= limits.max_shader_buffers)
                || slots
                    .texture
                    .is_some_and(|n| n >= limits.max_shader_textures)
                || slots
                    .sampler
                    .is_some_and(|n| n >= limits.max_shader_samplers)
            {
                return Err(Ir::OutOfRange.into());
            }
        }
        Ok(())
    }
}

/// Supported filtering for exact regional image blits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlitSupport {
    /// No regional blit implementation.
    #[default]
    None,
    /// Nearest filtering only.
    Nearest,
    /// Nearest and linear filtering.
    Linear,
}
impl BlitSupport {
    /// Whether the requested filter preserves exact regional blit semantics.
    #[must_use]
    pub fn supports(self, filter: crate::FilterMode) -> bool {
        self == Self::Linear || (self == Self::Nearest && filter == crate::FilterMode::Nearest)
    }
}
