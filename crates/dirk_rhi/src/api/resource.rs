use super::{Backend, Ir, Object, Result};
use crate::{
    BufferUsages, ImageInfo, ImageSubresourceRange, MemoryDomain, NativeBuffer as _, ShaderStage,
};

/// Immutable buffer allocation information.
#[derive(Clone, Copy, Debug)]
pub struct BufferInfo {
    /// Allocation size.
    pub size: u64,
    /// Permitted uses.
    pub usage: BufferUsages,
    /// Host access domain.
    pub memory: MemoryDomain,
}
/// Unique buffer with checked host access.
pub type GpuBuffer<B> = Object<B, <B as crate::Api>::Buffer, BufferInfo>;
/// Unique image with portable allocation metadata.
pub type GpuImage<B> = Object<B, <B as crate::Api>::Image, ImageInfo>;
/// Unique view; image ownership remains with the caller.
pub type GpuImageView<B> = Object<B, <B as crate::Api>::ImageView, ViewMetadata>;
/// Unique sampler.
pub type GpuSampler<B> = Object<B, <B as crate::Api>::Sampler, SamplerInfo>;
/// Shader module with its declared stage.
pub type GpuShader<B> = Object<B, <B as crate::Api>::Shader, ShaderStage>;
/// Immutable group layout.
pub type GpuBindGroupLayout<B> =
    Object<B, <B as crate::Api>::BindGroupLayout, Vec<crate::BindGroupLayoutEntry>>;
/// Immutable group with non-owning native bindings.
pub type GpuBindGroup<B> = Object<B, <B as crate::Api>::BindGroup, GroupMetadata>;
/// Ordered group layouts.
pub type GpuPipelineLayout<B> =
    Object<B, <B as crate::Api>::PipelineLayout, Vec<Vec<crate::BindGroupLayoutEntry>>>;
/// Graphics pipeline with immutable render compatibility metadata.
pub type GpuGraphicsPipeline<B> = Object<B, <B as crate::Api>::GraphicsPipeline, PipelineMetadata>;
/// Retained native surface.
pub type GpuSurface<B> = Object<B, <B as crate::Api>::Surface, ()>;
impl<B: Backend> GpuImage<B> {
    /// Portable allocation description.
    #[must_use]
    pub fn description(&self) -> ImageInfo {
        *self.info()
    }
}
/// Image view selection. The image owner must outlive all uses of this view.
#[derive(Clone, Copy, Debug)]
pub struct ViewMetadata {
    /// Source allocation description, not ownership.
    pub image: ImageInfo,
    /// Resolved aspects, mips, and layers.
    pub range: ImageSubresourceRange,
}
/// Filtering properties used to validate texture/sampler pairs.
#[derive(Clone, Copy, Debug)]
pub struct SamplerInfo {
    /// Whether any filtering requires linear format support.
    pub linear: bool,
}
/// Immutable binding declarations; resource ownership remains with the caller.
pub struct GroupMetadata {
    /// Implemented layout.
    pub layout: Vec<crate::BindGroupLayoutEntry>,
}
impl GroupMetadata {
    /// Consumes this group's offsets in ascending layout-binding order.
    pub(super) fn validate_dynamic_offsets<'a>(
        &self,
        group: u32,
        offsets: &mut impl Iterator<Item = &'a u64>,
        capabilities: crate::Capabilities,
    ) -> crate::Result<()> {
        for entry in &self.layout {
            let alignment = match entry.ty {
                crate::BindingType::UniformBuffer {
                    dynamic_offset: true,
                } => capabilities.min_uniform_buffer_offset_alignment,
                crate::BindingType::StorageBuffer {
                    dynamic_offset: true,
                    ..
                } => capabilities.min_storage_buffer_offset_alignment,
                _ => continue,
            }
            .max(1);
            let offset = offsets.next().ok_or_else(|| {
                crate::InvalidResourceKind::Mismatch.with_detail(format!(
                    "missing dynamic offset for group {group} binding {}",
                    entry.binding
                ))
            })?;
            if !offset.is_multiple_of(alignment) {
                return Err(crate::InvalidResourceKind::Mismatch
                    .with_detail(format!(
                        "dynamic offset for group {group} binding {} must be aligned to {alignment} bytes",
                        entry.binding
                    ))
                    .into());
            }
        }
        Ok(())
    }
}

/// Immutable pipeline render compatibility.
#[derive(Clone)]
pub struct PipelineMetadata {
    /// Required index format when strip restart is enabled.
    pub primitive_restart: Option<crate::IndexFormat>,
    /// Color targets.
    pub colors: Vec<crate::ColorTargetState>,
    /// Depth/stencil state.
    pub depth: Option<crate::DepthState>,
    /// Rasterization sample count.
    pub samples: crate::SampleCount,
}
impl<B: Backend> GpuBuffer<B> {
    /// Allocation size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.info().size
    }
    /// Writes a checked range of host-visible upload memory.
    ///
    /// # Safety
    /// GPU access to these bytes must have completed before the host write.
    pub unsafe fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.host_range(offset, bytes.len(), false)?;
        // SAFETY: range/domain checked; the caller guarantees GPU exclusion and &mut excludes CPU aliases.
        unsafe { self.raw().write(offset, bytes) }
    }
    /// Reads completed readback memory.
    ///
    /// # Safety
    /// Wait for all GPU writes to this range before reading it.
    pub unsafe fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        self.host_range(offset, bytes.len(), true)?;
        // SAFETY: range/domain checked; no pending GPU access or concurrent host operation.
        unsafe { self.raw().read(offset, bytes) }
    }
    fn host_range(&self, offset: u64, length: usize, read: bool) -> Result<()> {
        let domain = if read {
            MemoryDomain::Readback
        } else {
            MemoryDomain::Upload
        };
        if self.info().memory != domain {
            return Err(Ir::NotHostAccessible.into());
        }
        if offset
            .checked_add(u64::try_from(length).map_err(|_| Ir::OutOfRange)?)
            .is_none_or(|end| end > self.size())
        {
            return Err(Ir::OutOfRange.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod dynamic_offset_tests {
    use super::*;

    #[test]
    fn dynamic_offsets_follow_sorted_bindings_and_alignment() {
        let group = GroupMetadata {
            layout: vec![
                crate::BindGroupLayoutEntry {
                    binding: 2,
                    ty: crate::BindingType::StorageBuffer {
                        read_only: true,
                        dynamic_offset: true,
                    },
                    visibility: crate::ShaderStages::FRAGMENT,
                },
                crate::BindGroupLayoutEntry {
                    binding: 7,
                    ty: crate::BindingType::UniformBuffer {
                        dynamic_offset: true,
                    },
                    visibility: crate::ShaderStages::VERTEX,
                },
            ],
        };
        let capabilities = crate::Capabilities {
            limits: crate::Limits::default(),
            depth_bias_clamp: false,
            max_sampler_anisotropy: 1,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 16,
            buffer_copy_offset_alignment: 4,
            buffer_copy_row_pitch_alignment: 4,
            dedicated_compute_queue: false,
            dedicated_copy_queue: false,
        };

        assert!(
            group
                .validate_dynamic_offsets(0, &mut [16, 256].iter(), capabilities)
                .is_ok()
        );
        assert!(
            group
                .validate_dynamic_offsets(0, &mut [256, 16].iter(), capabilities)
                .is_err()
        );
        assert!(
            group
                .validate_dynamic_offsets(0, &mut [16].iter(), capabilities)
                .is_err()
        );
    }
}
