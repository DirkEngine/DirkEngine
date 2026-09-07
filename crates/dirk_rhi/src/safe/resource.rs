use super::{Arc, AtomicU64, Backend, FrameState, Ir, Object, Ordering, Result, lock};
use crate::{
    Buffer as _, BufferUsages, ImageInfo, ImageSubresourceRange, MemoryDomain, ShaderStage,
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
/// Shared buffer with synchronized, nonblocking host access.
pub type GpuBuffer<B> = Object<B, <B as crate::Api>::Buffer, BufferInfo>;
/// Shared image with portable allocation metadata.
pub type GpuImage<B> = Object<B, <B as crate::Api>::Image, ImageMetadata>;
/// Shared view retaining its image.
pub type GpuImageView<B> = Object<B, <B as crate::Api>::ImageView, ViewMetadata<B>>;
/// Shared sampler.
pub type GpuSampler<B> = Object<B, <B as crate::Api>::Sampler, SamplerInfo>;
/// Shader module with its declared stage.
pub type GpuShader<B> = Object<B, <B as crate::Api>::Shader, ShaderStage>;
/// Immutable group layout.
pub type GpuBindGroupLayout<B> =
    Object<B, <B as crate::Api>::BindGroupLayout, Vec<crate::BindGroupLayoutEntry>>;
/// Resource group retaining every bound resource.
pub type GpuBindGroup<B> = Object<B, <B as crate::Api>::BindGroup, GroupMetadata<B>>;
/// Ordered group layouts.
pub type GpuPipelineLayout<B> =
    Object<B, <B as crate::Api>::PipelineLayout, Vec<GpuBindGroupLayout<B>>>;
/// Graphics pipeline retaining its layout and shaders.
pub type GpuGraphicsPipeline<B> =
    Object<B, <B as crate::Api>::GraphicsPipeline, PipelineMetadata<B>>;
/// Retained native surface.
pub type GpuSurface<B> = Object<B, <B as crate::Api>::Surface, ()>;
/// Native timeline controlled through RHI submissions.
pub type GpuTimeline<B> = Object<B, <B as crate::Api>::TimelineSemaphore, AtomicU64>;

/// Image information and optional acquisition lifetime.
#[derive(Debug)]
pub struct ImageMetadata {
    /// Allocation information.
    pub allocation: ImageInfo,
    pub(super) frame: Option<Arc<FrameState>>,
}
impl<B: Backend> GpuImage<B> {
    /// Portable image description.
    #[must_use]
    pub fn description(&self) -> ImageInfo {
        self.info().allocation
    }
    pub(super) fn require_live(&self) -> Result<()> {
        if self
            .info()
            .frame
            .as_ref()
            .is_some_and(|state| state.phase.load(Ordering::Acquire) > 1)
        {
            return Err(Ir::BadState
                .with_detail("surface acquisition has ended")
                .into());
        }
        Ok(())
    }
}
/// Image view selection and ownership.
#[derive(Debug)]
pub struct ViewMetadata<B: Backend> {
    /// Source image.
    pub image: GpuImage<B>,
    /// Resolved aspects, mips, and layers.
    pub range: ImageSubresourceRange,
}
/// Filtering properties used to validate texture/sampler pairs.
#[derive(Clone, Copy, Debug)]
pub struct SamplerInfo {
    /// Whether any filtering requires linear format support.
    pub linear: bool,
}
/// Resource retained by a group.
pub enum OwnedBinding<B: Backend> {
    /// Buffer slice and access type from its layout.
    Buffer(GpuBuffer<B>, crate::BufferRange, crate::BindingType),
    /// Sampled view and sampler.
    Sampled(GpuImageView<B>, GpuSampler<B>),
    /// Storage view.
    Storage(GpuImageView<B>),
}
/// Immutable bound resources.
pub struct GroupMetadata<B: Backend> {
    /// Implemented layout.
    pub layout: GpuBindGroupLayout<B>,
    /// Resources in ascending binding order.
    pub entries: Vec<(u32, OwnedBinding<B>)>,
}
/// Immutable pipeline dependencies and render compatibility.
pub struct PipelineMetadata<B: Backend> {
    /// Pipeline layout.
    pub layout: GpuPipelineLayout<B>,
    /// Vertex module.
    pub vertex: GpuShader<B>,
    /// Optional fragment module.
    pub fragment: Option<GpuShader<B>>,
    /// Required index format when strip restart is enabled.
    pub primitive_restart: Option<crate::IndexFormat>,
    /// Color targets.
    pub colors: Vec<crate::ColorTargetState>,
    /// Depth/stencil state.
    pub depth: Option<crate::DepthState>,
    /// Rasterization sample count.
    pub samples: crate::SampleCount,
    /// Vertex layouts retained for draw range checks.
    pub vertices: Vec<(u64, crate::VertexStepMode, Vec<crate::VertexAttribute>)>,
}
impl<B: Backend> GpuBuffer<B> {
    /// Allocation size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.info().size
    }
    /// Writes host-visible memory, rejecting access while submitted work retains it.
    pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let _gate = lock(&self.0.gate)?;
        self.host_range(offset, bytes.len(), false)?;
        // SAFETY: shared submission gate excludes new GPU use and overlapping host calls.
        unsafe { self.raw().write(offset, bytes) }
    }
    /// Reads completed readback memory. Wait on the submission before calling.
    pub fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        let _gate = lock(&self.0.gate)?;
        self.host_range(offset, bytes.len(), true)?;
        // SAFETY: range/domain checked; no pending GPU access or concurrent host operation.
        unsafe { self.raw().read(offset, bytes) }
    }
    fn host_range(&self, offset: u64, length: usize, read: bool) -> Result<()> {
        if self.0.busy.load(Ordering::Acquire) != 0 {
            return Err(Ir::BadState.with_detail("buffer is in use by a pending submission; wait for completion before host access").into());
        }
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
impl<B: Backend> GpuTimeline<B> {
    /// Waits for a value. Native host access is serialized.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> Result<()> {
        use crate::TimelineSemaphore as _;
        // Native timeline implementations synchronize waits; holding the submission gate
        // during a future-value wait would prevent another thread from submitting its signal.
        unsafe { self.raw().wait(value, timeout_ns) }
    }
    /// Queries native progress.
    pub fn value(&self) -> Result<u64> {
        use crate::TimelineSemaphore as _;
        unsafe { self.raw().value() }
    }
}
