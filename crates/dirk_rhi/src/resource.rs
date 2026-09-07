use crate::{
    AddressMode, Backend, BindingType, BlendFactor, BlendOp, BufferUsages, ColorWrites, CompareOp,
    CullMode, Extent3d, FilterMode, FrontFace, ImageAspects, ImageDimension, ImageUsages,
    ImageViewType, MemoryDomain, PrimitiveTopology, Result, SampleCount, ShaderStage, ShaderStages,
    StencilOp, TextureFormat, VertexBufferLayout, error::ShaderLanguage,
};

/// Backend buffer with host access when its memory domain permits it.
pub trait Buffer: Clone + Send + Sync + 'static {
    /// Returns this buffer's allocation size in bytes.
    fn size(&self) -> u64;

    /// Writes bytes into this buffer's host-visible memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer is not host-visible or the requested
    /// range falls outside its allocation.
    ///
    /// # Ordering and visibility
    ///
    /// Implementations must serialize overlapping host operations made through
    /// cloned handles. They must also prevent a safe host write from racing
    /// native GPU access, either by staging the write, waiting, or returning
    /// [`InvalidResourceKind::BadState`](crate::InvalidResourceKind::BadState).
    ///
    /// Implementations present host-visible memory as coherent from the
    /// caller's perspective; any cache flush or invalidation required by the
    /// native memory type is handled inside the backend.
    fn write(&self, offset: u64, data: &[u8]) -> Result<()>;

    /// Reads bytes from this buffer's host-visible memory.
    ///
    /// Implementations must serialize overlapping host operations and wait for
    /// or reject pending GPU writes. Any native cache invalidation is handled
    /// by the backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer is not host-readable, the range is out
    /// of bounds, or GPU work still owns the range.
    fn read(&self, offset: u64, data: &mut [u8]) -> Result<()>;
}

/// Buffer allocation description.
#[derive(Clone, Copy, Debug)]
pub struct BufferDesc<'a> {
    /// Debug label.
    pub label: &'a str,
    /// Buffer size in bytes.
    pub size: u64,
    /// Permitted buffer uses.
    pub usage: BufferUsages,
    /// Preferred host/GPU access pattern.
    pub memory: MemoryDomain,
}

/// Image allocation description.
#[derive(Clone, Copy, Debug)]
pub struct ImageDesc<'a> {
    /// Debug label.
    pub label: &'a str,
    /// Image dimensionality and view compatibility.
    pub dimension: ImageDimension,
    /// Image dimensions. `TwoD` and `Cube` images require a depth of one;
    /// `ThreeD` images require one array layer.
    pub extent: Extent3d,
    /// Texel format.
    pub format: TextureFormat,
    /// Permitted image uses.
    pub usage: ImageUsages,
    /// Mip level count.
    pub mip_levels: u32,
    /// Array layer count. Cube-compatible images require a multiple of six.
    pub array_layers: u32,
    /// Multisampling count.
    pub samples: SampleCount,
}

/// Image-view description.
pub struct ImageViewDesc<'a, B: Backend> {
    /// Debug label.
    pub label: &'a str,
    /// Image being viewed.
    pub image: &'a B::Image,
    /// View dimensionality.
    pub view_type: ImageViewType,
    /// Selected image aspects.
    pub aspects: ImageAspects,
    /// First visible mip level.
    pub base_mip_level: u32,
    /// Visible mip level count.
    pub mip_level_count: u32,
    /// First visible array layer.
    pub base_array_layer: u32,
    /// Visible array layer count.
    pub array_layer_count: u32,
}

/// Texture sampler description.
#[derive(Clone, Copy, Debug)]
pub struct SamplerDesc<'a> {
    /// Debug label.
    pub label: &'a str,
    /// Magnification filtering.
    pub mag_filter: FilterMode,
    /// Minification filtering.
    pub min_filter: FilterMode,
    /// Mipmap filtering.
    pub mip_filter: FilterMode,
    /// Horizontal address mode.
    pub address_u: AddressMode,
    /// Vertical address mode.
    pub address_v: AddressMode,
    /// Depth address mode.
    pub address_w: AddressMode,
    /// Maximum anisotropy, where one disables anisotropic filtering.
    pub max_anisotropy: u16,
    /// Minimum mip level.
    pub lod_min: f32,
    /// Maximum mip level.
    pub lod_max: f32,
}

/// Shader source accepted by graphics backends.
///
/// This surface is expected to be overhauled alongside the planned shader
/// handling rework: a unified translation pipeline will normalize source
/// languages, add specialization, and derive bind-group layouts from
/// reflection rather than hand-maintained descriptors. The refactor must also
/// establish one authoritative shader/backend binding map and explicit
/// coordinate conventions; see `docs/rhi-shaders.md` in the repository.
/// Until then, backends accept only the representations they consume natively
/// and reject others
/// with [`crate::Error::Unsupported`] and the received
/// [`ShaderLanguage`], so callers can fall back or fail loudly.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// SPIR-V words generated by the engine shader build.
    SpirV(&'a [u32]),
    /// Metal Shading Language source for native Metal specialization.
    Msl(&'a str),
}

impl ShaderSource<'_> {
    /// Returns the source representation carried by this shader.
    #[must_use]
    pub const fn language(&self) -> ShaderLanguage {
        match self {
            Self::SpirV(_) => ShaderLanguage::SpirV,
            Self::Msl(_) => ShaderLanguage::Msl,
        }
    }
}

/// Shader module description.
#[derive(Clone, Copy, Debug)]
pub struct ShaderDesc<'a> {
    /// Debug label.
    pub label: &'a str,
    /// Programmable stage.
    pub stage: ShaderStage,
    /// Entry-point function name.
    pub entry: &'a str,
    /// Shader source.
    pub source: ShaderSource<'a>,
}

/// One resource binding in a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutEntry {
    /// Binding index.
    pub binding: u32,
    /// Bound resource type.
    pub ty: BindingType,
    /// Shader stages allowed to access the binding.
    pub visibility: ShaderStages,
}

/// Bind-group layout description.
#[derive(Clone, Copy, Debug)]
pub struct BindGroupLayoutDesc<'a> {
    /// Debug label.
    pub label: &'a str,
    /// Binding declarations.
    pub entries: &'a [BindGroupLayoutEntry],
}

/// Resource written into a bind group.
pub enum BindingResource<'a, B: Backend> {
    /// Byte range of a buffer.
    ///
    /// The offset is a static base offset. Layout entries with dynamic offsets
    /// add their bind-time offset to this value.
    Buffer {
        /// Buffer resource.
        buffer: &'a B::Buffer,
        /// Byte offset.
        offset: u64,
        /// Byte size.
        size: u64,
    },
    /// Sampled image view and sampler.
    SampledImage {
        /// Image view.
        view: &'a B::ImageView,
        /// Texture sampler.
        sampler: &'a B::Sampler,
    },
    /// Storage image view.
    StorageImage(&'a B::ImageView),
}

/// One bind-group resource entry.
pub struct BindGroupEntry<'a, B: Backend> {
    /// Binding index from the layout.
    pub binding: u32,
    /// Bound resource.
    pub resource: BindingResource<'a, B>,
}

/// Bind-group creation description.
pub struct BindGroupDesc<'a, B: Backend> {
    /// Debug label.
    pub label: &'a str,
    /// Layout implemented by the bind group.
    pub layout: &'a B::BindGroupLayout,
    /// Resources written into the group.
    pub entries: &'a [BindGroupEntry<'a, B>],
}

/// Pipeline-layout description.
pub struct PipelineLayoutDesc<'a, B: Backend> {
    /// Debug label.
    pub label: &'a str,
    /// Bind-group layouts ordered by set/group index.
    pub bind_group_layouts: &'a [&'a B::BindGroupLayout],
}

/// Rasterization state.
#[derive(Clone, Copy, Debug, Default)]
pub struct RasterState {
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Front-face winding.
    pub front_face: FrontFace,
    /// Face culling mode.
    pub cull_mode: CullMode,
}

/// Depth attachment pipeline state.
#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    /// Depth texture format.
    pub format: TextureFormat,
    /// Whether fragment depth is written.
    pub write_enabled: bool,
    /// Depth comparison operation.
    pub compare: CompareOp,
    /// Optional stencil state applied when the attachment has stencil
    /// aspects.
    pub stencil: Option<StencilState>,
}

/// Stencil state shared by both faces of a depth/stencil attachment.
#[derive(Clone, Copy, Debug)]
pub struct StencilState {
    /// Front-facing polygon operations.
    pub front: StencilFaceState,
    /// Back-facing polygon operations.
    pub back: StencilFaceState,
    /// Read mask applied to stored stencil values before comparison.
    pub read_mask: u32,
    /// Mask limiting which bits write operations may modify.
    pub write_mask: u32,
}

/// Comparison and update operations for one stencil face.
///
/// Reference values are supplied at recording time through
/// [`crate::CommandBuffer::set_stencil_reference`].
#[derive(Clone, Copy, Debug)]
pub struct StencilFaceState {
    /// Comparison between the reference value and the stored value.
    pub compare: CompareOp,
    /// Operation applied when the stencil test fails.
    pub fail_op: StencilOp,
    /// Operation applied when the stencil test passes but the depth test
    /// fails.
    pub depth_fail_op: StencilOp,
    /// Operation applied when both the stencil and depth tests pass.
    pub pass_op: StencilOp,
}

/// Depth bias applied to rasterized primitives.
///
/// Zero factors leave biasing disabled; backends enable their native bias
/// path only when a factor is non-zero. `clamp` is ignored by devices that
/// do not support depth bias clamping.
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthBiasState {
    /// Constant factor added to each fragment's depth.
    pub constant_factor: f32,
    /// Factor scaled by the fragment's depth slope.
    pub slope_factor: f32,
    /// Maximum representable depth bias, where clamping is supported.
    pub clamp: f32,
}

/// One color or alpha blend equation.
#[derive(Clone, Copy, Debug)]
pub struct BlendComponent {
    /// Multiplier applied to the shader output.
    pub source: BlendFactor,
    /// Multiplier applied to the existing attachment value.
    pub destination: BlendFactor,
    /// Operation combining the multiplied terms.
    pub operation: BlendOp,
}

/// Color attachment blending state for one pipeline color target.
#[derive(Clone, Copy, Debug)]
pub struct BlendState {
    /// RGB blend equation.
    pub color: BlendComponent,
    /// Alpha blend equation.
    pub alpha: BlendComponent,
}

/// Format, blending, and write mask for one graphics color target.
#[derive(Clone, Copy, Debug)]
pub struct ColorTargetState {
    /// Target texture format.
    pub format: TextureFormat,
    /// Optional blending for this target.
    pub blend: Option<BlendState>,
    /// Color channels written by the target.
    pub write_mask: ColorWrites,
}

/// Graphics pipeline description.
pub struct GraphicsPipelineDesc<'a, B: Backend> {
    /// Debug label.
    pub label: &'a str,
    /// Pipeline resource layout.
    pub layout: &'a B::PipelineLayout,
    /// Vertex shader.
    pub vertex: &'a B::Shader,
    /// Optional fragment shader. Depth-only passes leave this unset.
    pub fragment: Option<&'a B::Shader>,
    /// Vertex-buffer layouts.
    pub vertex_buffers: &'a [VertexBufferLayout<'a>],
    /// Rasterization state.
    pub raster: RasterState,
    /// Color attachment state ordered by target index.
    pub color_targets: &'a [ColorTargetState],
    /// Optional depth state.
    pub depth: Option<DepthState>,
    /// Depth bias applied to rasterized primitives.
    pub depth_bias: DepthBiasState,
    /// Index format whose maximum value restarts triangle or line strips.
    /// `None` disables primitive restart.
    pub primitive_restart: Option<crate::IndexFormat>,
    /// Whether fragment alpha is used as the multisample coverage mask.
    pub alpha_to_coverage: bool,
    /// Multisampling count.
    pub samples: SampleCount,
}
