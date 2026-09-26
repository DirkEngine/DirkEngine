use crate::flags::define_flags;

/// Semantic queue used for command execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Rendering and general-purpose graphics commands.
    Graphics,
    /// Compute-only work.
    Compute,
    /// Resource copies and uploads.
    Copy,
}

/// Transfer of exclusive resource ownership between semantic queues.
///
/// Backends that require separate release and acquire dependencies interpret
/// this value according to the queue of the recording command buffer: the
/// source queue releases ownership and the destination queue acquires it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueueTransfer {
    /// Queue that currently owns the resource.
    pub source: QueueType,
    /// Queue that will own the resource after the dependency.
    pub destination: QueueType,
}

/// Dimensionality and allocation compatibility of an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    /// A two-dimensional image or image array.
    #[default]
    TwoD,
    /// A three-dimensional image. Three-dimensional images cannot have array
    /// layers.
    ThreeD,
    /// A cube-compatible two-dimensional image. Array layer counts must be a
    /// multiple of six.
    Cube,
}

/// Three-dimensional unsigned extent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Extent3d {
    /// Width in pixels or buffer elements.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of depth slices.
    ///
    /// Array layers are specified by the descriptor that owns this extent.
    pub depth: u32,
}

/// Three-dimensional unsigned texel origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Origin3d {
    /// Horizontal texel offset.
    pub x: u32,
    /// Vertical texel offset.
    pub y: u32,
    /// Depth-slice offset.
    pub z: u32,
}

impl Extent3d {
    /// Creates a two-dimensional extent with a depth of one.
    #[must_use]
    pub const fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }
}

/// Supported texel formats for images, views, attachments, and swapchains.
///
/// Vertex input uses the separate [`VertexFormat`] enumeration; formats here
/// describe how a texel is stored, not how vertex data is fetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureFormat {
    /// Four-channel normalized RGBA.
    Rgba8Unorm,
    /// Four-channel sRGB RGBA.
    Rgba8Srgb,
    /// Four-channel normalized BGRA.
    Bgra8Unorm,
    /// Four-channel sRGB BGRA.
    Bgra8Srgb,
    /// One 16-bit floating-point channel.
    R16Float,
    /// Two 16-bit floating-point channels.
    Rg16Float,
    /// Four 16-bit floating-point channels.
    ///
    /// The standard high-dynamic-range color attachment format.
    Rgba16Float,
    /// One 32-bit floating-point channel.
    R32Float,
    /// Two 32-bit floating-point channels.
    Rg32Float,
    /// Four 32-bit floating-point channels.
    Rgba32Float,
    /// Packed positive-only floating-point channels for high-dynamic-range
    /// storage at 32 bits per texel.
    R11G11B10Float,
    /// 16-bit normalized depth.
    Depth16Unorm,
    /// 24-bit normalized depth with 8-bit stencil.
    Depth24UnormStencil8,
    /// 32-bit floating-point depth.
    Depth32Float,
    /// 32-bit floating-point depth with 8-bit stencil.
    Depth32FloatStencil8,
}

impl TextureFormat {
    /// Returns the number of bytes occupied by one uncompressed texel block.
    ///
    /// This can be combined with the backend's buffer-copy pitch alignment to
    /// construct [`crate::BufferImageCopy`] layouts.
    #[must_use]
    pub const fn texel_size(self) -> u32 {
        match self {
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rg16Float
            | Self::R32Float
            | Self::R11G11B10Float
            | Self::Depth24UnormStencil8
            | Self::Depth32Float => 4,
            Self::R16Float | Self::Depth16Unorm => 2,
            Self::Rgba16Float | Self::Rg32Float | Self::Depth32FloatStencil8 => 8,
            Self::Rgba32Float => 16,
        }
    }
}

/// Supported vertex attribute formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    /// One 32-bit floating-point component.
    Float32,
    /// Two 32-bit floating-point components.
    Float32x2,
    /// Three 32-bit floating-point components.
    Float32x3,
    /// Four 32-bit floating-point components.
    Float32x4,
    /// Four normalized unsigned 8-bit components.
    Unorm8x4,
    /// Four unsigned 16-bit components.
    Uint16x4,
}

/// Texture sample count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SampleCount {
    /// One sample per pixel.
    #[default]
    One = 1,
    /// Two samples per pixel.
    Two = 2,
    /// Four samples per pixel.
    Four = 4,
    /// Eight samples per pixel.
    Eight = 8,
}

define_flags! {
    /// Supported texture sample counts.
    pub struct SampleCounts(u8) {
        /// One sample per pixel.
        const ONE = 1 << 0;
        /// Two samples per pixel.
        const TWO = 1 << 1;
        /// Four samples per pixel.
        const FOUR = 1 << 2;
        /// Eight samples per pixel.
        const EIGHT = 1 << 3;
    }
}

impl SampleCounts {
    /// Returns whether `count` is present in this set.
    #[must_use]
    pub const fn supports(self, count: SampleCount) -> bool {
        let flag = match count {
            SampleCount::One => Self::ONE,
            SampleCount::Two => Self::TWO,
            SampleCount::Four => Self::FOUR,
            SampleCount::Eight => Self::EIGHT,
        };
        self.contains(flag)
    }
}

/// Preferred memory access pattern for a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    /// GPU-local memory.
    Device,
    /// CPU-written memory used to upload data.
    Upload,
    /// CPU-readable memory used to retrieve GPU results.
    Readback,
}

define_flags! {
    /// Permitted uses of a buffer.
    pub struct BufferUsages(u32) {
        /// Source of a copy operation.
        const COPY_SRC = 1 << 0;
        /// Destination of a copy operation.
        const COPY_DST = 1 << 1;
        /// Vertex input.
        const VERTEX = 1 << 2;
        /// Index input.
        const INDEX = 1 << 3;
        /// Uniform data.
        const UNIFORM = 1 << 4;
        /// Shader storage data.
        const STORAGE = 1 << 5;
    }
}

define_flags! {
    /// Memory accesses made visible by a dependency.
    pub struct AccessTypes(u32) {
        /// Indirect command data reads.
        const INDIRECT_COMMAND_READ = 1 << 0;
        /// Index buffer reads.
        const INDEX_READ = 1 << 1;
        /// Vertex attribute reads.
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        /// Uniform buffer reads.
        const UNIFORM_READ = 1 << 3;
        /// Shader reads from storage resources, textures, or attachments.
        const SHADER_READ = 1 << 4;
        /// Shader writes to storage resources.
        const SHADER_WRITE = 1 << 5;
        /// Color attachment reads.
        const COLOR_ATTACHMENT_READ = 1 << 6;
        /// Color attachment writes.
        const COLOR_ATTACHMENT_WRITE = 1 << 7;
        /// Depth/stencil attachment reads.
        const DEPTH_STENCIL_READ = 1 << 8;
        /// Depth/stencil attachment writes.
        const DEPTH_STENCIL_WRITE = 1 << 9;
        /// Transfer source reads.
        const COPY_READ = 1 << 10;
        /// Transfer destination writes.
        const COPY_WRITE = 1 << 11;
        /// Host reads.
        const HOST_READ = 1 << 12;
        /// Host writes.
        const HOST_WRITE = 1 << 13;
        /// Any memory read when a narrower access cannot be expressed.
        const MEMORY_READ = 1 << 14;
        /// Any memory write when a narrower access cannot be expressed.
        const MEMORY_WRITE = 1 << 15;
    }
}

define_flags! {
    /// Permitted uses of an image.
    pub struct ImageUsages(u32) {
        /// Source of a copy operation.
        const COPY_SRC = 1 << 0;
        /// Destination of a copy operation.
        const COPY_DST = 1 << 1;
        /// Sampled by a shader.
        const SAMPLED = 1 << 2;
        /// Read or written as shader storage.
        const STORAGE = 1 << 3;
        /// Color render target.
        const COLOR_ATTACHMENT = 1 << 4;
        /// Depth or stencil render target.
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
        /// Memory may be transient outside a render pass.
        const TRANSIENT_ATTACHMENT = 1 << 6;
        /// Presented to a display surface.
        const PRESENT = 1 << 7;
    }
}

define_flags! {
    /// Shader stages visible to a binding.
    pub struct ShaderStages(u32) {
        /// Vertex stage.
        const VERTEX = 1 << 0;
        /// Fragment stage.
        const FRAGMENT = 1 << 1;
        /// Compute stage.
        const COMPUTE = 1 << 2;
    }
}

define_flags! {
    /// Pipeline stages used to scope dependencies and submission waits.
    pub struct PipelineStages(u32) {
        /// Indirect argument consumption.
        const INDIRECT = 1 << 0;
        /// Vertex and index input assembly.
        const VERTEX_INPUT = 1 << 1;
        /// Vertex shader execution.
        const VERTEX_SHADER = 1 << 2;
        /// Early depth and stencil tests.
        const EARLY_DEPTH_STENCIL = 1 << 3;
        /// Fragment shader execution.
        const FRAGMENT_SHADER = 1 << 4;
        /// Late depth and stencil tests.
        const LATE_DEPTH_STENCIL = 1 << 5;
        /// Color output.
        const COLOR_OUTPUT = 1 << 6;
        /// Compute shader execution.
        const COMPUTE_SHADER = 1 << 7;
        /// Transfer commands.
        const COPY = 1 << 8;
        /// Host access.
        const HOST = 1 << 9;
    }
}

define_flags! {
    /// Color channels written by a graphics pipeline target.
    pub struct ColorWrites(u8) {
        /// Red channel.
        const RED = 1 << 0;
        /// Green channel.
        const GREEN = 1 << 1;
        /// Blue channel.
        const BLUE = 1 << 2;
        /// Alpha channel.
        const ALPHA = 1 << 3;
    }
}

define_flags! {
    /// Image aspects selected by a view or barrier.
    pub struct ImageAspects(u8) {
        /// Color data.
        const COLOR = 1 << 0;
        /// Depth data.
        const DEPTH = 1 << 1;
        /// Stencil data.
        const STENCIL = 1 << 2;
    }
}

/// Image dimensionality exposed through a view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageViewType {
    /// A two-dimensional image.
    #[default]
    TwoD,
    /// An array of two-dimensional images.
    TwoDArray,
    /// A three-dimensional image.
    ThreeD,
    /// A cube image.
    Cube,
    /// An array of cube images.
    CubeArray,
}

/// Shader stage implemented by a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shader.
    Vertex,
    /// Fragment shader.
    Fragment,
    /// Compute shader.
    Compute,
}

/// Shader-visible resource type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// Read-only uniform buffer.
    UniformBuffer {
        /// Whether a byte offset is supplied when the bind group is bound.
        dynamic_offset: bool,
    },
    /// Storage buffer.
    StorageBuffer {
        /// Whether shaders are restricted to reading this binding.
        read_only: bool,
        /// Whether a byte offset is supplied when the bind group is bound.
        dynamic_offset: bool,
    },
    /// Sampled image and sampler pair.
    SampledImage,
    /// Read-write storage image.
    StorageImage,
}

/// Primitive topology consumed by a graphics pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    /// Independent triangles.
    #[default]
    TriangleList,
    /// Connected triangle strip.
    TriangleStrip,
    /// Independent lines.
    LineList,
}

/// Face winding considered front-facing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Counter-clockwise winding.
    #[default]
    CounterClockwise,
    /// Clockwise winding.
    Clockwise,
}

/// Triangle face culling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// Do not cull triangles.
    None,
    /// Cull front-facing triangles.
    Front,
    /// Cull back-facing triangles.
    #[default]
    Back,
}

/// Depth or stencil comparison operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Comparison always fails.
    Never,
    /// Incoming value is less than stored value.
    #[default]
    Less,
    /// Incoming value is less than or equal to stored value.
    LessEqual,
    /// Incoming value is equal to stored value.
    Equal,
    /// Incoming value is not equal to stored value.
    NotEqual,
    /// Incoming value is greater than stored value.
    Greater,
    /// Incoming value is greater than or equal to stored value.
    GreaterEqual,
    /// Comparison always succeeds.
    Always,
}

/// Index element width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    /// 16-bit unsigned indices.
    Uint16,
    /// 32-bit unsigned indices.
    Uint32,
}

/// Operation applied to stencil bits when a stencil test resolves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StencilOp {
    /// Preserve the current stencil value.
    #[default]
    Keep,
    /// Set the stencil value to zero.
    Zero,
    /// Replace the stencil value with the reference value.
    Replace,
    /// Increment the stencil value, clamping at the maximum.
    IncrementClamp,
    /// Decrement the stencil value, clamping at zero.
    DecrementClamp,
    /// Increment the stencil value, wrapping to zero at the maximum.
    IncrementWrap,
    /// Decrement the stencil value, wrapping to the maximum at zero.
    DecrementWrap,
    /// Bitwise-invert the stencil value.
    Invert,
}

/// Arithmetic operation used when blending color attachments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendOp {
    /// Adds the source and destination terms.
    #[default]
    Add,
    /// Subtracts the destination term from the source term.
    Subtract,
    /// Subtracts the source term from the destination term.
    ReverseSubtract,
    /// Selects the smaller term.
    Min,
    /// Selects the larger term.
    Max,
}

/// Source or destination multiplier used when blending color attachments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// Multiplies by zero.
    Zero,
    /// Multiplies by one.
    #[default]
    One,
    /// Multiplies by the source color channels.
    SourceColor,
    /// Multiplies by one minus the source color channels.
    OneMinusSourceColor,
    /// Multiplies by the source alpha channel.
    SourceAlpha,
    /// Multiplies by one minus the source alpha channel.
    OneMinusSourceAlpha,
    /// Multiplies by the destination color channels.
    DestinationColor,
    /// Multiplies by one minus the destination color channels.
    OneMinusDestinationColor,
    /// Multiplies by the destination alpha channel.
    DestinationAlpha,
    /// Multiplies by one minus the destination alpha channel.
    OneMinusDestinationAlpha,
    /// Multiplies by the dynamic blend constant color channels.
    ConstantColor,
    /// Multiplies by one minus the dynamic blend constant color channels.
    OneMinusConstantColor,
    /// Multiplies by the dynamic blend constant alpha channel.
    ConstantAlpha,
    /// Multiplies by one minus the dynamic blend constant alpha channel.
    OneMinusConstantAlpha,
}

/// Texture filtering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Nearest texel.
    Nearest,
    /// Linear interpolation.
    #[default]
    Linear,
}

/// Texture address mode outside normalized coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Clamp coordinates to the edge.
    #[default]
    ClampToEdge,
    /// Repeat the image.
    Repeat,
    /// Repeat and mirror alternate copies.
    MirrorRepeat,
}

/// Semantic image access; shared with buffer dependency descriptions.
pub type ImageState = crate::ResourceAccess;

/// Rectangle in framebuffer coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Horizontal origin.
    pub x: i32,
    /// Vertical origin.
    pub y: i32,
    /// Rectangle width.
    pub width: u32,
    /// Rectangle height.
    pub height: u32,
}

impl Rect {
    /// Creates a rectangle from its origin and dimensions.
    #[must_use]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Floating-point viewport and depth range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Horizontal origin.
    pub x: f32,
    /// Vertical origin.
    pub y: f32,
    /// Viewport width.
    pub width: f32,
    /// Viewport height.
    pub height: f32,
    /// Minimum depth.
    pub min_depth: f32,
    /// Maximum depth.
    pub max_depth: f32,
}

impl Viewport {
    /// Creates a viewport covering `width` by `height` pixels with the
    /// supplied origin and depth range.
    #[must_use]
    pub const fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        }
    }

    /// Creates a full-range viewport with the given origin and size and a
    /// depth range of zero through one.
    #[must_use]
    pub const fn dimensions(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::new(x, y, width, height, 0.0, 1.0)
    }
}

/// Per-vertex input rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VertexStepMode {
    /// Advance once per vertex.
    #[default]
    Vertex,
    /// Advance once per instance.
    Instance,
}

/// Vertex attribute description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// Shader location.
    pub location: u32,
    /// Attribute format.
    pub format: VertexFormat,
    /// Byte offset within one vertex.
    pub offset: u32,
}

/// Vertex-buffer input layout.
#[derive(Clone, Copy, Debug)]
pub struct VertexBufferLayout<'a> {
    /// Byte stride between elements.
    pub stride: u32,
    /// Element advancement mode.
    pub step_mode: VertexStepMode,
    /// Attributes sourced from this buffer.
    pub attributes: &'a [VertexAttribute],
}

/// Color clear value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
    /// Alpha component.
    pub a: f32,
}

impl Color {
    /// Creates a color from linear components.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    /// Opaque gray at half intensity.
    pub const GRAY: Self = Self::new(0.5, 0.5, 0.5, 1.0);
    /// Opaque red.
    pub const RED: Self = Self::new(1.0, 0.0, 0.0, 1.0);
    /// Opaque green.
    pub const GREEN: Self = Self::new(0.0, 1.0, 0.0, 1.0);
    /// Opaque blue.
    pub const BLUE: Self = Self::new(0.0, 0.0, 1.0, 1.0);
}

/// Attachment load operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Preserve existing contents.
    Load,
    /// Clear to the supplied value.
    Clear(T),
    /// Existing contents are irrelevant.
    DontCare,
}

/// Attachment store operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Preserve rendered contents.
    #[default]
    Store,
    /// Rendered contents may be discarded.
    DontCare,
}

/// Surface operation result that still produced a usable frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SurfaceStatus {
    /// Surface remains optimal.
    #[default]
    Optimal,
    /// Frame is usable, but the surface should soon be recreated.
    Suboptimal,
}

/// Presentation mode requested when creating a swapchain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PresentMode {
    /// Wait for vertical blanking and queue frames in order.
    #[default]
    Fifo,
    /// Prefer low-latency queued presentation when supported.
    Mailbox,
    /// Present immediately without synchronizing to vertical blanking.
    Immediate,
}

/// Color space associated with presentation images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorSpace {
    /// Standard sRGB color space.
    #[default]
    Srgb,
    /// Display P3 color space.
    DisplayP3,
    /// HDR10 using the ST 2084 transfer function.
    Hdr10,
}

/// Texture format and color space selected for presentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceFormat {
    /// Storage format of presentation images.
    pub texture: TextureFormat,
    /// Color space used by the display surface.
    pub color_space: ColorSpace,
}
