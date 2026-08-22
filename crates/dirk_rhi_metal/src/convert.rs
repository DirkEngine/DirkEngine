use dirk_rhi::{
    AddressMode, BlendFactor, BlendOp, CompareOp, CullMode, FilterMode, FrontFace,
    PrimitiveTopology, SampleCount, StencilOp as RhiStencilOp, StoreOp, TextureFormat,
    VertexFormat, VertexStepMode,
};
use metal::{
    MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode, MTLIndexType,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLSamplerAddressMode, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLStencilOperation, MTLStoreAction, MTLVertexFormat,
    MTLVertexStepFunction, MTLWinding,
};

pub(crate) fn format(value: TextureFormat) -> MTLPixelFormat {
    match value {
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        TextureFormat::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        TextureFormat::R16Float => MTLPixelFormat::R16Float,
        TextureFormat::Rg16Float => MTLPixelFormat::RG16Float,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::R32Float => MTLPixelFormat::R32Float,
        TextureFormat::Rg32Float => MTLPixelFormat::RG32Float,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        TextureFormat::R11G11B10Float => MTLPixelFormat::RG11B10Float,
        TextureFormat::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
        TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        // Apple GPUs have no packed 24-bit depth; it maps to float32+stencil.
        TextureFormat::Depth24UnormStencil8 | TextureFormat::Depth32FloatStencil8 => {
            MTLPixelFormat::Depth32Float_Stencil8
        }
    }
}

pub(crate) fn vertex_format(value: VertexFormat) -> MTLVertexFormat {
    match value {
        VertexFormat::Float32 => MTLVertexFormat::Float,
        VertexFormat::Float32x2 => MTLVertexFormat::Float2,
        VertexFormat::Float32x3 => MTLVertexFormat::Float3,
        VertexFormat::Float32x4 => MTLVertexFormat::Float4,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
        VertexFormat::Uint16x4 => MTLVertexFormat::UShort4,
    }
}

pub(crate) fn stencil_op(value: RhiStencilOp) -> MTLStencilOperation {
    match value {
        RhiStencilOp::Keep => MTLStencilOperation::Keep,
        RhiStencilOp::Zero => MTLStencilOperation::Zero,
        RhiStencilOp::Replace => MTLStencilOperation::Replace,
        RhiStencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        RhiStencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        RhiStencilOp::Invert => MTLStencilOperation::Invert,
    }
}

pub(crate) fn blend_factor(value: BlendFactor) -> MTLBlendFactor {
    match value {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SourceAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DestinationAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDestinationAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
    }
}

pub(crate) fn blend_op(value: BlendOp) -> MTLBlendOperation {
    match value {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

pub(crate) const fn bytes_per_pixel(value: TextureFormat) -> u64 {
    match value {
        TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8Srgb
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8Srgb
        | TextureFormat::Rg16Float
        | TextureFormat::R32Float
        | TextureFormat::R11G11B10Float
        | TextureFormat::Depth32Float
        | TextureFormat::Depth24UnormStencil8 => 4,
        TextureFormat::R16Float | TextureFormat::Depth16Unorm => 2,
        // Packed float depth plus an 8-bit stencil face.
        TextureFormat::Rgba16Float
        | TextureFormat::Rg32Float
        | TextureFormat::Depth32FloatStencil8 => 8,
        TextureFormat::Rgba32Float => 16,
    }
}

pub(crate) const fn samples(value: SampleCount) -> u64 {
    value as u8 as u64
}

pub(crate) fn min_mag_filter(value: FilterMode) -> MTLSamplerMinMagFilter {
    match value {
        FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

pub(crate) fn mip_filter(value: FilterMode) -> MTLSamplerMipFilter {
    match value {
        FilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

pub(crate) fn address_mode(value: AddressMode) -> MTLSamplerAddressMode {
    match value {
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
    }
}

pub(crate) fn compare(value: CompareOp) -> MTLCompareFunction {
    match value {
        CompareOp::Never => MTLCompareFunction::Never,
        CompareOp::Less => MTLCompareFunction::Less,
        CompareOp::LessEqual => MTLCompareFunction::LessEqual,
        CompareOp::Equal => MTLCompareFunction::Equal,
        CompareOp::Greater => MTLCompareFunction::Greater,
        CompareOp::Always => MTLCompareFunction::Always,
    }
}

pub(crate) fn topology(value: PrimitiveTopology) -> MTLPrimitiveType {
    match value {
        PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
        PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
        PrimitiveTopology::LineList => MTLPrimitiveType::Line,
    }
}

pub(crate) fn winding(value: FrontFace) -> MTLWinding {
    match value {
        FrontFace::CounterClockwise => MTLWinding::CounterClockwise,
        FrontFace::Clockwise => MTLWinding::Clockwise,
    }
}

pub(crate) fn cull(value: CullMode) -> MTLCullMode {
    match value {
        CullMode::None => MTLCullMode::None,
        CullMode::Front => MTLCullMode::Front,
        CullMode::Back => MTLCullMode::Back,
    }
}

pub(crate) fn vertex_step(value: VertexStepMode) -> MTLVertexStepFunction {
    match value {
        VertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
        VertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
    }
}

pub(crate) fn load<T>(value: &dirk_rhi::LoadOp<T>) -> MTLLoadAction {
    match value {
        dirk_rhi::LoadOp::Load => MTLLoadAction::Load,
        dirk_rhi::LoadOp::Clear(_) => MTLLoadAction::Clear,
        dirk_rhi::LoadOp::DontCare => MTLLoadAction::DontCare,
    }
}

pub(crate) fn store(value: StoreOp, resolve: bool) -> MTLStoreAction {
    match (value, resolve) {
        (StoreOp::Store, true) => MTLStoreAction::StoreAndMultisampleResolve,
        (StoreOp::DontCare, true) => MTLStoreAction::MultisampleResolve,
        (StoreOp::Store, false) => MTLStoreAction::Store,
        (StoreOp::DontCare, false) => MTLStoreAction::DontCare,
    }
}

pub(crate) fn index(value: dirk_rhi::IndexFormat) -> MTLIndexType {
    match value {
        dirk_rhi::IndexFormat::Uint16 => MTLIndexType::UInt16,
        dirk_rhi::IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}
