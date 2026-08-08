use dirk_rhi::{
    AddressMode, CompareOp, CullMode, FilterMode, Format, FrontFace, PrimitiveTopology,
    SampleCount, StoreOp, VertexStepMode,
};
use metal::{
    MTLCompareFunction, MTLCullMode, MTLIndexType, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLStoreAction,
    MTLVertexFormat, MTLVertexStepFunction, MTLWinding,
};

pub(crate) fn format(value: Format) -> MTLPixelFormat {
    match value {
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::Rg32Float => MTLPixelFormat::RG32Float,
        Format::Rgb32Float => MTLPixelFormat::RGBA32Float,
        Format::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::Depth24UnormStencil8 => MTLPixelFormat::Depth24Unorm_Stencil8,
        Format::Depth32Float => MTLPixelFormat::Depth32Float,
        Format::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

pub(crate) const fn bytes_per_pixel(value: Format) -> u64 {
    match value {
        Format::Rgba8Unorm
        | Format::Rgba8Srgb
        | Format::Bgra8Unorm
        | Format::Bgra8Srgb
        | Format::Depth32Float
        | Format::Depth24UnormStencil8 => 4,
        Format::Rg32Float | Format::Depth32FloatStencil8 => 8,
        Format::Rgb32Float => 12,
        Format::Depth16Unorm => 2,
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

pub(crate) fn vertex_format(value: Format) -> Option<MTLVertexFormat> {
    match value {
        Format::Rg32Float => Some(MTLVertexFormat::Float2),
        Format::Rgb32Float => Some(MTLVertexFormat::Float3),
        _ => None,
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
