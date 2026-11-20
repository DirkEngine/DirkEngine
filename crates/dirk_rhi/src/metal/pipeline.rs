use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction, MTLCompileOptions,
    MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLFunction,
    MTLLanguageVersion, MTLLibrary, MTLPrimitiveType, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLWinding,
};

use super::{Check, Device, Object, device::Garbage, heap::Usage};
use crate::{
    BlendFactor, BlendOp, ColorWrites, CompareOp, CullMode, DepthBias, Error, FrontFace,
    GraphicsPipelineDesc, PushData, Result, ShaderModule as Module, Topology,
};

/// Identifies a pipeline while it is alive, to skip rebinding it.
pub(crate) type PipelineId = *const ProtocolObject<dyn MTLRenderPipelineState>;

/// The entry point of a Metal library compiled from the shader build's MSL.
pub(crate) struct ShaderModule {
    pub(super) device: Arc<Device>,
    pub(super) function: Object<dyn MTLFunction>,
}

impl ShaderModule {
    /// Compiles the MSL source `code`, which holds only `entry_point`. The
    /// shader build may have renamed it to be a valid MSL name, so the
    /// library's only function is used.
    pub(crate) fn new(device: &Arc<Device>, entry_point: &str, code: &[u8]) -> Result<Self> {
        let invalid = |reason: &str| Error::invalid("shader", entry_point, reason);
        let source = str::from_utf8(code).map_err(|_| invalid("is not UTF-8 MSL source"))?;
        let library = device.library(source)?;
        let names = library.functionNames();
        let [ref name] = names.to_vec()[..] else {
            return Err(invalid("does not hold exactly one entry point"));
        };
        let function: Object<_> = library
            .newFunctionWithName(name)
            .check("newFunctionWithName")?
            .into();
        function.setLabel(Some(&NSString::from_str(entry_point)));
        Ok(Self {
            device: device.clone(),
            function,
        })
    }
}

impl Device {
    /// Compiles the MSL `source`, of Metal 3.
    pub(super) fn library(&self, source: &str) -> Result<Object<dyn MTLLibrary>> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version3_0);
        Ok(self
            .raw
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .check("compiling MSL")?
            .into())
    }
}

/// A Metal render pipeline and the state Metal sets on the encoder instead.
pub(crate) struct GraphicsPipeline {
    pub(super) device: Arc<Device>,
    pub(super) raw: Object<dyn MTLRenderPipelineState>,
    /// The depth test, with a depth target.
    pub(super) depth: Option<Object<dyn MTLDepthStencilState>>,
    pub(super) primitive: MTLPrimitiveType,
    pub(super) cull_mode: MTLCullMode,
    pub(super) winding: MTLWinding,
    pub(super) bias: DepthBias,
    /// The usage its draws need resources declared up to.
    pub(super) usage: Usage,
}

impl GraphicsPipeline {
    /// Creates a pipeline for a validated description.
    pub(crate) fn new<P: PushData>(
        device: &Arc<Device>,
        desc: &GraphicsPipelineDesc<'_, P>,
    ) -> Result<Self> {
        let pipeline = MTLRenderPipelineDescriptor::new();
        pipeline.setLabel(Some(&NSString::from_str(desc.label)));
        pipeline.setVertexFunction(Some(&desc.vertex.raw.function));
        pipeline.setFragmentFunction(desc.fragment.map(|fragment| &*fragment.raw.function));
        pipeline.setRasterSampleCount(desc.samples.count());
        for (index, target) in desc.color_targets.iter().enumerate() {
            // SAFETY: the index is below the color target limit.
            let attachment = unsafe { pipeline.colorAttachments().objectAtIndexedSubscript(index) };
            attachment.setPixelFormat(target.format.mtl());
            attachment.setWriteMask(target.writes.mtl());
            if let Some(blend) = target.blend {
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(blend.color.src.mtl());
                attachment.setDestinationRGBBlendFactor(blend.color.dst.mtl());
                attachment.setRgbBlendOperation(blend.color.op.mtl());
                attachment.setSourceAlphaBlendFactor(blend.alpha.src.mtl());
                attachment.setDestinationAlphaBlendFactor(blend.alpha.dst.mtl());
                attachment.setAlphaBlendOperation(blend.alpha.op.mtl());
            }
        }
        let depth = desc
            .depth
            .map(|depth| {
                let format = depth.format.mtl();
                pipeline.setDepthAttachmentPixelFormat(format);
                if depth.format.has_stencil() {
                    pipeline.setStencilAttachmentPixelFormat(format);
                }
                let state = MTLDepthStencilDescriptor::new();
                state.setDepthCompareFunction(depth.compare.mtl());
                state.setDepthWriteEnabled(depth.write);
                device
                    .raw
                    .newDepthStencilStateWithDescriptor(&state)
                    .check("newDepthStencilStateWithDescriptor")
                    .map(Object::from)
            })
            .transpose()?;
        let raw = device
            .raw
            .newRenderPipelineStateWithDescriptor_error(&pipeline)
            .check("newRenderPipelineStateWithDescriptor")?
            .into();
        Ok(Self {
            device: device.clone(),
            raw,
            depth,
            primitive: desc.topology.mtl(),
            cull_mode: desc.cull_mode.mtl(),
            winding: desc.front_face.mtl(),
            bias: desc.depth.map_or_default(|depth| depth.bias),
            usage: Usage::of_pipeline(
                desc.vertex.writes_storage() || desc.fragment.is_some_and(Module::writes_storage),
            ),
        })
    }

    /// Identifies the pipeline while it is alive.
    pub(crate) fn id(&self) -> PipelineId {
        &raw const *self.raw
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        self.device
            .bury(Garbage::object((self.raw.clone(), self.depth.take())));
    }
}

impl Topology {
    /// The equivalent Metal primitive type.
    const fn mtl(self) -> MTLPrimitiveType {
        match self {
            Self::TriangleList => MTLPrimitiveType::Triangle,
            Self::TriangleStrip => MTLPrimitiveType::TriangleStrip,
            Self::LineList => MTLPrimitiveType::Line,
            Self::LineStrip => MTLPrimitiveType::LineStrip,
            Self::PointList => MTLPrimitiveType::Point,
        }
    }
}

impl CullMode {
    /// The equivalent Metal cull mode.
    const fn mtl(self) -> MTLCullMode {
        match self {
            Self::None => MTLCullMode::None,
            Self::Front => MTLCullMode::Front,
            Self::Back => MTLCullMode::Back,
        }
    }
}

impl FrontFace {
    /// The equivalent Metal winding. Both are in framebuffer coordinates,
    /// which flipping y in vertex shaders leaves as Vulkan's.
    const fn mtl(self) -> MTLWinding {
        match self {
            Self::CounterClockwise => MTLWinding::CounterClockwise,
            Self::Clockwise => MTLWinding::Clockwise,
        }
    }
}

impl CompareOp {
    /// The equivalent Metal compare function.
    pub(super) const fn mtl(self) -> MTLCompareFunction {
        match self {
            Self::Never => MTLCompareFunction::Never,
            Self::Less => MTLCompareFunction::Less,
            Self::Equal => MTLCompareFunction::Equal,
            Self::LessOrEqual => MTLCompareFunction::LessEqual,
            Self::Greater => MTLCompareFunction::Greater,
            Self::NotEqual => MTLCompareFunction::NotEqual,
            Self::GreaterOrEqual => MTLCompareFunction::GreaterEqual,
            Self::Always => MTLCompareFunction::Always,
        }
    }
}

impl BlendFactor {
    /// The equivalent Metal blend factor.
    const fn mtl(self) -> MTLBlendFactor {
        match self {
            Self::Zero => MTLBlendFactor::Zero,
            Self::One => MTLBlendFactor::One,
            Self::Src => MTLBlendFactor::SourceColor,
            Self::OneMinusSrc => MTLBlendFactor::OneMinusSourceColor,
            Self::SrcAlpha => MTLBlendFactor::SourceAlpha,
            Self::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
            Self::Dst => MTLBlendFactor::DestinationColor,
            Self::OneMinusDst => MTLBlendFactor::OneMinusDestinationColor,
            Self::DstAlpha => MTLBlendFactor::DestinationAlpha,
            Self::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        }
    }
}

impl BlendOp {
    /// The equivalent Metal blend operation.
    const fn mtl(self) -> MTLBlendOperation {
        match self {
            Self::Add => MTLBlendOperation::Add,
            Self::Subtract => MTLBlendOperation::Subtract,
            Self::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
            Self::Min => MTLBlendOperation::Min,
            Self::Max => MTLBlendOperation::Max,
        }
    }
}

impl ColorWrites {
    /// The equivalent Metal write mask, whose bits run the other way.
    fn mtl(self) -> MTLColorWriteMask {
        [
            (Self::RED, MTLColorWriteMask::Red),
            (Self::GREEN, MTLColorWriteMask::Green),
            (Self::BLUE, MTLColorWriteMask::Blue),
            (Self::ALPHA, MTLColorWriteMask::Alpha),
        ]
        .into_iter()
        .filter(|(write, _)| self.contains(*write))
        .fold(MTLColorWriteMask::None, |all, (_, mask)| all | mask)
    }
}
