use std::marker::PhantomData;

use bitflags::bitflags;

use crate::{
    Capabilities, Error, Features, Format, FormatFeatures, ImageUsage, PushData, Result, Rhi,
    SampleCount, ShaderModule, ShaderStage, native,
};

/// Most color targets the RHI supports in a pipeline or render pass. The
/// device may support fewer; see
/// [`Limits::max_color_targets`](crate::Limits::max_color_targets).
pub const MAX_COLOR_TARGETS: usize = 8;

/// How vertices are assembled into primitives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Topology {
    /// Every three vertices form a triangle.
    #[default]
    TriangleList,
    /// Every vertex forms a triangle with the two before it.
    TriangleStrip,
    /// Every two vertices form a line.
    LineList,
    /// Every vertex forms a line with the one before it.
    LineStrip,
    /// Every vertex is a point.
    PointList,
}

/// Which triangles are discarded by the way they face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CullMode {
    /// Keep every triangle.
    #[default]
    None,
    /// Discard front-facing triangles.
    Front,
    /// Discard back-facing triangles.
    Back,
}

/// The winding order of front-facing triangles, in framebuffer coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontFace {
    /// Counter-clockwise triangles face the front.
    #[default]
    CounterClockwise,
    /// Clockwise triangles face the front.
    Clockwise,
}

/// How a new value compares against the stored one to pass a test: a
/// fragment's depth against the depth attachment's, or a comparison
/// sampler's reference against a depth texel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Never passes.
    Never,
    /// Passes when the new value is less.
    #[default]
    Less,
    /// Passes when the values are equal.
    Equal,
    /// Passes when the new value is less or equal.
    LessOrEqual,
    /// Passes when the new value is greater.
    Greater,
    /// Passes when the values differ.
    NotEqual,
    /// Passes when the new value is greater or equal.
    GreaterOrEqual,
    /// Always passes.
    Always,
}

/// Depth added to fragments before the depth test, to avoid shadow acne.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthBias {
    /// Constant bias, in units of the depth format's precision.
    pub constant: f32,
    /// Bias scaled by the primitive's depth slope.
    pub slope: f32,
    /// Largest absolute bias; zero for no limit. Needs
    /// [`Features::DEPTH_BIAS_CLAMP`] when not zero.
    pub clamp: f32,
}

/// The depth target of a pipeline and how fragments are tested against it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthState {
    /// Format of the depth target.
    pub format: Format,
    /// Whether passing fragments write their depth.
    pub write: bool,
    /// The depth test.
    pub compare: CompareOp,
    /// Depth bias.
    pub bias: DepthBias,
}

/// Factors source and destination values are multiplied by when blending.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    /// 0.
    Zero,
    /// 1.
    One,
    /// The source value.
    Src,
    /// 1 minus the source value.
    OneMinusSrc,
    /// The source alpha.
    SrcAlpha,
    /// 1 minus the source alpha.
    OneMinusSrcAlpha,
    /// The destination value.
    Dst,
    /// 1 minus the destination value.
    OneMinusDst,
    /// The destination alpha.
    DstAlpha,
    /// 1 minus the destination alpha.
    OneMinusDstAlpha,
}

/// How weighted source and destination values combine when blending.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendOp {
    /// Source plus destination.
    #[default]
    Add,
    /// Source minus destination.
    Subtract,
    /// Destination minus source.
    ReverseSubtract,
    /// The smaller of source and destination, unweighted.
    Min,
    /// The larger of source and destination, unweighted.
    Max,
}

/// How one part of a color, its RGB or its alpha, is blended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// Factor of the new value.
    pub src: BlendFactor,
    /// Factor of the stored value.
    pub dst: BlendFactor,
    /// How the weighted values combine.
    pub op: BlendOp,
}

/// How new colors blend with stored ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// Blending of red, green and blue.
    pub color: BlendComponent,
    /// Blending of alpha.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Blends colors that are not premultiplied by their alpha.
    pub const ALPHA: Self = Self {
        color: BlendComponent {
            src: BlendFactor::SrcAlpha,
            dst: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
        alpha: Self::PREMULTIPLIED_ALPHA.alpha,
    };
    /// Blends colors premultiplied by their alpha.
    pub const PREMULTIPLIED_ALPHA: Self = Self {
        color: BlendComponent {
            src: BlendFactor::One,
            dst: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
        alpha: BlendComponent {
            src: BlendFactor::One,
            dst: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
    };
}

bitflags! {
    /// Color channels a pipeline writes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ColorWrites: u8 {
        /// Red.
        const RED = 1 << 0;
        /// Green.
        const GREEN = 1 << 1;
        /// Blue.
        const BLUE = 1 << 2;
        /// Alpha.
        const ALPHA = 1 << 3;
    }
}

impl Default for ColorWrites {
    /// Every channel.
    fn default() -> Self {
        Self::all()
    }
}

/// A color target of a pipeline and how fragments are written to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    /// Format of the target.
    pub format: Format,
    /// Blending with the stored color; `None` replaces it.
    pub blend: Option<BlendState>,
    /// Channels written.
    pub writes: ColorWrites,
}

impl ColorTarget {
    /// A target of `format` whose colors are replaced.
    #[must_use]
    pub fn new(format: Format) -> Self {
        Self {
            format,
            blend: None,
            writes: ColorWrites::all(),
        }
    }
}

/// Describes a graphics pipeline whose shaders read push data `P`.
///
/// Pipelines have no vertex input layouts: shaders read vertices from
/// storage buffers through the bindless heap. Viewport and scissor are set
/// per render pass.
#[derive(Clone, Copy)]
pub struct GraphicsPipelineDesc<'a, P: PushData> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// The vertex shader.
    pub vertex: &'a ShaderModule<P>,
    /// The fragment shader; `None` for depth-only pipelines.
    pub fragment: Option<&'a ShaderModule<P>>,
    /// How vertices form primitives.
    pub topology: Topology,
    /// Which triangles are discarded.
    pub cull_mode: CullMode,
    /// Which triangles face the front.
    pub front_face: FrontFace,
    /// Color targets, in attachment order.
    pub color_targets: &'a [ColorTarget],
    /// The depth target and test; `None` without one.
    pub depth: Option<DepthState>,
    /// Samples per pixel of every target.
    pub samples: SampleCount,
}

/// The formats and sample count of a pipeline's or render pass's targets.
/// Pipelines draw only in render passes with the same targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Targets {
    pub(crate) colors: [Option<Format>; MAX_COLOR_TARGETS],
    pub(crate) depth: Option<Format>,
    pub(crate) samples: SampleCount,
}

/// A graphics pipeline whose shaders read push data `P`. Destroyed when
/// dropped, once the GPU is done with it.
pub struct GraphicsPipeline<P: PushData> {
    pub(crate) raw: native::GraphicsPipeline,
    pub(crate) targets: Targets,
    label: Box<str>,
    push: PhantomData<fn(&P)>,
}

impl Rhi {
    /// Creates a graphics pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if a shader runs in the wrong stage or
    /// belongs to another device, or a target's format, blending or sample
    /// count is unsupported, exceeds a device limit or needs a missing
    /// feature, or a native error if the pipeline cannot be created.
    pub fn create_graphics_pipeline<P: PushData>(
        &self,
        desc: &GraphicsPipelineDesc<'_, P>,
    ) -> Result<GraphicsPipeline<P>> {
        let targets = desc.check(self.capabilities())?;
        for module in [Some(desc.vertex), desc.fragment].into_iter().flatten() {
            if !module.raw.is_from(&self.device) {
                return Err(desc.invalid("a shader belongs to another device"));
            }
        }
        Ok(GraphicsPipeline {
            raw: native::GraphicsPipeline::new(&self.device, desc)?,
            targets,
            label: desc.label.into(),
            push: PhantomData,
        })
    }
}

impl<P: PushData> GraphicsPipelineDesc<'_, P> {
    /// Validates the description against `capabilities` and returns its
    /// targets.
    fn check(&self, capabilities: &Capabilities) -> Result<Targets> {
        if self.vertex.stage() != ShaderStage::Vertex
            || self
                .fragment
                .is_some_and(|fragment| fragment.stage() != ShaderStage::Fragment)
        {
            return Err(self.invalid(format!(
                "`{}` and `{}` are not a vertex and a fragment shader",
                self.vertex.entry_point(),
                self.fragment.map_or("", |fragment| fragment.entry_point())
            )));
        }
        let limit = capabilities.limits.color_targets();
        if self.color_targets.len() > limit {
            return Err(self.invalid(format!(
                "{} color targets exceed the limit of {limit}",
                self.color_targets.len()
            )));
        }
        if self.color_targets.is_empty() && self.depth.is_none() {
            return Err(self.invalid("a pipeline needs a color or depth target"));
        }
        if !self.color_targets.is_empty() && self.fragment.is_none() {
            return Err(self.invalid("color targets need a fragment shader"));
        }
        let mut targets = Targets {
            samples: self.samples,
            ..Targets::default()
        };
        for (slot, target) in targets.colors.iter_mut().zip(self.color_targets) {
            self.check_format(capabilities, target.format, ImageUsage::COLOR_ATTACHMENT)?;
            let blendable = capabilities
                .format(target.format)
                .features
                .contains(FormatFeatures::BLEND);
            if target.blend.is_some() && !blendable {
                return Err(self.invalid(format!(
                    "{:?} cannot be blended on this device",
                    target.format
                )));
            }
            *slot = Some(target.format);
        }
        let independent = capabilities.features.contains(Features::INDEPENDENT_BLEND);
        if !independent
            && self.color_targets.iter().any(|target| {
                (target.blend, target.writes)
                    != (self.color_targets[0].blend, self.color_targets[0].writes)
            })
        {
            return Err(self.invalid(
                "color targets blend or write differently, which this device does not support",
            ));
        }
        if let Some(depth) = self.depth {
            self.check_format(
                capabilities,
                depth.format,
                ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            )?;
            let DepthBias {
                constant,
                slope,
                clamp,
            } = depth.bias;
            if !(constant.is_finite() && slope.is_finite() && clamp.is_finite()) {
                return Err(self.invalid(format!("depth bias {:?} is not finite", depth.bias)));
            }
            if clamp != 0.0 && !capabilities.features.contains(Features::DEPTH_BIAS_CLAMP) {
                return Err(self.invalid("clamped depth bias is not supported on this device"));
            }
            targets.depth = Some(depth.format);
        }
        Ok(targets)
    }

    /// Checks that targets of `format` support `usage` and the sample count.
    fn check_format(
        &self,
        capabilities: &Capabilities,
        format: Format,
        usage: ImageUsage,
    ) -> Result<()> {
        let supported = capabilities.format(format);
        let depth = usage == ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        if format.is_depth_stencil() != depth || !supported.usage.contains(usage) {
            return Err(self.invalid(format!(
                "{format:?} cannot be a {} target on this device",
                if depth { "depth" } else { "color" }
            )));
        }
        if !supported.sample_counts.contains(self.samples.flag()) {
            return Err(self.invalid(format!(
                "{format:?} does not support {:?} samples on this device",
                self.samples
            )));
        }
        Ok(())
    }

    /// Rejects this description.
    fn invalid(&self, reason: impl Into<String>) -> Error {
        Error::invalid("graphics pipeline", self.label, reason)
    }
}

impl<P: PushData> GraphicsPipeline<P> {
    /// The pipeline's debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}
