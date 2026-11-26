use std::{ffi::CString, io::Cursor, sync::Arc};

use ash::vk;

use super::{Check, Device, device::Garbage};
use crate::{
    BlendFactor, BlendOp, BlendState, ColorWrites, CompareOp, CullMode, Error, Format, FrontFace,
    GraphicsPipelineDesc, PushData, Result, Topology,
};

/// Shader stages that read push data. Pushes name all of them, as the global
/// pipeline layout's push range does.
pub(super) const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw()
        | vk::ShaderStageFlags::FRAGMENT.as_raw()
        | vk::ShaderStageFlags::COMPUTE.as_raw(),
);

/// Identifies a pipeline while it is alive, to skip rebinding it.
pub(crate) type PipelineId = vk::Pipeline;

/// A Vulkan shader module and the name of its entry point.
pub(crate) struct ShaderModule {
    pub(super) device: Arc<Device>,
    raw: vk::ShaderModule,
    entry_point: CString,
}

impl ShaderModule {
    /// Creates a module from SPIR-V the shader build produced.
    pub(crate) fn new(device: &Arc<Device>, entry_point: &str, spirv: &[u8]) -> Result<Self> {
        let words = ash::util::read_spv(&mut Cursor::new(spirv))
            .map_err(|error| Error::native("reading SPIR-V", error))?;
        let entry_point = CString::new(entry_point)
            .map_err(|_| Error::invalid("shader", entry_point, "contains a NUL byte"))?;
        let info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: the `Shader` implementation vouches for the SPIR-V.
        let raw = unsafe { device.raw.create_shader_module(&info, None) }
            .check("vkCreateShaderModule")?;
        device.name(raw, &entry_point.to_string_lossy());
        Ok(Self {
            device: device.clone(),
            raw,
            entry_point,
        })
    }

    /// The stage creation info of this module's entry point in `stage`.
    fn stage(&self, stage: vk::ShaderStageFlags) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(self.raw)
            .name(&self.entry_point)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        // SAFETY: pipelines do not reference the modules they were created
        // from, and no pipeline creation is using this one.
        unsafe { self.device.raw.destroy_shader_module(self.raw, None) };
    }
}

/// A Vulkan graphics pipeline.
pub(crate) struct GraphicsPipeline {
    pub(super) device: Arc<Device>,
    pub(super) raw: vk::Pipeline,
}

impl GraphicsPipeline {
    /// Creates a pipeline for a validated description, with the global
    /// pipeline layout and dynamic viewport and scissor.
    pub(crate) fn new<P: PushData>(
        device: &Arc<Device>,
        desc: &GraphicsPipelineDesc<'_, P>,
    ) -> Result<Self> {
        let mut stages = vec![desc.vertex.raw.stage(vk::ShaderStageFlags::VERTEX)];
        stages.extend(
            desc.fragment
                .map(|fragment| fragment.raw.stage(vk::ShaderStageFlags::FRAGMENT)),
        );
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology.vk());
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let bias = desc.depth.map_or_default(|depth| depth.bias);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(desc.cull_mode.vk())
            .front_face(desc.front_face.vk())
            .depth_bias_enable(bias.constant != 0.0 || bias.slope != 0.0)
            .depth_bias_constant_factor(bias.constant)
            .depth_bias_slope_factor(bias.slope)
            .depth_bias_clamp(bias.clamp)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(desc.samples.vk());
        let depth_stencil =
            desc.depth
                .map_or_else(vk::PipelineDepthStencilStateCreateInfo::default, |depth| {
                    vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(true)
                        .depth_write_enable(depth.write)
                        .depth_compare_op(depth.compare.vk())
                });
        let attachments: Vec<_> = desc
            .color_targets
            .iter()
            .map(|target| {
                let blend = target.blend.map_or_else(
                    vk::PipelineColorBlendAttachmentState::default,
                    BlendState::vk,
                );
                blend.color_write_mask(target.writes.vk())
            })
            .collect();
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let formats: Vec<_> = desc
            .color_targets
            .iter()
            .map(|target| target.format.vk())
            .collect();
        let depth_format = desc.depth.map(|depth| depth.format);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(depth_format.map_or(vk::Format::UNDEFINED, Format::vk))
            .stencil_attachment_format(
                depth_format
                    .filter(|format| format.has_stencil())
                    .map_or(vk::Format::UNDEFINED, Format::vk),
            );
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(device.heap.pipeline_layout)
            .push_next(&mut rendering);
        // SAFETY: the description was validated against the device, and
        // every pointer in `info` outlives the call.
        let raw = unsafe {
            device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, result)| result)
        .check("vkCreateGraphicsPipelines")?[0];
        device.name(raw, desc.label);
        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    /// Identifies the pipeline while it is alive.
    pub(crate) fn id(&self) -> PipelineId {
        self.raw
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        self.device.bury(Garbage::Pipeline(self.raw));
    }
}

impl Topology {
    /// The equivalent Vulkan topology.
    const fn vk(self) -> vk::PrimitiveTopology {
        match self {
            Self::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::LineList => vk::PrimitiveTopology::LINE_LIST,
            Self::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Self::PointList => vk::PrimitiveTopology::POINT_LIST,
        }
    }
}

impl CullMode {
    /// The equivalent Vulkan cull mode.
    const fn vk(self) -> vk::CullModeFlags {
        match self {
            Self::None => vk::CullModeFlags::NONE,
            Self::Front => vk::CullModeFlags::FRONT,
            Self::Back => vk::CullModeFlags::BACK,
        }
    }
}

impl FrontFace {
    /// The equivalent Vulkan front face.
    const fn vk(self) -> vk::FrontFace {
        match self {
            Self::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            Self::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

impl CompareOp {
    /// The equivalent Vulkan compare op.
    pub(super) const fn vk(self) -> vk::CompareOp {
        match self {
            Self::Never => vk::CompareOp::NEVER,
            Self::Less => vk::CompareOp::LESS,
            Self::Equal => vk::CompareOp::EQUAL,
            Self::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            Self::Greater => vk::CompareOp::GREATER,
            Self::NotEqual => vk::CompareOp::NOT_EQUAL,
            Self::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            Self::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl BlendState {
    /// The equivalent Vulkan blend state, writing every channel.
    fn vk(self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(self.color.src.vk())
            .dst_color_blend_factor(self.color.dst.vk())
            .color_blend_op(self.color.op.vk())
            .src_alpha_blend_factor(self.alpha.src.vk())
            .dst_alpha_blend_factor(self.alpha.dst.vk())
            .alpha_blend_op(self.alpha.op.vk())
    }
}

impl BlendFactor {
    /// The equivalent Vulkan blend factor.
    const fn vk(self) -> vk::BlendFactor {
        match self {
            Self::Zero => vk::BlendFactor::ZERO,
            Self::One => vk::BlendFactor::ONE,
            Self::Src => vk::BlendFactor::SRC_COLOR,
            Self::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            Self::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            Self::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            Self::Dst => vk::BlendFactor::DST_COLOR,
            Self::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            Self::DstAlpha => vk::BlendFactor::DST_ALPHA,
            Self::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }
}

impl BlendOp {
    /// The equivalent Vulkan blend op.
    const fn vk(self) -> vk::BlendOp {
        match self {
            Self::Add => vk::BlendOp::ADD,
            Self::Subtract => vk::BlendOp::SUBTRACT,
            Self::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            Self::Min => vk::BlendOp::MIN,
            Self::Max => vk::BlendOp::MAX,
        }
    }
}

impl ColorWrites {
    /// The equivalent Vulkan color mask; the bits match.
    fn vk(self) -> vk::ColorComponentFlags {
        vk::ColorComponentFlags::from_raw(u32::from(self.bits()))
    }
}
