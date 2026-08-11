use dirk_rhi::{
    BindGroupLayoutEntry, ShaderStage, VertexAttribute, VertexBufferLayout, VertexStepMode,
};

use crate::{
    Result,
    resources::{ActiveShader, device::RenderDevice},
    shaders::ShaderCode,
};

/// Host vertex type and its backend-neutral input layout.
pub trait VertexInput: Sized + Copy {
    const ATTRIBUTES: &'static [VertexAttribute];

    fn layout() -> VertexBufferLayout<'static> {
        #[allow(clippy::cast_possible_truncation)]
        let stride = size_of::<Self>() as u32;
        VertexBufferLayout {
            stride,
            step_mode: VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}

/// Reflected metadata shared by all shader stages.
pub trait Shader {
    const CODE: ShaderCode;
    const ENTRYPOINT: &'static str;
    const STAGE: ShaderStage;
    const SET_LAYOUTS: &'static [&'static [BindGroupLayoutEntry]];

    fn create(device: &RenderDevice) -> Result<ActiveShader> {
        Self::CODE.create(device, Self::STAGE, Self::ENTRYPOINT)
    }
}

pub trait VertexShader: Shader {
    const INPUT_LAYOUTS: &'static [VertexBufferLayout<'static>];
}

pub trait FragmentShader: Shader {}
