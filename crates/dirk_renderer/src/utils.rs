use dirk_rhi::{SampleCount, TextureFormat, VertexAttribute, VertexFormat};

use crate::{
    resources::{
        command_pool::{CommandBuffer, CommandPool, Graphics},
        sync::Fence,
    },
    shaders::metadata::VertexInput,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub texcoord: [f32; 2],
}

impl VertexInput for Vertex {
    // the offset is far from u32::MAX
    #[allow(clippy::cast_possible_truncation)]
    const ATTRIBUTES: &'static [VertexAttribute] = &[
        VertexAttribute {
            location: 0,
            format: VertexFormat::Float32x3,
            offset: std::mem::offset_of!(Self, position) as u32,
        },
        VertexAttribute {
            location: 1,
            format: VertexFormat::Float32x3,
            offset: std::mem::offset_of!(Self, normal) as u32,
        },
        VertexAttribute {
            location: 2,
            format: VertexFormat::Float32x2,
            offset: std::mem::offset_of!(Self, texcoord) as u32,
        },
    ];
}

pub struct Frame {
    /// Command pool to allocate command buffers on every frame
    pub command_pool: CommandPool<Graphics>,
    /// Submitted command buffers kept alive until this frame's fence completes.
    pub submitted_command_buffers: Vec<CommandBuffer>,
    /// Main synchronization fence
    pub fence: Fence,
    // TODO: have one primary command buffer that is allocated once and
    // secondary command for each scene. Should be allocated every time
    // there is a change in scene count. If not reallocated, reset.
}

pub struct RendererProperties {
    pub msaa_samples: SampleCount,
    #[allow(unused)]
    pub anisotropy: bool,
    pub surface_format: TextureFormat,
    pub depth_format: TextureFormat,
}
