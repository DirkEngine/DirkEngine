use ash::vk;

use crate::{
    physical_device,
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Vertex field offsets are far below u32::MAX."
    )]
    const ATTRIBUTES: &'static [vk::VertexInputAttributeDescription] = &[
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Self, position) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Self, normal) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Self, texcoord) as u32,
        },
    ];
}

pub fn make_version(version: dirk_utils::Version) -> u32 {
    vk::make_api_version(0, version.major(), version.minor(), version.patch())
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

impl Drop for Frame {
    fn drop(&mut self) {
        self.submitted_command_buffers.clear();
        self.command_pool.destroy();
    }
}

pub struct RendererProperties {
    pub msaa_samples: vk::SampleCountFlags,
    #[expect(
        dead_code,
        reason = "Not read until samplers enable anisotropy only where the device supports it."
    )]
    pub anisotropy: bool,
    pub surface_format: vk::SurfaceFormatKHR,
    pub queue_family_indices: physical_device::QueueFamilyIndices,
    pub depth_format: vk::Format,
    pub present_mode: vk::PresentModeKHR,
}
