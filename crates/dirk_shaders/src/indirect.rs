//! Arguments of indirect draws, read by the GPU from a buffer that the host
//! or shaders write. Laid out as both Vulkan and Metal read them, so shaders
//! writing draws (such as GPU culling) and the host share these types.

use bytemuck::NoUninit;

/// Arguments of one indirect draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, NoUninit)]
pub struct DrawArguments {
    /// Vertices drawn.
    pub vertex_count: u32,
    /// Instances drawn.
    pub instance_count: u32,
    /// Index of the first vertex.
    pub first_vertex: u32,
    /// Index of the first instance.
    pub first_instance: u32,
}

/// Arguments of one indirect indexed draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, NoUninit)]
pub struct DrawIndexedArguments {
    /// Indices drawn.
    pub index_count: u32,
    /// Instances drawn.
    pub instance_count: u32,
    /// Position of the first index in the index buffer.
    pub first_index: u32,
    /// Added to each index before reading the vertex.
    pub base_vertex: i32,
    /// Index of the first instance.
    pub first_instance: u32,
}
