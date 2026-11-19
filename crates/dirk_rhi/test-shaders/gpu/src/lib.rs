//! Shaders for `dirk_rhi`'s native tests, compiled by rust-gpu. Push data
//! types are shared with the tests through `dirk_rhi_test_shaders`.

#![no_std]

use bytemuck::NoUninit;
use dirk_shaders::bindless::{BufferIndex, SamplerIndex, TextureIndex};
use spirv_std::{
    Image, RuntimeArray, Sampler, TypedBuffer,
    glam::{Vec2, Vec4, vec4},
    spirv,
};

/// Push data of the fill shaders: a solid color at a depth.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Fill {
    /// Linear RGBA color.
    pub color: [f32; 4],
    /// Depth of the triangle, in `0.0..=1.0`.
    pub depth: f32,
}

/// Covers the render area with one triangle at `fill.depth`; draw three
/// vertices.
#[spirv(vertex)]
pub fn fill_vs(
    #[spirv(vertex_index)] index: i32,
    #[spirv(push_constant)] fill: &Fill,
    #[spirv(position)] position: &mut Vec4,
) {
    // Vertices (-1, -1), (3, -1) and (-1, 3).
    let corner = |bit: i32| if index & bit == 0 { -1.0 } else { 3.0 };
    *position = vec4(corner(1), corner(2), fill.depth, 1.0);
}

/// Writes `fill.color`.
#[spirv(fragment)]
pub fn fill_fs(#[spirv(push_constant)] fill: &Fill, color: &mut Vec4) {
    *color = Vec4::from_array(fill.color);
}

/// A vertex of the textured shaders, pulled from a storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Vertex {
    /// Clip-space position.
    pub position: [f32; 2],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

/// Push data of the textured shaders: where their vertices, texture and
/// sampler are in the bindless heap.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Textured {
    /// The vertices, drawn in order.
    pub vertices: BufferIndex<Vertex>,
    /// A 2D texture.
    pub texture: TextureIndex,
    /// The sampler reading it.
    pub sampler: SamplerIndex,
}

/// Places the vertex `push.vertices[index]`.
#[spirv(vertex)]
pub fn textured_vs(
    #[spirv(vertex_index)] index: i32,
    #[spirv(push_constant)] push: &Textured,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] buffers: &RuntimeArray<
        TypedBuffer<[Vertex]>,
    >,
    #[spirv(position)] position: &mut Vec4,
    uv: &mut Vec2,
) {
    let vertex = push.vertices.get(buffers)[index as usize];
    *position = vec4(vertex.position[0], vertex.position[1], 0.0, 1.0);
    *uv = Vec2::from_array(vertex.uv);
}

/// Push data of the instanced shader: buffers of two element types.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Instanced {
    /// The vertices, drawn in order.
    pub vertices: BufferIndex<Vertex>,
    /// An RGBA8 color per instance, red in the lowest byte.
    pub colors: BufferIndex<u32>,
}

/// Places the vertex `push.vertices[index]` and passes on the color
/// `push.colors[instance]`. Reads one heap array as two types, like meshes
/// reading their vertices and per-object data.
#[spirv(vertex)]
pub fn instanced_vs(
    #[spirv(vertex_index)] index: i32,
    #[spirv(instance_index)] instance: i32,
    #[spirv(push_constant)] push: &Instanced,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] vertices: &RuntimeArray<
        TypedBuffer<[Vertex]>,
    >,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] colors: &RuntimeArray<
        TypedBuffer<[u32]>,
    >,
    #[spirv(position)] position: &mut Vec4,
    color: &mut Vec4,
) {
    let vertex = push.vertices.get(vertices)[index as usize];
    *position = vec4(vertex.position[0], vertex.position[1], 0.0, 1.0);
    let rgba = push.colors.get(colors)[instance as usize];
    let channel = |byte: u32| ((rgba >> (8 * byte)) & 0xff) as f32 / 255.0;
    *color = vec4(channel(0), channel(1), channel(2), channel(3));
}

/// Writes the color it is given.
#[spirv(fragment)]
pub fn color_fs(input: Vec4, color: &mut Vec4) {
    *color = input;
}

/// Samples `push.texture` with `push.sampler`.
#[spirv(fragment)]
pub fn textured_fs(
    #[spirv(push_constant)] push: &Textured,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<
        Image!(2D, type = f32, sampled),
    >,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    color: &mut Vec4,
) {
    *color = push
        .texture
        .get(textures)
        .sample(*push.sampler.get(samplers), uv);
}
