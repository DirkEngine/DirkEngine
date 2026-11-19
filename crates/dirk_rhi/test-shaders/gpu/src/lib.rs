//! Shaders for `dirk_rhi`'s native tests, compiled by rust-gpu. Push data
//! types are shared with the tests through `dirk_rhi_test_shaders`.

#![no_std]

use bytemuck::NoUninit;
use dirk_shaders::bindless::{
    BufferIndex, DepthTextureIndex, SamplerIndex, StorageImageIndex, TextureIndex,
};
use spirv_std::{
    Image, RuntimeArray, Sampler, TypedBuffer,
    glam::{UVec3, Vec2, Vec4, vec4},
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

/// Covers the render area with one triangle whose texture coordinates run
/// from zero to one across it; draw three vertices. Reads no push data, so
/// it pairs with fragment shaders of any.
#[spirv(vertex)]
pub fn covering_vs(
    #[spirv(vertex_index)] index: i32,
    #[spirv(position)] position: &mut Vec4,
    uv: &mut Vec2,
) {
    // Texture coordinates (0, 0), (2, 0) and (0, 2).
    let corner = |bit: i32| if index & bit == 0 { 0.0 } else { 2.0 };
    *uv = Vec2::new(corner(1), corner(2));
    *position = vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
}

/// Push data of the depth comparison shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct DepthCompare {
    /// A 2D depth texture.
    pub texture: DepthTextureIndex,
    /// The comparison sampler reading it.
    pub sampler: SamplerIndex,
    /// The depth compared with the texture's.
    pub reference: f32,
}

/// Writes the share of `push.texture`'s filtered texels passing the
/// comparison of `push.reference` with them to every channel.
#[spirv(fragment)]
pub fn depth_compare_fs(
    #[spirv(push_constant)] push: &DepthCompare,
    #[spirv(descriptor_set = 0, binding = 10)] textures: &RuntimeArray<
        Image!(2D, type = f32, sampled, depth),
    >,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    color: &mut Vec4,
) {
    let passed = push.texture.get(textures).sample_depth_reference(
        *push.sampler.get(samplers),
        uv,
        push.reference,
    );
    *color = Vec4::splat(passed);
}

/// Push data of the counting shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Count {
    /// The buffer written.
    pub output: BufferIndex<u32>,
    /// Added to each element's index.
    pub base: u32,
}

/// Writes `push.base + i` to element `i` of `push.output`, in workgroups of
/// 64 elements.
#[spirv(compute(threads(64)))]
pub fn count_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &Count,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] buffers: &mut RuntimeArray<
        TypedBuffer<[u32]>,
    >,
) {
    let output = push.output.get_mut(buffers);
    let index = id.x as usize;
    if index < output.len() {
        output[index] = push.base + id.x;
    }
}

/// Push data of the painting shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct Paint {
    /// A 2D storage image.
    pub image: StorageImageIndex,
    /// Linear RGBA color written.
    pub color: [f32; 4],
}

/// Writes `push.color` to the texels of `push.image`, in 8x8 workgroups;
/// dispatches cover exactly the image.
#[spirv(compute(threads(8, 8)))]
pub fn paint_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &Paint,
    #[spirv(descriptor_set = 0, binding = 5)] images: &RuntimeArray<
        Image!(2D, type = f32, sampled = false),
    >,
) {
    // SAFETY: dispatches cover only the image's texels.
    unsafe {
        push.image
            .get(images)
            .write(id.truncate(), Vec4::from_array(push.color));
    }
}
