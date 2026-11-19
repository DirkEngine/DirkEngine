//! Types shared by the renderer and its shaders: vertices and per-object
//! records the shaders pull from storage buffers, and each pipeline's push
//! data.
//!
//! Everything is laid out with 4-byte scalars only (arrays rather than
//! vectors), so the host's `#[repr(C)]` layout is exactly the layout shaders
//! read from storage buffers and push constants.

use bytemuck::{NoUninit, Pod, Zeroable};

use crate::bindless::{BufferIndex, BufferSlice, SamplerIndex, TextureIndex};

/// A column-major 4x4 matrix.
pub type Matrix = [[f32; 4]; 4];

/// A vertex of a mesh, pulled by the mesh shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, NoUninit)]
pub struct MeshVertex {
    /// Position in model space.
    pub position: [f32; 3],
    /// Normal in model space.
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

/// Data of one drawn object, in frame memory the mesh shaders index with
/// the instance index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, NoUninit)]
pub struct ObjectData {
    /// Transforms model space to world space.
    pub model: Matrix,
}

/// Push data of the mesh shaders: one draw of a mesh primitive. The drawn
/// instance's index selects its [`ObjectData`].
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct MeshPush {
    /// Transforms world space to clip space.
    pub view_projection: Matrix,
    /// This frame's objects, in frame memory.
    pub objects: BufferSlice<ObjectData>,
    /// The primitive's vertices.
    pub vertices: BufferIndex<MeshVertex>,
    /// The base color texture, holding sRGB colors sampled as linear ones.
    pub texture: TextureIndex,
    /// The sampler reading it.
    pub sampler: SamplerIndex,
}

/// A vertex of egui's meshes, laid out like `epaint::Vertex`, so egui's
/// vertices are uploaded as they are.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EguiVertex {
    /// Position in points from the top left corner.
    pub position: [f32; 2],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// sRGB-encoded color with premultiplied alpha, red in the lowest byte.
    pub color: u32,
}

/// Push data of the egui shaders: one egui mesh.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct EguiPush {
    /// Size of the screen in points.
    pub screen_size: [f32; 2],
    /// The mesh's vertices, in frame memory.
    pub vertices: BufferSlice<EguiVertex>,
    /// The mesh's texture, holding linear colors with premultiplied alpha.
    pub texture: TextureIndex,
    /// The sampler reading it.
    pub sampler: SamplerIndex,
}

/// Push data of the presentation shaders, which cover a window with an
/// image of linear colors.
#[repr(C)]
#[derive(Clone, Copy, Debug, NoUninit)]
pub struct PresentPush {
    /// The image shown.
    pub source: TextureIndex,
    /// The sampler reading it.
    pub sampler: SamplerIndex,
}
