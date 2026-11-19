//! Shaders for `dirk_rhi`'s native tests, compiled by rust-gpu. Push data
//! types are shared with the tests through `dirk_rhi_test_shaders`.

#![no_std]

use bytemuck::NoUninit;
use spirv_std::{
    glam::{Vec4, vec4},
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
