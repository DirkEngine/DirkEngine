//! The engine's shaders, compiled to SPIR-V by `dirk_renderer`'s build script
//! with rust-gpu. Each entry point becomes its own module, which the build
//! script reflects into a type with its stage, descriptor set layouts and
//! vertex inputs.

#![no_std]
#![expect(
    clippy::too_many_arguments,
    reason = "Shader entry points take one argument per input, output and binding."
)]

use spirv_std::{
    glam::{Vec2, Vec3, Vec4},
    image::{Image2d, SampledImage},
    spirv,
};

use dirk_shaders::types::{ProxyUbo, SceneUbo};

/// Transforms a model's vertices into clip space, passing the texture
/// coordinate and normal on to [`main_fs`].
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(uniform, descriptor_set = 0, binding = 0)] scene: &SceneUbo,
    #[spirv(uniform, descriptor_set = 1, binding = 0)] proxy: &ProxyUbo,
    #[spirv(location = 0)] in_position: Vec3,
    #[spirv(location = 1)] in_normal: Vec3,
    #[spirv(location = 2)] in_tex_coord: Vec2,
    #[spirv(position)] out_position: &mut Vec4,
    #[spirv(location = 0)] frag_tex_coord: &mut Vec2,
    #[spirv(location = 1)] frag_normal: &mut Vec3,
) {
    *out_position = scene.proj * scene.view * proxy.model * in_position.extend(1.0);
    *frag_tex_coord = in_tex_coord;
    *frag_normal = in_normal;
}

/// Samples the model's texture, shaded by its normal's z component over a
/// 35% ambient floor.
#[spirv(fragment)]
pub fn main_fs(
    #[spirv(descriptor_set = 2, binding = 0)] tex_sampler: &SampledImage<Image2d>,
    #[spirv(location = 0)] frag_tex_coord: Vec2,
    #[spirv(location = 1)] frag_normal: Vec3,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    let diffuse = 0.35 + 0.65 * frag_normal.z.abs();
    *out_color = tex_sampler.sample(frag_tex_coord) * Vec4::new(diffuse, diffuse, diffuse, 1.0);
}
