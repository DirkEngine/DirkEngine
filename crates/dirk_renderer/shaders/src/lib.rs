#![no_std]
// shaders often have a lot of different inputs, we can't help it
#![allow(clippy::too_many_arguments)]

use spirv_std::{
    glam::{Vec2, Vec3, Vec4},
    image::{Image2d, SampledImage},
    num_traits::Float,
    spirv,
};

use dirk_shaders::types::{EguiUbo, ProxyUbo, SceneUbo};

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

#[spirv(vertex)]
pub fn egui_vs(
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &EguiUbo,
    #[spirv(location = 0)] in_position: Vec2,
    #[spirv(location = 1)] in_tex_coord: Vec2,
    #[spirv(location = 2)] in_color: Vec4,
    #[spirv(position)] out_position: &mut Vec4,
    #[spirv(location = 0)] frag_tex_coord: &mut Vec2,
    #[spirv(location = 1)] frag_color: &mut Vec4,
) {
    let position = in_position / frame.screen_size;
    *out_position = Vec4::new(2.0 * position.x - 1.0, 2.0 * position.y - 1.0, 0.0, 1.0);
    *frag_tex_coord = in_tex_coord;
    *frag_color = Vec4::new(
        Float::powf(in_color.x, 2.2),
        Float::powf(in_color.y, 2.2),
        Float::powf(in_color.z, 2.2),
        in_color.w,
    );
}

#[spirv(fragment)]
pub fn egui_fs(
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &EguiUbo,
    #[spirv(descriptor_set = 1, binding = 0)] tex_sampler: &SampledImage<Image2d>,
    #[spirv(location = 0)] frag_tex_coord: Vec2,
    #[spirv(location = 1)] frag_color: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    let color = frag_color * tex_sampler.sample(frag_tex_coord);
    *out_color = if frame.output_is_srgb > 0.5 {
        color
    } else {
        Vec4::new(
            Float::powf(color.x, 1.0 / 2.2),
            Float::powf(color.y, 1.0 / 2.2),
            Float::powf(color.z, 1.0 / 2.2),
            color.w,
        )
    };
}
