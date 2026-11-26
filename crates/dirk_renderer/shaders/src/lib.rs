//! The renderer's shaders, compiled by rust-gpu. They read every resource
//! through the bindless heap, pull vertices from storage buffers, and share
//! their push data types with the renderer through `dirk_shaders::types`.
//!
//! Colors are linear inside the renderer. Images with sRGB formats encode
//! and decode them in hardware; the presentation shaders encode them
//! themselves for windows without an sRGB format, so colors are encoded
//! exactly once.

#![no_std]

use dirk_shaders::types::{EguiPush, EguiVertex, MeshPush, MeshVertex, ObjectData, PresentPush};
use spirv_std::{
    Image, RuntimeArray, Sampler, TypedBuffer,
    glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4},
    spirv,
};
// Float functions on the GPU; host builds of the crate get them from `core`.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// The heap's 2D textures, at `dirk_shaders::bindless::heap::TEXTURES_2D`.
type Textures = RuntimeArray<Image!(2D, type = f32, sampled)>;

/// Places vertex `vertex_index` of `push.vertices`, transformed by the
/// object `instance_index` of `push.objects`.
#[expect(
    clippy::too_many_arguments,
    reason = "Shader entry points take one argument per input, output and binding."
)]
#[spirv(vertex)]
pub fn mesh_vs(
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(instance_index)] instance_index: i32,
    #[spirv(push_constant)] push: &MeshPush,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] vertices: &RuntimeArray<
        TypedBuffer<[MeshVertex]>,
    >,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] objects: &RuntimeArray<
        TypedBuffer<[ObjectData]>,
    >,
    #[spirv(position)] position: &mut Vec4,
    uv: &mut Vec2,
    normal: &mut Vec3,
) {
    let vertex = push.vertices.get(vertices)[vertex_index.cast_unsigned() as usize];
    let object = *push.objects.get(objects, instance_index.cast_unsigned());
    let model = Mat4::from_cols_array_2d(&object.model);
    let view_projection = Mat4::from_cols_array_2d(&push.view_projection);
    *position = view_projection * model * Vec3::from_array(vertex.position).extend(1.0);
    *uv = Vec2::from_array(vertex.uv);
    *normal = Vec3::from_array(vertex.normal);
}

/// Samples the base color texture, shaded by how much the model-space
/// normal faces the viewer's axis.
#[spirv(fragment)]
pub fn mesh_fs(
    #[spirv(push_constant)] push: &MeshPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    normal: Vec3,
    color: &mut Vec4,
) {
    let diffuse = 0.35 + 0.65 * normal.z.abs();
    let base: Vec4 = push
        .texture
        .get(textures)
        .sample(*push.sampler.get(samplers), uv);
    *color = base * vec4(diffuse, diffuse, diffuse, 1.0);
}

/// Places vertex `vertex_index` of `push.vertices`, given in points, and
/// passes on its color as linear and premultiplied.
#[spirv(vertex)]
pub fn egui_vs(
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(push_constant)] push: &EguiPush,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] vertices: &RuntimeArray<
        TypedBuffer<[EguiVertex]>,
    >,
    #[spirv(position)] position: &mut Vec4,
    uv: &mut Vec2,
    color: &mut Vec4,
) {
    let vertex = *push.vertices.get(vertices, vertex_index.cast_unsigned());
    let clip = 2.0 * Vec2::from_array(vertex.position) / Vec2::from_array(push.screen_size) - 1.0;
    *position = vec4(clip.x, clip.y, 0.0, 1.0);
    *uv = Vec2::from_array(vertex.uv);
    *color = linear_premultiplied(unpack_color(vertex.color));
}

/// Modulates the texture, whose colors are linear and premultiplied, by the
/// vertex color.
#[spirv(fragment)]
pub fn egui_fs(
    #[spirv(push_constant)] push: &EguiPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    color: Vec4,
    output: &mut Vec4,
) {
    let texel: Vec4 = push
        .texture
        .get(textures)
        .sample(*push.sampler.get(samplers), uv);
    *output = color * texel;
}

/// Covers the target with one triangle; draw three vertices.
#[spirv(vertex)]
pub fn present_vs(
    #[spirv(vertex_index)] index: i32,
    #[spirv(position)] position: &mut Vec4,
    uv: &mut Vec2,
) {
    // Texture coordinates (0, 0), (2, 0) and (0, 2).
    #[expect(
        clippy::cast_precision_loss,
        reason = "The coordinates are 0 or 2, exact in an f32."
    )]
    let corner = vec2(((index << 1) & 2) as f32, (index & 2) as f32);
    *uv = corner;
    *position = vec4(2.0 * uv.x - 1.0, 2.0 * uv.y - 1.0, 0.0, 1.0);
}

/// Shows the source's linear colors on a target with an sRGB format, which
/// encodes them.
#[spirv(fragment)]
pub fn present_fs(
    #[spirv(push_constant)] push: &PresentPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    output: &mut Vec4,
) {
    *output = present(*push, textures, samplers, uv).extend(1.0);
}

/// Shows the source's linear colors on a target without an sRGB format,
/// encoding them first.
#[spirv(fragment)]
pub fn present_encode_fs(
    #[spirv(push_constant)] push: &PresentPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
    #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    output: &mut Vec4,
) {
    let linear = present(*push, textures, samplers, uv);
    *output = vec4(
        srgb_from_linear(linear.x),
        srgb_from_linear(linear.y),
        srgb_from_linear(linear.z),
        1.0,
    );
}

/// The source's linear color at `uv`.
fn present(
    push: PresentPush,
    textures: &Textures,
    samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
) -> Vec3 {
    let color: Vec4 = push
        .source
        .get(textures)
        .sample(*push.sampler.get(samplers), uv);
    color.xyz()
}

/// The bytes of `color`, red in the lowest, as channels in `0.0..=1.0`.
fn unpack_color(color: u32) -> Vec4 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "Channels are 8-bit, exact in an f32."
    )]
    let channel = |shift: u32| ((color >> shift) & 0xff) as f32 / 255.0;
    vec4(channel(0), channel(8), channel(16), channel(24))
}

/// The linear, premultiplied color of egui's sRGB-encoded, premultiplied
/// `color`, like `ecolor::Rgba::from(Color32)`: the color is divided by its
/// alpha before decoding. A zero alpha marks an additive color, decoded as
/// it is.
fn linear_premultiplied(color: Vec4) -> Vec4 {
    let alpha = color.w;
    let straight = if alpha > 0.0 {
        color.xyz() / alpha
    } else {
        color.xyz()
    };
    let linear = vec3(
        linear_from_srgb(straight.x),
        linear_from_srgb(straight.y),
        linear_from_srgb(straight.z),
    );
    (if alpha > 0.0 { linear * alpha } else { linear }).extend(alpha)
}

/// Decodes an sRGB-encoded channel.
fn linear_from_srgb(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Encodes a linear channel with the sRGB curve.
fn srgb_from_linear(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}
