//! Compiled shader blobs used by the renderer.

macro_rules! shader_code {
    ($name:literal) => {
        ShaderCode {
            #[cfg(not(target_vendor = "apple"))]
            spirv: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")),
            #[cfg(target_vendor = "apple")]
            msl: include_str!(concat!(env!("OUT_DIR"), "/", $name, ".metal")),
        }
    };
}

pub mod metadata;

pub use dirk_render_utils::shader::ShaderCode;

include!(concat!(env!("OUT_DIR"), "/generated_shaders.rs"));

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;

    #[test]
    fn metal_vertex_shader_flips_vulkan_clip_space_y() {
        let source = <MainVS as metadata::Shader>::CODE.msl;

        assert!(source.contains("Invert Y-axis for Metal"));
    }
}
