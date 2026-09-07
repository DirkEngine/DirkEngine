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

/// A block of shader bytecode and the shader entry point name.
pub(crate) struct ShaderCode {
    #[cfg(not(target_vendor = "apple"))]
    spirv: &'static [u8],
    #[cfg(target_vendor = "apple")]
    msl: &'static str,
}

impl ShaderCode {
    fn create(
        &self,
        device: &crate::resources::device::RenderDevice,
        stage: dirk_rhi::ShaderStage,
        entry: &'static str,
    ) -> crate::Result<crate::resources::ActiveShader> {
        #[cfg(not(target_vendor = "apple"))]
        let words = self.code_as_u32();
        #[cfg(not(target_vendor = "apple"))]
        let source = dirk_rhi::ShaderSource::SpirV(&words);
        #[cfg(target_vendor = "apple")]
        let source = dirk_rhi::ShaderSource::Msl(self.msl);
        // SAFETY: trusted engine shaders are compiled and reflected together by build.rs.
        Ok(unsafe {
            device.rhi.create_shader(&dirk_rhi::ShaderDesc {
                label: entry,
                stage,
                entry,
                source,
            })
        }?)
    }

    /// Returns the shader code as little-endian `u32` words.
    ///
    /// # Panics
    ///
    /// Panics if the SPIR-V code size is not a multiple of 4 bytes.
    #[must_use]
    #[cfg(not(target_vendor = "apple"))]
    fn code_as_u32(&self) -> Vec<u32> {
        assert!(
            self.spirv.len().is_multiple_of(4),
            "SPIR-V size must be a multiple of 4"
        );
        self.spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }
}

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
