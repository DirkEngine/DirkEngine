//! Compiled shader blobs used by the renderer.

use dirk_rhi::Backend as _;

macro_rules! shader_code {
    ($name:literal) => {
        ShaderCode {
            spirv: include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")),
        }
    };
}

pub mod metadata;

/// A block of shader bytecode and the shader entry point name.
pub(crate) struct ShaderCode {
    spirv: &'static [u8],
}

impl ShaderCode {
    fn create(
        &self,
        device: &crate::resources::device::RenderDevice,
        stage: dirk_rhi::ShaderStage,
        entry: &'static str,
    ) -> crate::Result<crate::resources::ActiveShader> {
        let words = self.code_as_u32();
        Ok(device.rhi.create_shader(&dirk_rhi::ShaderDesc {
            label: entry,
            stage,
            entry,
            source: dirk_rhi::ShaderSource::SpirV(&words),
        })?)
    }

    /// Returns the shader code as little-endian `u32` words.
    ///
    /// # Panics
    ///
    /// Panics if the SPIR-V code size is not a multiple of 4 bytes.
    #[must_use]
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
