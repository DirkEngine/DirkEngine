//! Compiled shader blobs used by the renderer.

macro_rules! shader_code {
    ($name:literal) => {
        ShaderCode {
            code: (include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))),
        }
    };
}

pub mod metadata;

/// A block of shader bytecode and the shader entry point name.
pub(crate) struct ShaderCode {
    code: &'static [u8],
}

impl ShaderCode {
    fn create_module(&self, device: &ash::Device) -> crate::Result<ash::vk::ShaderModule> {
        let code = self.code_as_u32();
        let info = ash::vk::ShaderModuleCreateInfo::default().code(code.as_slice());
        Ok(unsafe { device.create_shader_module(&info, None)? })
    }

    /// Returns the shader code as little-endian `u32` words.
    ///
    /// # Panics
    ///
    /// Panics if the SPIR-V code size is not a multiple of 4 bytes.
    #[must_use]
    fn code_as_u32(&self) -> Vec<u32> {
        assert!(
            self.code.len().is_multiple_of(4),
            "SPIR-V size must be a multiple of 4"
        );
        self.code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }
}

include!(concat!(env!("OUT_DIR"), "/generated_shaders.rs"));
