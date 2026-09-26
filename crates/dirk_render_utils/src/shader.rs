//! Shader imports and reflected layout contracts.
use dirk_rhi::{BindGroupLayoutEntry, ShaderStage, VertexBufferLayout};
/// A block of shader bytecode and the shader entry point name.
pub struct ShaderCode {
    #[cfg(not(target_vendor = "apple"))]
    /// SPIR-V bytes generated for the selected Vulkan backend.
    pub spirv: &'static [u8],
    #[cfg(target_vendor = "apple")]
    /// Metal source generated alongside its reflected metadata.
    pub msl: &'static str,
}

impl ShaderCode {
    /// Imports native shader code for a named entry point.
    ///
    /// # Safety
    /// Code must satisfy `Rhi::create_shader`: valid native code, matching stage,
    /// entry point and pipeline bindings, with shader accesses within resource bounds.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn create(
        &self,
        device: &dirk_rhi::Rhi,
        stage: dirk_rhi::ShaderStage,
        entry: &'static str,
    ) -> dirk_rhi::Result<dirk_rhi::Shader> {
        #[cfg(not(target_vendor = "apple"))]
        let words = self.code_as_u32()?;
        #[cfg(not(target_vendor = "apple"))]
        let source = dirk_rhi::ShaderSource::SpirV(&words);
        #[cfg(target_vendor = "apple")]
        let source = dirk_rhi::ShaderSource::Msl(self.msl);
        // SAFETY: trusted engine shaders are compiled and reflected together by build.rs.
        unsafe {
            device.create_shader(&dirk_rhi::ShaderDesc {
                label: entry,
                stage,
                entry,
                source,
            })
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    fn code_as_u32(&self) -> dirk_rhi::Result<Vec<u32>> {
        if !self.spirv.len().is_multiple_of(4) {
            return Err(dirk_rhi::InvalidResourceKind::Mismatch
                .with_detail("SPIR-V size must be a multiple of four bytes")
                .into());
        }
        Ok(self
            .spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect())
    }
}

/// Reflected metadata shared by all shader stages.
///
/// # Safety
/// Code and metadata must be produced and validated together. The code must obey
/// `Rhi::create_shader` and all reflected layouts must match its actual interface.
pub unsafe trait Shader {
    /// Native shader code.
    const CODE: ShaderCode;
    /// Entry point exported by the code.
    const ENTRYPOINT: &'static str;
    /// Stage implemented by the entry point.
    const STAGE: ShaderStage;
    /// Reflected descriptor layouts in set order.
    const SET_LAYOUTS: &'static [&'static [BindGroupLayoutEntry]];

    /// Creates the shader from its trusted metadata.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    fn create(device: &dirk_rhi::Rhi) -> dirk_rhi::Result<dirk_rhi::Shader> {
        unsafe { Self::CODE.create(device, Self::STAGE, Self::ENTRYPOINT) }
    }
}

/// Shader with a reflected vertex input interface.
pub trait VertexShader: Shader {
    /// Reflected vertex buffers in binding order.
    const INPUT_LAYOUTS: &'static [VertexBufferLayout<'static>];
}

/// Fragment-stage shader marker.
pub trait FragmentShader: Shader {}
