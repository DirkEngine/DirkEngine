use std::marker::PhantomData;

use bytemuck::NoUninit;

use crate::{Error, Result, Rhi, ShaderStages, native};

/// Largest push data any pipeline may use, in bytes. Every device supports
/// it.
pub const MAX_PUSH_DATA_SIZE: u32 = 128;

/// Data pushed to a pipeline's shaders with each draw: a `#[repr(C)]` struct
/// shared with the shaders, without padding, at most [`MAX_PUSH_DATA_SIZE`]
/// bytes and a multiple of four bytes long. Pipelines without push data use
/// `()`.
pub trait PushData: NoUninit {}

impl<T: NoUninit> PushData for T {}

/// The stage a shader entry point runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Runs once per vertex.
    Vertex,
    /// Runs once per fragment.
    Fragment,
    /// Runs once per compute invocation.
    Compute,
}

impl ShaderStage {
    /// This stage as a flag in a set of stages.
    #[must_use]
    pub const fn flag(self) -> ShaderStages {
        match self {
            Self::Vertex => ShaderStages::VERTEX,
            Self::Fragment => ShaderStages::FRAGMENT,
            Self::Compute => ShaderStages::COMPUTE,
        }
    }
}

/// A shader entry point compiled by the engine's shader build, which
/// generates one type implementing this trait per entry point, with the
/// stage and push data size it reflected and the push data type its source
/// reads.
///
/// # Safety
///
/// [`SPIRV`](Self::SPIRV) is a valid SPIR-V module for Vulkan 1.3 declaring
/// [`ENTRY_POINT`](Self::ENTRY_POINT) for [`STAGE`](Self::STAGE). The entry
/// point only binds the bindless heap, reads its push data block as
/// [`Push`](Self::Push), a block of exactly [`PUSH_SIZE`](Self::PUSH_SIZE)
/// bytes (zero if it reads none), and uses only features the RHI enables.
pub unsafe trait Shader {
    /// The push data the entry point reads; `()` if it reads none.
    type Push: PushData;
    /// Name of the entry point in the module.
    const ENTRY_POINT: &'static str;
    /// The stage the entry point runs in.
    const STAGE: ShaderStage;
    /// Size of the push data block the entry point reads, in bytes; zero if
    /// it reads none.
    const PUSH_SIZE: usize;
    /// The SPIR-V module, as bytes.
    const SPIRV: &'static [u8];
}

/// A shader entry point ready for pipelines whose push data is `P`, the
/// push data type of its [`Shader`]. Destroyed when dropped; pipelines
/// created from it stay valid.
///
/// Pipelines take shaders of one push data type, so a shader reading
/// another type does not compile:
/// ```compile_fail
/// use dirk_rhi::{Result, Rhi, Shader, ShaderModule, ShaderStage};
/// struct Reads16Bytes;
/// // SAFETY: never created; this only shows the check.
/// unsafe impl Shader for Reads16Bytes {
///     type Push = [u32; 4];
///     const ENTRY_POINT: &'static str = "main";
///     const STAGE: ShaderStage = ShaderStage::Vertex;
///     const PUSH_SIZE: usize = 16;
///     const SPIRV: &'static [u8] = &[];
/// }
/// fn main() -> Result<()> {
///     let rhi = Rhi::new(&Default::default())?;
///     let module: ShaderModule<[u32; 2]> = rhi.create_shader::<Reads16Bytes>()?;
///     Ok(())
/// }
/// ```
///
/// The push data block the shader build reflected must also fit
/// [`Shader::Push`]. That check happens after monomorphization, so like every
/// such error, `cargo check` and `cargo clippy` do not report it; `cargo
/// build` and `cargo test` do:
/// ```compile_fail
/// use dirk_rhi::{Result, Rhi, Shader, ShaderStage};
/// struct Reads16Bytes;
/// // SAFETY: never created; this only shows the check.
/// unsafe impl Shader for Reads16Bytes {
///     type Push = [u32; 2];
///     const ENTRY_POINT: &'static str = "main";
///     const STAGE: ShaderStage = ShaderStage::Vertex;
///     const PUSH_SIZE: usize = 16;
///     const SPIRV: &'static [u8] = &[];
/// }
/// fn main() -> Result<()> {
///     let rhi = Rhi::new(&Default::default())?;
///     rhi.create_shader::<Reads16Bytes>()?;
///     Ok(())
/// }
/// ```
pub struct ShaderModule<P: PushData> {
    pub(crate) raw: native::ShaderModule,
    stage: ShaderStage,
    entry_point: &'static str,
    push: PhantomData<fn(&P)>,
}

impl Rhi {
    /// Creates a module for shader `S`, used with its push data type.
    ///
    /// Fails to build if `S` reads a push data block of another size than
    /// [`Shader::Push`], or if that type is larger than
    /// [`MAX_PUSH_DATA_SIZE`] or not a multiple of four bytes; see
    /// [`ShaderModule`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the push data exceeds the device's push
    /// data limit, or a native error if the module cannot be created.
    pub fn create_shader<S: Shader>(&self) -> Result<ShaderModule<S::Push>> {
        let size = size_of::<S::Push>();
        const {
            // A shader that never reads its push data has no block.
            assert!(
                S::PUSH_SIZE == 0 || S::PUSH_SIZE == size_of::<S::Push>(),
                "the shader's push data block differs in size from its push data type"
            );
            assert!(
                size_of::<S::Push>() <= MAX_PUSH_DATA_SIZE as usize
                    && size_of::<S::Push>().is_multiple_of(4),
                "push data is at most 128 bytes and a multiple of four bytes"
            );
        }
        let limit = self.capabilities().limits.max_push_data_size;
        if size > limit as usize {
            return Err(Error::invalid(
                "shader",
                S::ENTRY_POINT,
                format!("{size} bytes of push data exceed the device limit of {limit}"),
            ));
        }
        Ok(ShaderModule {
            raw: native::ShaderModule::new(&self.device, S::ENTRY_POINT, S::SPIRV)?,
            stage: S::STAGE,
            entry_point: S::ENTRY_POINT,
            push: PhantomData,
        })
    }
}

impl ShaderModule<()> {
    /// This module, whose shader reads no push data, for pipelines whose
    /// other shaders read `P`.
    #[must_use]
    pub fn with_push<P: PushData>(self) -> ShaderModule<P> {
        ShaderModule {
            raw: self.raw,
            stage: self.stage,
            entry_point: self.entry_point,
            push: PhantomData,
        }
    }
}

impl<P: PushData> ShaderModule<P> {
    /// The stage the entry point runs in.
    #[must_use]
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    /// Name of the entry point.
    #[must_use]
    pub fn entry_point(&self) -> &'static str {
        self.entry_point
    }
}
