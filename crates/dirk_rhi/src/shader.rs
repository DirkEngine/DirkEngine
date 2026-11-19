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
/// stage and push data size it reflected.
///
/// # Safety
///
/// [`CODE`](Self::CODE) is a valid module for the target's backend holding
/// [`ENTRY_POINT`](Self::ENTRY_POINT) for [`STAGE`](Self::STAGE): SPIR-V for
/// Vulkan 1.3, or, on Apple targets, Metal Shading Language source holding
/// only that entry point, translated by the shader build. The entry point
/// only binds the bindless heap, reads a push data block of exactly
/// [`PUSH_SIZE`](Self::PUSH_SIZE) bytes (zero if it reads none), runs
/// workgroups of [`WORKGROUP_SIZE`](Self::WORKGROUP_SIZE) if it is a compute
/// shader, and uses only features the RHI enables.
pub unsafe trait Shader {
    /// Name of the entry point in the module.
    const ENTRY_POINT: &'static str;
    /// The stage the entry point runs in.
    const STAGE: ShaderStage;
    /// Size of the push data block the entry point reads, in bytes; zero if
    /// it reads none.
    const PUSH_SIZE: usize;
    /// Invocations per workgroup along each axis, for compute shaders; zeros
    /// for other stages.
    const WORKGROUP_SIZE: [u32; 3];
    /// The module: SPIR-V, or Metal Shading Language source on Apple
    /// targets.
    const CODE: &'static [u8];
}

/// A shader entry point ready for pipelines whose push data is `P`.
/// Destroyed when dropped; pipelines created from it stay valid.
///
/// Its push data block must match `P`. The check happens at compile time, so
/// a shader reading a block of another size fails to build. Like every
/// post-monomorphization error, `cargo check` and `cargo clippy` do not report
/// it; `cargo build` and `cargo test` do:
/// ```compile_fail
/// use dirk_rhi::{Result, Rhi, Shader, ShaderModule, ShaderStage};
/// struct Reads16Bytes;
/// // SAFETY: never created; this only shows the check.
/// unsafe impl Shader for Reads16Bytes {
///     const ENTRY_POINT: &'static str = "main";
///     const STAGE: ShaderStage = ShaderStage::Vertex;
///     const PUSH_SIZE: usize = 16;
///     const WORKGROUP_SIZE: [u32; 3] = [0; 3];
///     const CODE: &'static [u8] = &[];
/// }
/// fn main() -> Result<()> {
///     let rhi = Rhi::new(&Default::default())?;
///     let module: ShaderModule<[u32; 2]> = rhi.create_shader::<Reads16Bytes, _>()?;
///     Ok(())
/// }
/// ```
pub struct ShaderModule<P: PushData> {
    pub(crate) raw: native::ShaderModule,
    stage: ShaderStage,
    workgroup_size: [u32; 3],
    entry_point: &'static str,
    push: PhantomData<fn(&P)>,
}

impl Rhi {
    /// Creates a module for shader `S`, used with push data `P`.
    ///
    /// Fails to build if `S` reads a push data block other than `P`, or if
    /// `P` is larger than [`MAX_PUSH_DATA_SIZE`] or not a multiple of four
    /// bytes; see [`ShaderModule`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `P` exceeds the device's push data
    /// limit, or a native error if the module cannot be created.
    pub fn create_shader<S: Shader, P: PushData>(&self) -> Result<ShaderModule<P>> {
        const {
            assert!(
                S::PUSH_SIZE == 0 || S::PUSH_SIZE == size_of::<P>(),
                "the shader's push data block differs in size from the push data type"
            );
            assert!(
                size_of::<P>() <= MAX_PUSH_DATA_SIZE as usize && size_of::<P>().is_multiple_of(4),
                "push data is at most 128 bytes and a multiple of four bytes"
            );
        }
        let limit = self.capabilities().limits.max_push_data_size;
        if size_of::<P>() > limit as usize {
            return Err(Error::invalid(
                "shader",
                S::ENTRY_POINT,
                format!(
                    "{} bytes of push data exceed the device limit of {limit}",
                    size_of::<P>()
                ),
            ));
        }
        Ok(ShaderModule {
            raw: native::ShaderModule::new(&self.device, S::ENTRY_POINT, S::CODE)?,
            stage: S::STAGE,
            workgroup_size: S::WORKGROUP_SIZE,
            entry_point: S::ENTRY_POINT,
            push: PhantomData,
        })
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

    /// Invocations per workgroup along each axis, for compute shaders; zeros
    /// for other stages.
    #[must_use]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}
