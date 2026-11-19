//! Shaders for `dirk_rhi`'s native tests, built by `dirk_build` exactly like
//! the engine's, so the tests also cover the shader build and its push data
//! check, and the setup those tests share with the crates built on the RHI.

pub use dirk_rhi_test_shaders_gpu::{Fill, Textured, Vertex};

#[expect(
    unsafe_code,
    reason = "the shader build vouches for the SPIR-V it generates `Shader` implementations for"
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

pub use dirk_rhi_test_shaders_gpu::Instanced;
pub use generated::*;

pub mod support;
