//! Metal implementation of the RHI.
//!
//! The implementation is available on Apple targets. Shader bindings use the
//! shared [`crate::BindingMap`], independently for each shader stage. Vertex
//! buffers start at Metal buffer index 16. Use [`crate::Rhi`] for shared
//! retirement, validation, and recording scopes.
//!
//! Exact-region filtered blits are currently unsupported. Callers query this
//! before recording and select an explicit fallback; whole-chain native mipmap
//! generation is not substituted for a regional operation.
//!
//! Operations creating autoreleased Objective-C objects provide their own
//! draining pools. Objects that escape those operations must be retained Rust
//! owners; callers do not need an `AppKit` event loop or an autorelease pool.

#![cfg(target_vendor = "apple")]

mod backend;
mod command;
mod convert;
mod presentation;
mod resource;

pub use backend::MetalBackend;

fn backend_error(error: impl std::fmt::Display) -> crate::Error {
    crate::Error::Backend(anyhow::anyhow!("{error}"))
}
