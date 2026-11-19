//! Shared types used by the engine's shaders.

#![no_std]

// Lets `#[derive(HeapIndices)]` name this crate from inside it.
extern crate self as dirk_shaders;

pub mod bindless;
pub mod indirect;
pub mod types;
