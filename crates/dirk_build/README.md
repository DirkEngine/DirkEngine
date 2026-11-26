# build

This crate has a bunch of utilities for the engine build scripts.
This mainly avoids duplicating path setup across build scripts.
It also builds rust-gpu shader crates into SPIR-V, translated to MSL for Apple
targets, with the metadata `dirk_rhi` checks at compile time; see
`shaders::build_shaders`.
