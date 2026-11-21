//! Builds the renderer's shaders with the engine's shader build.

fn main() -> anyhow::Result<()> {
    dirk_build::shaders::build_shaders("shaders")
}
