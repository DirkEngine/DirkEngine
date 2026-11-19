# `DirkEngine`

A Vulkan and Metal Game Engine in Rust

The renderer uses Metal 3 on Apple platforms and Vulkan 1.3 everywhere else;
`MoltenVK` is not supported. Apple builds translate the engine's shaders to the
Metal Shading Language at build time, which needs an Apple host.

Debug builds also enable the graphics API's validation. On Vulkan, that is
`VK_LAYER_KHRONOS_validation`; make its Vulkan layer manifest discoverable with
`VK_ADD_LAYER_PATH` (or a standard Vulkan layer search path). On Metal, run with
`MTL_DEBUG_LAYER=1`, and `MTL_SHADER_VALIDATION=1` to validate shaders too.

The engine is assembled with plugins and runtime subsystems:

```rust,no_run
# fn main() -> anyhow::Result<()> {
let mut builder = dirk_engine::Engine::builder();
// Let Ctrl-C request a graceful shutdown. Disabled by default so libraries
// embedding the engine keep the host's signal handling.
builder.with_os_signals(true);
builder.with_plugin(dirk_assets::AssetsPlugin)?;
builder.with_plugin(dirk_platform::PlatformPlugin)?;
builder.with_plugin(dirk_player::PlayerPlugin)?;
builder.with_plugin(dirk_world::WorldPlugin)?;
builder.with_plugin(dirk_renderer::RendererPlugin)?;

let engine = builder.build()?;
engine.run()?;
# Ok(()) }
```
