# `DirkEngine`

A Vulkan Game Engine in Rust

On macOS, the Vulkan renderer uses `MoltenVK` through the Vulkan loader. The
engine does not install or bundle a Vulkan SDK or `MoltenVK`; provide the
Vulkan loader and configure it to discover the `MoltenVK` ICD at runtime. Native
Metal support is available through `dirk_rhi_metal`; the engine renderer remains
on the Vulkan backend while its typed pipeline and egui compatibility seams are
migrated.

Debug builds also enable `VK_LAYER_KHRONOS_validation`; make its Vulkan layer
manifest discoverable with `VK_ADD_LAYER_PATH` (or a standard Vulkan layer
search path).

The engine is assembled with plugins and runtime subsystems:

```rust,no_run
# fn main() -> anyhow::Result<()> {
let mut builder = dirk_engine::Engine::builder();
builder.with_plugin(dirk_assets::AssetsPlugin)?;
builder.with_plugin(dirk_platform::PlatformPlugin)?;
builder.with_plugin(dirk_player::PlayerPlugin)?;
builder.with_plugin(dirk_world::WorldPlugin)?;
builder.with_plugin(dirk_renderer::RendererPlugin)?;

let engine = builder.build()?;
engine.run()?;
# Ok(()) }
```
