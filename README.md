# `DirkEngine`

A portable game engine in Rust

The renderer uses native Metal on Apple platforms and Vulkan elsewhere through
the shared `dirk_rhi` contract. Scene rendering, presentation, and the egui
editor painter follow the same backend-neutral path.

Vulkan debug builds enable `VK_LAYER_KHRONOS_validation`; make its layer
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
