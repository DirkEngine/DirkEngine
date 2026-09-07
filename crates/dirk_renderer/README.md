# renderer

The renderer owns GPU rendering for the engine. Resource creation, uploads,
presentation, synchronization, and render-graph execution go through
`dirk_rhi`. The renderer selects `dirk_rhi_metal` on Apple platforms and
`dirk_rhi_vulkan` elsewhere at compile time. Renderer code uses the selected
backend directly through the shared RHI contract, so there is no runtime
backend dispatch.

Shaders are compiled to SPIR-V by Rust GPU. Vulkan consumes that SPIR-V
directly, while Apple builds also translate it to Metal Shading Language.

The editor remains temporarily Vulkan-specific and is disabled by default.
Enable the `editor` feature only when building the Vulkan renderer.

Register `RendererPlugin` with an `EngineBuilder` to install the renderer
subsystem and its ECS integration systems. The plugin depends on
`PlatformPlugin` and `AssetsPlugin`, passes engine metadata to the active
backend, and renders once per engine tick.
