# renderer

The renderer owns GPU rendering for the engine. Resource creation, uploads,
presentation, synchronization, and render-graph execution go through
`dirk_rhi`. The renderer selects `dirk_rhi_metal` on Apple platforms and
`dirk_rhi_vulkan` elsewhere at compile time. Renderer code uses `Rhi<ActiveBackend>` for shared validation and ownership,
with static backend dispatch. Graphics callbacks borrow a render-pass scope;
transfer callbacks record outside it. Submission consumes finished recordings
and returns completion tokens, so frame code does not retain native commands.

Shaders are compiled to SPIR-V by Rust GPU. Vulkan consumes that SPIR-V
directly, while Apple builds also translate it to Metal Shading Language.

The optional editor painter also uses the shared RHI, including its managed
egui textures and renderer viewport images, so it works with either backend.

Register `RendererPlugin` with an `EngineBuilder` to install the renderer
subsystem and its ECS integration systems. The plugin depends on
`PlatformPlugin` and `AssetsPlugin`, passes engine metadata to the active
backend, and renders once per engine tick.

Asset and egui texture updates use one upload helper for aligned staging rows.
Asset mipmaps use exact filtered blits when supported, with CPU sRGB mip
generation as the explicit fallback. Image imports obtain allocation metadata
from their RHI handles. See [the RHI runtime contract](../../docs/rhi-runtime-contract.md)
for guarantees and current scope.
