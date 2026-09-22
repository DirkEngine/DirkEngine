# Renderer

The renderer owns one `dirk_rhi::Rhi` and orchestrates frame slots, final-state scene
updates, per-viewport cameras, uploads, graph execution, and presentation.
`dirk_render_utils` contains the renderer-independent graph, upload, typed binding,
buffer, and pipeline helpers. RHI backend selection is internal and compile-time:
Metal on Apple platforms, Vulkan elsewhere.

Two frame slots wait for completion before GPU buffer preparation. Resource drops
enter the RHI retirement queue, normally collected after three completed cycles.
Asset uploads are batched on the transfer queue and acquired on graphics. Asset
mipmaps use CPU sRGB filtering for consistent backend behavior. Editor rendering
uses the same RHI and graph, retaining pending texture updates while minimized.

Register `RendererPlugin` with `EngineBuilder`; it depends on `PlatformPlugin` and
`AssetsPlugin`. Rust GPU shaders produce SPIR-V and, on Apple targets, translated
MSL with shared binding metadata. Shader overhaul and transient allocation remain
deferred. See the [rendering contract](../../docs/rhi/runtime-contract.md).
