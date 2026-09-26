# Renderer

The renderer owns one `dirk_rhi::Rhi` and orchestrates frame slots, final-state scene
updates, per-viewport cameras, uploads, graph execution, and presentation.
`dirk_render_utils` contains the renderer-independent graph, upload, typed binding,
buffer, and pipeline helpers. RHI backend selection is internal and compile-time:
Metal on Apple platforms, Vulkan elsewhere.

Model loading uploads each glTF mesh once in its original coordinates. Scene
selection, node transforms, and repeated node instances are not applied; entity
transforms provide world placement. Triangle primitives, base-color textures with
the default repeat/trilinear sampler and UV set 0, base-color factors, and vertex
colors are supported. Missing indices are generated; missing normals split shared
vertices to preserve flat shading. Materials retain opaque, back-face-culled
rendering even when glTF requests alpha blending or double-sided rendering.
Other topology and custom base-color texture coordinates or sampler settings
return a load error. Base-color images in 8- or 16-bit integer formats are converted
to RGBA8; float images are unsupported.

Two frame slots wait for completion before GPU buffer preparation. Resource drops
enter the RHI retirement queue, normally collected after three completed cycles.
Asset uploads are batched on the transfer queue and acquired on graphics. Asset
mipmaps use CPU sRGB filtering for consistent backend behavior. Editor rendering
uses the same RHI and graph, retaining pending texture updates while minimized.

Register `RendererPlugin` with `EngineBuilder`; it depends on `PlatformPlugin` and
`AssetsPlugin`. Rust GPU shaders produce SPIR-V and, on Apple targets, translated
MSL with shared binding metadata. Shader overhaul and transient allocation remain
deferred. See the [rendering contract](../../docs/rhi/runtime-contract.md).

## Output colors

Scene shaders produce linear colors. Viewport images preserve that meaning when
sampled, using hardware decoding for sRGB image formats. Window creation prefers
sRGB attachments. Standalone presentation samples the viewport and encodes exactly
once: the attachment performs the conversion for sRGB formats, and a fragment
shader performs it for UNORM formats. Editor output uses the same sRGB transfer
function. Presentation pipelines follow each window's format across recreation.
