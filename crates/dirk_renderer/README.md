# Renderer

The renderer owns one `dirk_rhi::Rhi` and orchestrates frame slots, final-state scene
updates, per-viewport cameras, uploads, graph execution, and presentation.
`dirk_render_utils` contains the renderer-independent graph, upload, typed binding,
buffer, and pipeline helpers. RHI backend selection is internal and compile-time:
Metal on Apple platforms, Vulkan elsewhere.

Model loading uploads each glTF mesh once in its original coordinates. Scene
selection, node transforms, and repeated node instances are not applied; entity
transforms provide world placement. Triangle lists, strips, and fans, base-color
textures with their glTF sampler and UV set, base-color factors, and vertex colors
are supported. Missing indices are generated; missing normals split shared
vertices to preserve flat shading. Materials retain opaque, back-face-culled
rendering even when glTF requests alpha blending or double-sided rendering.
Base-color images in 8- or 16-bit integer formats are converted to RGBA8.
Unsupported content never stops the renderer: point and line primitives and
malformed primitives are skipped, float images are replaced by a white texture,
each with a warning, and models that still fail to load are logged and skipped.

Two frame slots wait for completion before GPU buffer preparation. Resource drops
enter the RHI retirement queue, normally collected after three completed cycles.
Asset uploads are batched on the transfer queue and acquired on graphics. Texture
mipmaps are blitted on the graphics queue in linear light when the device supports
filtered blits of the texture format; otherwise, as on Metal, they are filtered on
the CPU. Viewports are only rendered while a window that shows them has acquired
an image. Editor rendering uses the same RHI and graph, retaining pending texture
updates while minimized.

Register `RendererPlugin` with `EngineBuilder`; it depends on `PlatformPlugin` and
`AssetsPlugin`. Rust GPU shaders produce SPIR-V and, on Apple targets, translated
MSL with shared binding metadata. Shader overhaul and transient allocation remain
deferred. See the [rendering contract](../../docs/rhi/runtime-contract.md).

## Output colors

Scene shaders produce linear colors. Viewport images use an sRGB format
independent of the window surface and preserve that meaning when sampled. Window creation prefers
sRGB attachments. Standalone presentation samples the viewport and encodes exactly
once: the attachment performs the conversion for sRGB formats, and a fragment
shader performs it for UNORM formats. Editor output uses the same sRGB transfer
function. Presentation pipelines follow each window's format across recreation.
