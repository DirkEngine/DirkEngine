# renderer

The renderer draws the engine's worlds and the editor UI through the engine's
render hardware interface, `dirk_rhi`, and orders each frame's passes with
`dirk_render_utils`' render graph. It has no native graphics code of its own.

Each frame, the renderer:

1. uploads newly loaded models and egui textures with `dirk_render_utils`'
   `Uploads`, staged in the frame's memory: models wait for later frames once
   a frame staged its upload budget, so a large scene loads over several
   frames;
2. acquires an image of every visible window, and stops there when none is
   visible;
3. renders the viewports some window shows, each into its own output image,
   with one depth and multisampled color image per viewport;
4. draws every acquired window: the editor composites its UI, the player
   build shows the window's viewport;
5. submits all of it at once, in one render graph, and presents the windows.

Shaders are Rust, in `shaders/`, compiled by `dirk_build::shaders`. They read
every resource through the bindless heap and pull vertices from storage
buffers; push data types are shared with the renderer through
`dirk_shaders::types`. Per-object data and egui's vertices and indices are
allocated in the frame's memory each frame; meshes index their objects by
instance index.

Colors are linear throughout: the renderer's images are sRGB-encoded and
decoded in hardware, and presentation encodes once for windows without an
sRGB format. egui's sRGB-encoded, premultiplied colors are converted to linear
ones before compositing.

Debug builds request the RHI's validation: the Vulkan validation layer when it
is installed, or on Apple platforms Metal's, enabled with `MTL_DEBUG_LAYER=1`.

Register `RendererPlugin` with an `EngineBuilder` to install the renderer
subsystem and its universe integration. The plugin depends on
`PlatformPlugin`, `AssetsPlugin` and `PlayerPlugin`, and renders once per
engine tick.
