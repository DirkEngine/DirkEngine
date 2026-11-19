# DirkEngine rendering architecture (target design for the rebuilt renderer stack)

This is the binding design for every PR in the rebuilt renderer stack. The PR adding
`dirk_rhi` commits it; later PRs update it when they land a piece, and must not
contradict it without saying so in their PR description.

Goal: **simple and intuitive, yet very powerful and performant.** When in doubt, delete.

## Crates

| Crate | Owns |
| --- | --- |
| `dirk_rhi` | Device, resources, frames, queues, command recording, pipelines, bindless heap, presentation. One backend per target. |
| `dirk_shaders` | `no_std` types shared by host code and rust-gpu shaders: bindless index types, push-data structs, GPU records. Compiled for both. |
| `dirk_render_utils` | Render graph (barrier derivation) and staged uploads. Nothing else. |
| `dirk_renderer` | Engine integration: scenes, viewports, windows, models, egui. |

## Backend selection: no traits, no generics, no aliases

- Exactly one native backend per target, chosen with `cfg`:
  ```rust
  #[cfg_attr(target_vendor = "apple", path = "metal/mod.rs")]
  #[cfg_attr(not(target_vendor = "apple"), path = "vulkan/mod.rs")]
  mod native;
  ```
  Apple targets use Metal 3 (see [Metal](#metal)); every other target uses Vulkan 1.3.
  MoltenVK is not supported.
- There is **no** `Api`/`Backend`/`Native*` trait, **no** `B: Backend` type parameter,
  **no** `Object<B, T, M>` wrapper, **no** `pub type X = api::GpuX<Selected>` aliases.
- Each native module exposes the same `pub(crate)` concrete types and inherent methods.
  Cross-target `cargo clippy --target aarch64-apple-darwin` is the conformance check.
- Public types are defined once in shared code and hold the native object plus the
  immutable info needed for validation:
  ```rust
  pub struct Buffer { raw: native::Buffer, info: BufferInfo }
  ```
  Native types are plain structs over native handles. No `XInner` newtype layers. At most
  one `Arc` per object (the shared device context, needed by `Drop`), never nested.
- GPU objects are not `Clone`. Small value types (`Completion`, bindless indices) are `Copy`.
- An image owns its view. `create_image` creates a view of every mip and layer when the
  usage reads or writes the image through one (sampled, storage, attachment); transfer-only
  images get none. `image.view()` borrows it, so a view can never outlive its image without
  a second `Arc`. Subresource views arrive with per-layer/per-mip graph passes.

## Validation and errors

- Validate **once**, at the safe boundary, at **creation** time wherever possible
  (descriptor ranges, usage flags, limits such as vertex/attachment/push-constant limits,
  view usage, format capabilities). The native layer does not re-validate.
- Recording-time checks are limited to cheap, allocation-free checks that cannot be done
  at creation. Hot paths (bind, push, draw) do not allocate, hash, or lock.
- Every error carries detail (`thiserror` variants with fields, or `with_detail`). No bare
  "invalid operation" kinds. Never discard errors with `let _ =`; log them if they truly
  cannot propagate (e.g. in `Drop`).
- Format capability tables and limits are queried once at device creation and cached.
- `bitflags` (already in the lockfile) for flag types; no hand-written flag macros.
- `#[allow]` only when unavoidable, with a comment saying why (prefer `#[expect]`).

## Lifetime: owners drop, the RHI defers the native free until the GPU is done

- Resources are owned by their users and destroyed when dropped. Because the GPU may still
  be reading, the native free is **deferred to GPU completion**. That is the only deferral.
- Every queue has one timeline (Vulkan timeline semaphore, Metal `MTLSharedEvent`).
  There are **no per-submit fences**, no `Work`/`Payload` objects, no "cycles", no
  `finish_cycle`/`collect_garbage`/`flush`, no magic frame counts.
- Recording is scoped to a `Frame`. A resource dropped while frame F is open (or after it)
  is tagged with F; once every submission made up to and including frame F has completed
  on every queue, its native object is destroyed. This is provable without per-command
  resource tracking because encoders cannot outlive their frame.
- Freed entries are collected when a frame begins (it already waits for its slot), and on
  `Rhi::wait_idle` / `Drop for Rhi`.
- Beginning frame F waits for each queue's last submission as recorded when frame
  `F - FRAMES_IN_FLIGHT + 1` began, so the wait stays correct even if a `Frame` is leaked.
- `Completion` is `#[derive(Clone, Copy)] struct Completion { queue: QueueType, value: u64 }`
  with `rhi.wait(c, timeout)` and `rhi.is_complete(c)`.

## Frames

```rust
let frame = rhi.begin_frame()?;              // waits for this slot's previous work, frees garbage
scene_data.write(&frame, 0, bytes)?;         // PerFrame<Buffer>: safe per-slot host writes
let a = frame.acquire(&mut swapchain_a)?;      // AcquiredImage: render to a.image()
let b = frame.acquire(&mut swapchain_b)?;
let mut cmd = frame.encoder::<Graphics>("scene")?;
cmd.wait_for(upload)?;                       // the GPU waits for another queue's Completion
// ... record: render passes move a.image() from ImageAccess::Present and back ...
frame.submit(cmd)?;                          // waits for every acquisition since the last one
frame.present([a, b])?;                      // one submission and one present for both
frame.finish();                               // closes the frame (also on Drop)
```

- `FRAMES_IN_FLIGHT` is a single constant in `dirk_rhi`.
- `PerFrame<T>` replaces hand-rolled `[T; MAX_FRAMES_IN_FLIGHT]` arrays and their
  `// SAFETY: waited for this slot` comments. Holding `&Frame` proves the slot is idle;
  `get`/`get_mut` take the frame, and `PerFrame<Buffer>::write` is the safe host write.
- `begin_frame` borrows the `Rhi` mutably, so only one frame is open; `frame.rhi()` creates
  objects meanwhile. Encoders borrow the frame and record on its thread.
- Headless users (tests, tools) use frames too. A frame is a recording scope, not a window.

## Queues and recording

- Typed queues stay: `Graphics` and `CopyQueue` (sealed `QueueKind`; not `Copy`, which
  glob imports would let shadow the `Copy` trait), with the `compile_fail` doctests proving
  a copy encoder cannot blit or begin a render pass and a render pass borrows its encoder.
  Copy-queue ownership transfer is handled by the RHI/uploads, not callers; barriers carry
  an optional `QueueTransfer` recorded on both queues.
- No device-wide lock. Creation is lock-free (Vulkan creation needs no external sync)
  apart from `gpu-allocator`'s allocator, locked only to allocate and free memory, the
  list of dropped objects, locked to add and collect them, the bindless heap's slots,
  locked to assign, write and empty one (which also synchronizes descriptor writes), and
  the list of new images' first transitions, locked to add one and to take them at a
  graphics submission; only queue submission/present take that queue's lock, and waits
  hold none. On Metal, the heap's lock is also taken when a render pass begins, to
  declare its resources, and blit pipelines are created under a lock on first use.
- Safe by default. `unsafe` remains only on: raw `barrier()` (bypassing the graph), raw
  host writes outside `PerFrame`, and creating shaders from unvalidated bytes.
- **Images have a resting access.** `ImageInfo::rest`, chosen by `ImageDesc::rest` or
  defaulting from the usage (`ImageAccess::resting`: sampled by every stage, else storage
  read/write, else the attachment, else copy source, else copy destination), is the access
  every image is in between submissions, so nothing safe takes a caller's word for an
  image's state:
  - New images start undefined; `create_image` queues their transition to rest, which the
    next graphics submission records first. A copy-queue barrier from `Undefined` takes it
    over (uploads), since the copy queue cannot reach graphics accesses.
  - Safe commands (`copy_buffer_to_image`, `copy_image_to_buffer`, `copy_image`,
    `blit_image`) move the subresources they touch from rest into their access and back,
    with no barrier where the access is the resting one and only read.
  - Raw `unsafe barrier()` is the only way to leave rest. The encoder remembers where its
    barriers left each range (per mip and layer, the last barrier covering it), and safe
    commands find ranges there instead of wrapping them, or reject ranges left in another
    access; the copy queue only copies ranges a barrier moved into the copy accesses. The
    barrier's contract: describe the accesses since the previous barrier, and return every
    image to rest before the submission ends.
  - `ImageAccess::Undefined` before a graphics barrier discards the contents but is ordered
    after the image's resting access, so it is always a legal start.
  - Buffers have no layouts and no resting access: ordering host and GPU accesses to them
    stays the caller's job, through barriers, the render graph or the frame slot wait.
- Dynamic state defaults: viewport and scissor default to the render area.
- Render passes use dynamic rendering. Attachments are whole single-mip, single-layer 2D
  images (`&Image`, using the view it owns) of one extent; color attachments may resolve
  into a single-sample image. Like other safe commands, a pass moves its attachments from
  rest into their attachment access when it begins and back when it ends (or, if leaked,
  before the encoder's next command). `pass.draw(&pipeline, &push, vertices, instances)` and
  `pass.draw_indexed(&pipeline, &push, Indices { buffer, format, offset, range, .. }, instances)`
  check only that the pipeline's targets equal the pass's (one small `==`) and that the
  indices fit their buffer; rebinding an unchanged pipeline or index buffer is skipped.
- Clip space is Vulkan's: y points down and depth runs from 0 to 1. Backends with
  another convention flip y when translating shaders.

## Shaders and pipelines

- Shaders are Rust, compiled by rust-gpu. `dirk_build::shaders::build_shaders` compiles a
  shader crate from a build script into one SPIR-V module per entry point and generates a
  type per entry point implementing the `unsafe` trait `dirk_rhi::Shader` (code, stage,
  reflected push data size). The code is the backend's: SPIR-V, or for Apple targets the
  Metal Shading Language that SPIRV-Cross translates it to at build time (see
  [Metal](#metal)); only Apple hosts compile SPIRV-Cross. Implementing it is the only `unsafe` step; creating modules
  from it is safe. `dirk_rhi`'s tests build their shaders the same way
  (`crates/dirk_rhi/test-shaders`).
- `rhi.create_shader::<S, P>() -> ShaderModule<P>` asserts in a `const` block that `S`
  reads no push data or exactly `size_of::<P>()` bytes, and that `P` is at most 128 bytes
  and a multiple of four. Mismatches fail `cargo build` (not `cargo check`, as with every
  post-monomorphization error).
- `GraphicsPipeline<P>` is created from `ShaderModule<P>`s, so its push data type is the
  shaders'. Descriptions are validated fully at creation: stages, target formats and
  sample counts against format capabilities, blending against `BLEND` support and
  independent blend, the color target limit, depth bias clamp. Pipelines have topology,
  culling, front face, depth test and bias, per-target blending and write masks; stencil
  tests come when something needs them.
- Presentation is part of the frame: `frame.acquire(&mut swapchain) -> AcquiredImage`
  registers the acquisition with the frame, whose **next graphics submission waits for
  every acquisition since the last one** (at attachment output and transfer stages), so
  any encoder may render to `acquired.image()` with ordinary commands.
  `frame.present([a, b])` (any `IntoIterator<Item = AcquiredImage>`) makes one empty
  graphics submission signaling each image's present semaphore, whose first scope covers
  every earlier submission, then one `vkQueuePresentKHR` for all swapchains. Acquired
  images borrow the frame and their swapchain, and presenting consumes them, so outliving
  the frame, presenting twice, or resizing while an image is out do not compile
  (`compile_fail` doctests). An acquired image never presented marks its swapchain for
  recreation on the next acquisition, and a frame whose acquisitions no submission waited
  for waits for them in an empty submission when it closes, so acquire semaphores are
  always consumed within their frame. One image per swapchain and frame.

## Presentation

- `RhiCreateInfo::display` enables presentation (surface instance extensions, and
  `VK_KHR_swapchain` becomes a device requirement). `rhi.create_surface(Arc<T: SurfaceTarget>)`
  takes the platform's shared window owner (`dirk_platform::WindowSurfaceTarget`); the
  surface keeps it alive until its deferred free, so whoever drops the last clone must do
  so on the main thread. `rhi.create_swapchain(surface, &SwapchainDesc { extent, formats,
  present_mode, usage })` chooses the format (sRGB color space) and present mode (FIFO
  fallback) up front; images are created on the first acquisition.
- Swapchain images are `Image`s (single-mip 2D with a view, outside the heap) that the
  swapchain owns; usage is limited to color attachment and copies. They rest in
  `ImageAccess::Present`: each acquisition queues the image's transition from undefined
  contents to it, recorded first by the frame's next graphics submission after its
  acquire wait, and render passes and copies move it from there and back like any image.
- Lifecycle (Vulkan): acquire semaphores per frame slot, reusable because beginning a
  frame waits for its slot's previous work; present semaphores per image, reused when the
  image is reacquired. The acquire budget is one image per swapchain at a time, enforced by
  the borrow. Out-of-date or suboptimal acquisitions and presents mark the swapchain for
  recreation; a recreation that still finds it out of date, or a 0x0 surface, returns
  `Error::SwapchainOutOfDate`: skip presenting that frame. `swapchain.resize(w, h)` only
  marks it. Recreation passes the old swapchain as `oldSwapchain` without waiting; the old
  one is destroyed through deferred destruction after waiting for the graphics queue to
  idle, since nothing else proves presentation finished (`VK_EXT_swapchain_maintenance1`
  present fences would; not needed yet). That is the only stall, once per recreation.
- Headless tests cover what needs no display; `cargo run -p dirk_rhi --example present`
  checks acquisition, presentation of two windows at once and recreation on real windows,
  with each window rendered by its own ordinary submission.

## Binding model: bindless, verified at compile time

- One global resource heap: one array per kind of resource and, for images, per view
  type, so every slot of an array holds the descriptor type and dimension shaders declare
  for it. Bindings and power-of-two capacities are constants in
  `dirk_shaders::bindless::heap`: textures 2D, 2D array, cube, cube array and 3D (16384
  each), storage images 2D, 2D array and 3D (4096 each, declared without a format; cubes
  have no storage index), samplers (1024) and storage buffers (65536).
  - Vulkan: one descriptor set (set 0) with `UPDATE_AFTER_BIND | PARTIALLY_BOUND |
    UPDATE_UNUSED_WHILE_PENDING` arrays (Vulkan 1.2 descriptor indexing). Every slot is
    written at creation, and rewritten when freed, to a null descriptor
    (`VK_EXT_robustness2`'s `nullDescriptor`, now required) or, for samplers, a default
    sampler. Devices whose update-after-bind limits cannot hold the heap are rejected.
    Sampled descriptors use the `READ_ONLY_OPTIMAL` layout, which `ImageAccess::Sampled`
    now maps to; storage descriptors use `GENERAL`.
  - Metal: one tier-2 argument buffer with the same arrays, laid out back to back in
    binding order, eight bytes per element (see [Metal](#metal)). Empty slots hold a
    zeroed 1x1 texture of the array's kind, the default sampler, or a buffer of zero
    bytes, which shaders' bounds checks never read.
- Every `ImageView`/`Sampler`/`Buffer` created with a shader-visible usage gets a slot at
  creation and returns it through the deferred free path, so a slot is reused only once
  the GPU is done with its frame. `view.index() -> Option<TextureIndex>` (sampled usage),
  `view.storage_index() -> Option<StorageImageIndex>`, `sampler.index() -> SamplerIndex`,
  `buffer.index::<T>() -> Option<BufferIndex<T>>` (storage usage; `T` is chosen by the
  reader and memory-safe whatever it is). Storage buffers are limited to the device's
  storage buffer range at creation.
- One global pipeline layout: the heap + a push-constant block (<= 128 bytes).
- **Vertex pulling.** Pipelines have no vertex input layouts; shaders read vertices from
  storage buffers by index. Index buffers are still bound directly.
- No bind groups, bind group layouts, per-set pipeline layouts, descriptor pools per group,
  dynamic offsets, or set-tuple macros.
- **Compile-time verification.** `dirk_shaders` defines typed indices (`TextureIndex`,
  `StorageImageIndex`, `SamplerIndex`, `BufferIndex<T>`, all `NoUninit`) and each
  pipeline's push-data struct (`#[repr(C)]`, `NoUninit`). Shaders and host share these
  exact Rust types. With its `gpu` feature, shader crates read the heap through them
  (`push.texture.get(textures)`, `push.vertices.get(buffers)[i]`), declaring the arrays
  as entry point parameters at the documented bindings; the shader build declares the
  descriptor indexing capabilities, and accesses are `NonUniform`.
  `GraphicsPipeline<P: PushData>`; `pass.draw(&pipeline, &P, ...)` only accepts `P`.
  The shader build reflects each entry point's push block size and emits constants so a
  `const` assertion fails the build if `size_of::<P>()` disagrees.
- Stale indices are memory-safe: shaders mask every index to its array's capacity, and
  each slot holds a null descriptor or a live resource of the array's kind and
  dimension. In debug builds a generation check may report them later.

## Metal

Metal 3 GPUs (Apple silicon, recent Intel and AMD Macs) with tier-2 argument buffers, on
macOS 13 or later; other devices are rejected with the reason. Metal has no per-format queries, so
format capabilities follow Apple's feature tables, conservatively.

- **Queues.** Each RHI queue is an `MTLCommandQueue` signaling its own `MTLSharedEvent`
  timeline, so copies run beside graphics work (`dedicated_copy_queue` is true). Metal
  encodes event waits before the passes that follow them, and an encoder's waits are known
  only at submission, so they go into a command buffer committed just before it: a queue
  blocks at a wait until the event reaches its value. Command buffers that fail mark the
  device lost: waits check for that every 100 ms, since a failed command buffer may never
  reach its value.
- **Synchronization is Metal's hazard tracking.** Every resource is tracked, and each
  render pass declares the heap's resources with `useResources` (read for sampled
  textures, read-write for storage images and buffers), which also makes them resident.
  Metal then orders conflicting accesses within a queue on its own, so barriers record
  nothing and queue ownership transfers have no equivalent; they are still validated, and
  the encoder still remembers where they leave images. Metal textures have no layouts, so
  resting accesses move nothing: new images and acquisitions queue no first transition,
  and the safe commands' moves to and from rest record nothing. Across queues, `wait_for`
  orders work, as on Vulkan. Declaring every heap
  resource costs O(resources) per pass and serializes passes using storage resources; a
  residency set and explicit fences can replace it when that matters.
- **Shaders.** `dirk_build::shaders` translates each entry point with SPIRV-Cross (MSL
  3.0, argument buffers tier 2) and rewrites the heap's argument buffer struct so every
  shader declares every array in binding order: slot `s` of binding `b` is element
  `heap::first_element(b) + s`. An array read as several types, such as storage buffers
  of different elements, is declared once and cast for the others. Storage buffer
  lengths, which rust-gpu's bounds checks read, follow the arrays as a pointer to one
  `u32` per element
  (`heap::metal::BUFFER_LENGTHS`). The heap is buffer 0 and push data buffer 1
  (`heap::metal`); push data is set with `setVertexBytes`/`setFragmentBytes`, padded to
  16 bytes since MSL rounds structs up to their alignment. Vertex shaders flip y; front
  faces stay in framebuffer coordinates. Libraries are compiled from source when a
  `ShaderModule` is created.
- **Resources.** Device memory is private, upload and readback memory shared (coherent);
  resources are allocated by the device, without `gpu-allocator`, since its heaps are
  untracked. Images are their own full view. Buffer↔image copies take any row pitch that
  is a multiple of the texel size: tightly packed rows need no padding.
- **Blits** draw: Metal's blit encoder cannot scale, so each destination layer or depth
  slice is a small render pass sampling a view of the source mip and layer.
- **Presentation.** Surfaces are `CAMetalLayer`s of the window's `AppKit`/`UIKit` view,
  created on the main thread. Drawables are sized to the layer in pixels; each
  acquisition wraps the drawable's texture as the swapchain's image, and is suboptimal
  once the layer's size changed. Acquiring waits without timeout
  (`allowsNextDrawableTimeout` is off), as the API promises; occluded windows on macOS
  may then block until visible, so callers should skip frames while occluded.
  Acquisitions and presentation need no semaphores: `frame.present` skips its empty
  submission and commits one command buffer presenting every drawable after the work
  committed before it on the same queue. FIFO and immediate presentation are supported;
  the others fall back to FIFO.
- **Validation** is Metal's API and shader validation (`MTL_DEBUG_LAYER=1`,
  `MTL_SHADER_VALIDATION=1`), which stops the process on misuse;
  `validation_error_count` counts command buffers that failed.

## Render graph (`dirk_render_utils`)

```rust
let mut graph = RenderGraph::new();                        // one graph per frame
let target = graph.import(acquired.image())?;                // from and back to its resting access
let depth = graph.create_image(&ImageDesc { usage: ImageUsage::empty(), .. }); // usage derived
graph.pass("scene")
    .color(target, LoadOp::Clear(black))
    .depth(depth, LoadOp::Clear(1.0))
    .sampled(shadow, ShaderStages::FRAGMENT)
    .record(|pass| pass.draw(&pipeline, &push, 0..3, 0..1))?;
graph.execute(&frame, &mut encoder, &mut transients)?;    // one encoder, one barrier call per pass
```

- Passes declare their accesses; `record` validates them and derives the pass's barriers
  at once, leaving the graph unchanged on error: per-mip RAW/WAR/WAW (repeated writes
  included), reads of contents nothing wrote, a just-acquired swapchain image first used
  other than as an attachment or copy destination, shader accesses without stages,
  conflicting accesses in one pass, imports lacking a usage, and handles of other graphs.
- Pass kinds are typestate: declaring an attachment makes `record` take a
  `RenderPass` callback (boxed), otherwise a `CommandEncoder` one, so drawing outside a
  render pass does not compile. Callbacks reach transients through `pass.image(handle)`,
  which only resolves what the pass declared.
- Reads in one pass that can share a state merge (sampling in more stages). Reads in
  later passes widen the barrier that established the read state rather than adding one;
  only a read state an import starts in, with no earlier barrier, needs a read-to-read
  barrier for new stages.
- Imports are deduplicated with a hash map keyed by address; importing again with other
  accesses is an error. Images need no state from their importer: `graph.import(&image)`
  starts from the image's resting access with its contents, `graph.import_scratch(&image)`
  from undefined contents ordered after the resting access (always legal, with no
  first-frame special case), and both end at rest. Imported buffers still state the
  access they are in before the graph and leave it in (`import_buffer(buffer, before,
  after)`): buffers have no layouts, so a wrong `before` costs synchronization, not a
  layout. Callbacks recording safe commands on a declared image find it where the graph's
  barriers left it, which the encoder remembers: in place when the access matches,
  rejected otherwise.
- Host-written per-frame buffers (`PerFrame<Buffer>`) are not imported: the frame slot
  wait and submission cover them.
- Transients take their usage from their declarations (none: never created), so
  transfer-only images get no view. Attachment stores are inferred: a transient nothing
  reads later is discarded, as is a scratch import (`import_scratch`), whose
  contents are not needed after the graph, such as a depth buffer cleared every frame.
- `Transients` pools them across frames: a transient used in frame F is reused, by one of
  the same description and usage, from frame F + `FRAMES_IN_FLIGHT` on, so it starts
  undefined with no barrier against earlier work, and is left wherever its last pass left
  it, since nothing outside the graph reaches it; one that frame does not reuse is
  dropped through deferred destruction. Steady rendering keeps `FRAMES_IN_FLIGHT` sets.
- Subresource ranges (per-layer/mip passes) are a later, separate PR: the state tracker
  already works per mip span.

## Uploads (`dirk_render_utils`)

- A persistently mapped staging ring per frame slot, grown on demand; data is packed
  straight into it (no intermediate `Vec`).
- Copies are padded to the 4-byte rule internally; callers can upload any byte length.
- Uses the copy queue with ownership transfer when a dedicated one exists.
- `Uploader` owns the staging buffers; `uploader.begin(&mut frame)` opens a batch.
  `uploads.buffer(.., access, data)` and `uploads.image(&desc, mips)` return the new
  resource at once, buffers in the graphics access given and images in their resting
  access, and `uploads.submit()` precedes the
  graphics work using them, which then needs no wait: the batch borrows the frame mutably,
  so no other encoder exists until it is submitted (`uploads.rhi()` creates objects
  meanwhile).
- Uploads own their images' first transition: on the copy queue, a barrier from
  `Undefined` takes it over from the graphics queue's first submission, and the
  ownership transfer back to graphics ends in the resting access.
  `uploads.write_image(&image, region, texels)` writes a region of any image through the
  graphics queue's safe copy, which moves the region from rest and back.

## Renderer

- Owns one `Rhi`. Each frame uploads new models and UI textures, then acquires an image of
  every visible window (not occluded, not empty, presentable). When none is visible, it
  submits nothing else.
- The viewports some acquired window shows and every acquired window render in one graph
  and one submission, then one `frame.present` presents every window; the passes
  drawing windows and the UI sample the viewports' outputs.
- One output and depth/MSAA attachment set per viewport (not per frame slot), imported
  into each frame's graph with no state kept between frames: the output with
  `import`, the attachments with `import_scratch`, whose contents are discarded once the
  output is resolved. They rest between frames, so frames in flight take turns with a
  barrier between them, and the first frame needs no special case. A viewport without a
  world and camera clears its output.
- Per-object data lives in one per-slot storage buffer addressed by instance index. Meshes
  and egui pull their vertices through `BufferIndex`es; textures and samplers are
  `TextureIndex`es and `SamplerIndex`es in push data shared through `dirk_shaders::types`.
- Colors are linear everywhere. The renderer's color images (viewport outputs, the UI,
  textures) are `Rgba8Srgb`, encoded and decoded in hardware. egui's sRGB-encoded,
  premultiplied colors are unpremultiplied, decoded and premultiplied again (alpha zero
  marks an additive color), on the CPU for textures and in the vertex shader for vertex
  colors, so the UI composites in linear space. Presentation draws the composited image
  into the window, encoding once: in hardware on sRGB surfaces, in the shader otherwise.
- Model textures get their mips on the CPU, averaged in linear space, until GPU mip
  generation lands.

## Later, each as its own small PR

Compute pipelines and dispatch, indirect draws, timestamp queries, comparison samplers,
render-graph subresource ranges, async compute. Each adds API without breaking the above.
