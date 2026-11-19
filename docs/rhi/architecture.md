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
let image = frame.acquire(&mut swapchain)?;   // AcquiredImage: render to image.image()
let mut cmd = frame.encoder::<Graphics>("scene")?;
cmd.wait_for(upload)?;                       // the GPU waits for another queue's Completion
// ... record, leaving the image in ImageAccess::Present ...
let image = frame.submit_with(cmd, image)?;   // PresentableImage; frame.submit for other work
frame.present(image)?;                        // presentation is part of the frame
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
  list of dropped objects, locked to add and collect them, and the bindless heap's slots,
  locked to assign, write and empty one (which also synchronizes descriptor writes); only
  queue submission/present take that queue's lock, and waits hold none. On Metal, the
  heap's lock is also taken when a render pass begins, to declare its resources, and
  blit pipelines are created under a lock on first use.
- Safe by default. `unsafe` remains only on: raw `barrier()` (bypassing the graph), raw
  host writes outside `PerFrame`, and creating shaders from unvalidated bytes.
- Dynamic state defaults: viewport and scissor default to the render area.
- Render passes use dynamic rendering. Attachments are whole single-mip, single-layer 2D
  images (`&Image`, using the view it owns) of one extent; color attachments may resolve
  into a single-sample image. `pass.draw(&pipeline, &push, vertices, instances)` and
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
- Typestate presentation: `frame.acquire(&mut swapchain) -> AcquiredImage`,
  `frame.submit_with(encoder, acquired) -> PresentableImage`, `frame.present(presentable)`.
  Each step consumes the previous token, and tokens borrow the frame and the swapchain, so
  presenting before submit, submitting twice, outliving the frame, or resizing while an
  image is out do not compile (`compile_fail` doctests). No atomics: the swapchain keeps a
  plain `pending` field so an image abandoned at any step is released (its acquire
  semaphore consumed by an empty submission) and the swapchain recreated on the next
  acquisition. One image per swapchain and frame.

## Presentation

- `RhiCreateInfo::display` enables presentation (surface instance extensions, and
  `VK_KHR_swapchain` becomes a device requirement). `rhi.create_surface(Arc<T: SurfaceTarget>)`
  takes the platform's shared window owner (`dirk_platform::WindowSurfaceTarget`); the
  surface keeps it alive until its deferred free, so whoever drops the last clone must do
  so on the main thread. `rhi.create_swapchain(surface, &SwapchainDesc { extent, formats,
  present_mode, usage })` chooses the format (sRGB color space) and present mode (FIFO
  fallback) up front; images are created on the first acquisition.
- Swapchain images are `Image`s (single-mip 2D with a view, outside the heap) that the
  swapchain owns; usage is limited to color attachment and copies. The image access
  `ImageAccess::Present` describes them before their first barrier after acquisition
  (contents discarded) and before presenting.
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
  checks acquisition, presentation and recreation on a real window.

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
    and `ImageAccess::DepthStencilRead` now map to; storage descriptors use `GENERAL`.
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
  nothing and queue ownership transfers have no equivalent; they are still validated.
  Across queues, `wait_for` orders work, as on Vulkan. Declaring every heap
  resource costs O(resources) per pass and serializes passes using storage resources; a
  residency set and explicit fences can replace it when that matters.
- **Shaders.** `dirk_build::shaders` translates each entry point with SPIRV-Cross (MSL
  3.0, argument buffers tier 2) and rewrites the heap's argument buffer struct so every
  shader declares every array in binding order: slot `s` of binding `b` is element
  `heap::first_element(b) + s`. Storage buffer lengths, which rust-gpu's bounds checks
  read, follow the arrays as a pointer to one `u32` per element
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
  Presenting commits `presentDrawable` after the rendering on the same queue. FIFO and
  immediate presentation are supported; the others fall back to FIFO.
- **Validation** is Metal's API and shader validation (`MTL_DEBUG_LAYER=1`,
  `MTL_SHADER_VALIDATION=1`), which stops the process on misuse;
  `validation_error_count` counts command buffers that failed.

## Render graph (`dirk_render_utils`)

- Keep: per-mip barrier derivation (RAW/WAR/WAW), merged overlapping reads, boundary
  states for imported resources, and its focused tests.
- Imports are deduplicated with a hash map, not a linear scan.
- Host-written per-frame buffers are not imported (the frame slot wait covers them).
- Views are created only when a pass needs one (never for transfer-only textures).
- Transient textures/buffers come from a pool reused across frames.
- One barrier call per pass (buffers and images merged); read->read transitions widen the
  read state instead of emitting a barrier.
- One graph per frame.
- Subresource ranges (per-layer/mip passes) are a later, separate PR.

## Uploads (`dirk_render_utils`)

- A persistently mapped staging ring per frame slot, grown on demand; data is packed
  straight into it (no intermediate `Vec`).
- Copies are padded to the 4-byte rule internally; callers can upload any byte length.
- Uses the copy queue with ownership transfer when a dedicated one exists.

## Renderer

- Owns one `Rhi`. One graph per frame. One depth/MSAA attachment set per viewport (not per
  frame slot). Per-object data lives in one per-slot storage buffer addressed by instance
  index. Skip submission entirely when nothing is visible.
- Compose UI in linear space and encode to the surface once.

## Later, each as its own small PR

Compute pipelines and dispatch, indirect draws, timestamp queries, comparison samplers,
render-graph subresource ranges, async compute. Each adds API without breaking the above.
