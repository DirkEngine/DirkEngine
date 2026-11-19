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
  resource tracking because encoders cannot outlive their frame. Bindless indices, which
  are plain numbers, are the exception: later frames may still read an object's heap
  slots through them, which keeps it longer (see
  [Slots live until their last use](#binding-model-bindless-verified-at-compile-time)).
- Freed entries are collected when a frame begins (it already waits for its slot), and on
  `Rhi::wait_idle` / `Drop for Rhi`.
- Beginning frame F waits for each queue's last submission as recorded when frame
  `F - FRAMES_IN_FLIGHT + 1` began, so the wait stays correct even if a `Frame` is leaked.
- `Completion` is `#[derive(Clone, Copy)] struct Completion { queue: QueueType, value: u64 }`
  (plus the id of its RHI, so another RHI's completions are rejected) with
  `rhi.wait(c, timeout)` and `rhi.is_complete(c)`.

## Frames

```rust
let frame = rhi.begin_frame()?;              // waits for this slot's previous work, frees garbage
let objects = frame.alloc(&object_data)?;    // fresh host-visible memory only this frame reads
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
- **Frame memory.** Each frame slot owns one host-visible buffer of
  `RhiCreateInfo::frame_memory` bytes (8 MiB by default) with copy source, index and
  storage usage. `frame.alloc(&[T]) -> FrameSlice<T>` copies data into the next free range
  of the open frame's and returns where it is (buffer, byte offset, length); allocations
  start at multiples of 16 bytes and of `size_of::<T>()`, and reserve whole 16 bytes.
  Nothing submitted reads fresh memory, and a slot's memory is reused only once beginning
  a frame waited for its previous work, so these host writes are sound by construction:
  no "write before submitting" rule. An allocation that does not fit gets a host-visible
  buffer of its own, freed through deferred destruction when the frame closes; nothing
  grows. Per-frame data (per-object records, UI vertices and indices, upload staging)
  comes from here. Slices borrow the frame.
- Host-visible buffers are shared by every queue family (Vulkan `CONCURRENT` sharing when
  the copy queue has a family of its own), so both queues read frame memory without
  ownership transfers, and transfers of such buffers move no ownership.
- `PerFrame<T>` replaces hand-rolled `[T; FRAMES_IN_FLIGHT]` arrays, such as per-slot
  query pools, and their `// SAFETY: waited for this slot` comments. Holding `&Frame`
  proves the slot is idle; `get`/`get_mut` take the frame.
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
  locked to assign, write and empty one (which also synchronizes descriptor writes) and
  while submitting, so no slot is written meanwhile, and
  the list of new images' first transitions, locked to add one and to take them at a
  graphics submission; only queue submission/present take that queue's lock, and waits
  hold none. On Metal, allocating takes the heap list's lock, blit pipelines are
  created under a lock on first use, and each encoder takes a fence from its command
  pool under the pool's own lock, which only the frame's thread takes (see
  [Metal](#metal)).
- Safe by default. `unsafe` remains only on: raw `barrier()` (bypassing the graph), raw
  host writes (`Buffer::write`), and creating shaders from unvalidated bytes.
- **Images have a resting access.** `ImageInfo::rest`, chosen by `ImageDesc::rest` or
  defaulting from the usage (`ImageAccess::resting`: sampled by every stage, else storage
  read/write, else the attachment, else copy source, else copy destination), is the access
  every image is in between submissions, so nothing safe takes a caller's word for an
  image's state:
  - New images start undefined; `create_image` queues their transition to rest, which the
    next graphics submission records first. A copy-queue barrier from `Undefined` takes it
    over (uploads), since the copy queue cannot reach graphics accesses, once its encoder is
    submitted: an encoder dropped unsubmitted leaves the image to the graphics queue.
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
  - Buffers have no layouts and no resting access: ordering GPU accesses to them stays
    the caller's job, through barriers or the render graph. Host data the GPU reads each
    frame comes from frame memory, which needs no ordering.
- Dynamic state defaults: viewport and scissor default to the render area. `set_viewport`
  rejects viewports past the device's viewport limits, cached in `Limits`.
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
  reflected push data size, whether it writes storage buffers or images, push data type). The code is the backend's: SPIR-V, or for Apple targets the
  Metal Shading Language that SPIRV-Cross translates it to at build time (see
  [Metal](#metal)); only Apple hosts compile SPIRV-Cross. Implementing it is the only `unsafe` step; creating modules
  from it is safe. `dirk_rhi`'s tests build their shaders the same way
  (`crates/dirk_rhi/test-shaders`).
- `rhi.create_shader::<S>() -> ShaderModule<S::Push>`, where `S::Push` is the push data
  type the entry point's source reads, asserts in a `const` block that `S` reads no push
  data or exactly `size_of::<S::Push>()` bytes, and that `S::Push` is at most 128 bytes
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
  `dirk_shaders::bindless::heap`: textures 2D, 2D array, cube, cube array and 3D, and
  storage images 2D, 2D array and 3D (declared without a format; cubes have no storage
  index), 16384 each, one capacity for every image array; samplers (1024) and storage
  buffers (65536).
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
  the GPU is done with every frame that read it (below). `view.index() -> Option<TextureIndex>` (sampled usage),
  `view.storage_index() -> Option<StorageImageIndex>`, `sampler.index() -> SamplerIndex`,
  `buffer.index::<T>() -> Option<BufferIndex<T>>` (storage usage; `T` is chosen by the
  reader and memory-safe whatever it is). Storage buffers are limited to the device's
  storage buffer range at creation, frame memory included. A frame allocation's
  `slice.index() -> BufferSlice<T>` is its buffer's `BufferIndex<T>`, its first element and
  its length.
- **Slots live until their last use.** Indices are `Copy` numbers, so they may be used
  after their owner dropped. The RHI records, per slot, the last frame whose work may read
  it, whenever it sees an index: in the push data of every draw, and in the
  elements of frame allocations. Both list their indices through the `unsafe` trait
  `dirk_shaders::bindless::HeapIndices`, which `PushData` and `frame.alloc` require;
  `#[derive(HeapIndices)]` lists the indices of every field (each field's type implements
  it too: indices, scalars, arrays), so it is never written by hand. Every array an index
  kind reads shares that kind's record, one `u64` per slot, since a `TextureIndex` may read
  any texture array.
  - A dropped object is freed, its slots emptied before its native object is destroyed,
    once a frame after its last use (its drop, or a recorded read) closed and every frame
    that read its slots completed. So an index kept past its owner's drop stays valid
    through the next frame, and for as long as every frame uses it; `wait_idle` keeps
    what the next frame may still read. Taking an index from its owner (`view.index()`)
    needs no record: it happens before the drop, whose frame tags the object anyway.
  - Slots are reused first in, first out, and only once no recorded read is pending: a
    freed slot read again through a stale index is skipped until that frame completes.
  - Recording a read is a relaxed load and, the first time a frame reads the slot, a
    relaxed store, on the open frame's thread: no lock, allocation or hashing per draw.
    Reads are checked when frames begin, with no frame open, and when a slot is taken,
    under the heap's lock, which submissions hold too: either the submission came first
    and the check sees its reads, or the slot was written before the submission, which
    reads the new descriptor (Vulkan's `UPDATE_AFTER_BIND`; Metal reads the argument
    buffer when the work runs). So no slot is written while pending work may read it
    through an index the RHI saw, as `UPDATE_UNUSED_WHILE_PENDING` requires.
  - The RHI never sees indices in other memory: storage buffers written once (such as a
    material table) or by the GPU, frame memory of earlier frames, or indices shaders
    compute. Work reading those relies on its caller to keep their owners alive until it
    completes; otherwise they are stale, reading the empty descriptor or another resource
    of their kind, and their slots may be written while that work runs.
- One global pipeline layout: the heap + a push-constant block (<= 128 bytes).
- **Vertex pulling.** Pipelines have no vertex input layouts; shaders read vertices from
  storage buffers by index. Index buffers are still bound directly.
- No bind groups, bind group layouts, per-set pipeline layouts, descriptor pools per group,
  dynamic offsets, or set-tuple macros.
- **Compile-time verification.** `dirk_shaders` defines typed indices (`TextureIndex`,
  `StorageImageIndex`, `SamplerIndex`, `BufferIndex<T>`, `BufferSlice<T>`, all `NoUninit`) and each
  pipeline's push-data struct (`#[repr(C)]`, `NoUninit`, `HeapIndices`). Shaders and host share these
  exact Rust types. With its `gpu` feature, shader crates read the heap through them
  (`push.texture.get(textures)`, `push.vertices.get(buffers)[i]`, `push.objects.get(buffers,
  i)` for a slice, which checks `i` against its length), declaring the arrays
  as entry point parameters at the documented bindings; the shader build declares the
  descriptor indexing capabilities, and accesses are `NonUniform`.
  `GraphicsPipeline<P: PushData>`; `pass.draw(&pipeline, &P, ...)` only accepts `P`.
  The shader build names each entry point's push data type, the one its
  `#[spirv(push_constant)]` parameter refers to (`()` without one), as `Shader::Push`;
  `rhi.create_shader::<S>()` returns a `ShaderModule<S::Push>`, so pipelines pair only
  shaders of one push type (a shader reading none joins others with `with_push`). The
  build also reflects each push block's size, and a `const` assertion fails the build if
  it disagrees with the type's.
- Indices read only the arrays of their kind: `get` accepts only those arrays' element
  types (sealed traits over the `spirv_std` image, sampler and buffer types), the shader
  build checks every array a shader declares against the heap's bindings, and `get` masks
  the index to the capacity of every array of its kind. Each slot holds a descriptor of
  the array's kind and dimension: a live resource, or, once freed, a null one. So an
  index of any value reads inside its array: a stale or forged one reads the null
  descriptor or another live resource of its kind. Freed slots are reused first in, first
  out, so stale indices read the null descriptor for as long as possible. In debug builds
  a generation check may report them later.

## Metal

Metal 3 GPUs (Apple silicon, recent Intel and AMD Macs) with tier-2 argument buffers, on
macOS 13 or later; other devices are rejected with the reason. Metal has no per-format queries, so
format capabilities follow Apple's feature tables, conservatively.

- **Queues.** Each RHI queue is an `MTLCommandQueue` signaling its own `MTLSharedEvent`
  timeline, so copies run beside graphics work (`dedicated_copy_queue` is true). Metal
  encodes event waits before the passes that follow them, and an encoder's waits are known
  only at submission, so they go into a command buffer committed just before it, which
  ends with the queue's join (below). Command buffers that fail mark the device lost:
  waits check for that every 100 ms, since a failed command buffer may never reach its
  value.
- **Memory and residency.** Every buffer and texture is allocated from an untracked
  `MTLHeap` (automatic placement; 64 MiB heaps per memory kind, a heap of its own for
  larger resources; empty heaps are released when garbage is collected, but one spare per
  memory kind), so Metal tracks no hazards at all. Before each draw, render encoders make
  every heap resident with one `useHeap` each, O(heaps). Metal makes heaps resident for
  reads only and does not cover texture views, which shader validation enforces, so
  encoders also declare, with one `useResources` per usage, the resources that needs:
  texture views in the heap read, before every draw, and storage buffers and storage
  images read-write, once a bound pipeline writes storage (the shader build reflects
  which shaders do: `Shader::WRITES_STORAGE`); read-only passes rely on `useHeap`. Storage
  subviews are in both lists. The bindless heap keeps the lists as slots are filled and
  emptied. Both the heap list and the declared list carry a generation; command buffers keep a snapshot and check both
  generations (two atomic loads) before every draw, so resources created while an encoder
  records, even in a new heap, are resident before the commands using them, with no lock
  while recording. Residency sets (macOS 15) would drop the storage resources from that
  list.
- **Synchronization is fences.** RHI barriers, explicit or implied, become fences:
  - Each encoder (blit or render) waits, when it begins, for the fences of the encoders
    between the last two barriers, and updates a fence of its own when it ends (render
    encoders at and after both stages). A barrier ends the open encoder, and the encoders
    begun since the previous barrier become the ones later encoders wait for. Each of
    those waited for the ones before it, so every encoder after a barrier runs after
    every encoder before it, like a Vulkan barrier at encoder granularity; encoders
    between two barriers overlap freely. Barriers' accesses only matter for validation:
    Metal textures have no layouts and queues need no ownership transfers.
  - Implicit transitions are barriers too: the safe commands' moves from rest and back,
    a render pass's attachments and `end_pass` closing a leaked pass. Reads in the
    resting access need none, since whatever wrote the image moved it back to rest with
    a barrier.
  - Each queue has a join fence. A command buffer's first encoders wait for it, and
    submitting ends the command buffer with a join encoder (a 4-byte fill of a scratch
    buffer, so Metal keeps it) waiting for its last encoders and updating it. Metal
    evaluates fences in submission order, so every command buffer runs after all work
    submitted to its queue before it, and the timeline value, signaled after the join,
    means that work is done. The command buffer of cross-queue waits ends with a join
    too, so `wait_for` orders what follows; frame memory needs nothing, since only the
    host writes it and a slot is reused once its frame completed on every queue.
  - Fences are lent by each queue and frame slot's command pool, one per encoder, and
    reused when the slot comes round again.
  - Barriers order whole encoders; waiting for many encoders costs one wait each, and
    command buffers do not overlap on a queue. Recording safe commands needs no barrier
    beyond those, as on Vulkan; buffers are still ordered by their callers' barriers.
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
  resources come from the heaps above, without `gpu-allocator`. Images are their own full view. Buffer↔image copies take any row pitch that
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
  submission and commits one command buffer presenting every drawable, which first waits
  for the graphics timeline to reach its last submission: Metal presents drawables once
  that command buffer is scheduled, so the wait holds presentation back until the work
  rendering them is done. FIFO and immediate presentation are supported;
  the others fall back to FIFO.
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

- Data is staged in frame memory (`frame.alloc`), packed straight into it (no
  intermediate `Vec`); uploads own no buffers and nothing grows. Data larger than what is
  left of the frame's memory gets a buffer of its own, freed with the frame.
  `uploads.staged()` counts the bytes a batch staged, so callers can spread large uploads
  over frames.
- Copies are padded to the 4-byte rule internally; callers can upload any byte length.
- Uses the copy queue with ownership transfer when a dedicated one exists; both queues read
  frame memory, which every queue family shares.
- `Uploads::new(&mut frame)` opens a batch.
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

- Owns one `Rhi`. One graph per frame. One depth/MSAA attachment set per viewport (not per
  frame slot). Per-object data lives in one per-slot storage buffer addressed by instance
  index. Skip submission entirely when nothing is visible.
- Compose UI in linear space and encode to the surface once.

## Later, each as its own small PR

Compute pipelines and dispatch, indirect draws, timestamp queries, comparison samplers,
render-graph subresource ranges, async compute. Each adds API without breaking the above.
