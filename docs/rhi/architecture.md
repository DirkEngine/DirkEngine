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
  #[cfg_attr(any(target_os = "macos", target_os = "ios"), path = "metal/mod.rs")]
  #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), path = "vulkan/mod.rs")]
  mod native;
  ```
  Until the Metal PR lands, every target (Apple included, via MoltenVK) uses Vulkan.
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
- Lint exceptions follow the conventions (`docs/dev/rust-conventions.md`): never `#[allow]`,
  only `#[expect(lint, reason = "...")]` on the smallest item.

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
  (plus the id of its RHI, so another RHI's completions are rejected) with
  `rhi.wait(c, timeout)` and `rhi.is_complete(c)`.

## Frames

```rust
let frame = rhi.begin_frame()?;              // waits for this slot's previous work, frees garbage
let objects = frame.alloc(&object_data)?;    // fresh host-visible memory only this frame reads
let mut cmd = frame.encoder::<Graphics>("scene")?;
cmd.wait_for(upload)?;                       // the GPU waits for another queue's Completion
// ... record ...
frame.submit(cmd)?;                           // or submit several, across queues
frame.present(acquired)?;                     // presentation is part of the frame
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
- Vulkan copies share the graphics family for now: the transfer-only family path (queue
  ownership transfers, concurrently shared buffers) is kept but not selected
  (`QueueFamilies::DEDICATED_COPY`) until it runs on hardware that has such a family, since
  the devices it was tested on have none.
- No device-wide lock. Creation is lock-free (Vulkan creation needs no external sync)
  apart from `gpu-allocator`'s allocator, locked only to allocate and free memory, the
  list of dropped objects, locked to add and collect them, the list of new images'
  first transitions, locked to add one, to take them at a graphics submission, to forget
  those a copy queue submission takes over and to forget dropped images' when a frame
  begins or they are destroyed, and the list of images dropped since the last frame
  began, locked to add one, to take them and to remove one as it is destroyed; only
  queue submission/present take that queue's lock, and waits hold none.
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
    submitted: an encoder dropped unsubmitted leaves the image to the graphics queue. An
    image dropped before then keeps its transition, which commands recorded meanwhile
    expect, until the next frame begins or the image is destroyed: encoders cannot
    outlive their frame, so no command recorded later expects it. Beginning a frame moves
    the frame dropped objects are tagged with before it forgets, so an image dropped on
    another thread too late to be forgotten is tagged with the new frame, whose
    submissions may still record its transition, and is destroyed only after them.
    Destroying an image forgets what is left of it before the driver can reuse its
    handle, so a new image given the same handle keeps its own transition.
  - Safe commands (`copy_buffer_to_image`, `copy_image_to_buffer`, `copy_image`,
    `blit_image`) move the subresources they touch from rest into their access and back,
    with no barrier where the access is the resting one and only read.
  - Raw `unsafe barrier()` is the only way to leave rest. The encoder remembers where its
    barriers left each range (per mip and layer, the last barrier covering it), and safe
    commands find ranges there instead of wrapping them, or reject ranges left in another
    access; the copy queue only copies ranges a barrier moved into the copy accesses. The
    barrier's contract: describe the accesses since the previous barrier.
  - Submitting returns what barriers left elsewhere to rest, so every image is at rest
    between submissions whatever the recording did (a render graph that failed halfway, a
    barrier never undone): graphics submissions record one barrier from where each range
    was left back to rest, as do copy queue submissions when rest is a copy access; otherwise
    the copy queue, which cannot reach it, rejects the encoder. A range a barrier released to another queue is left
    to the barrier acquiring it there. `encoder.restore()` does the same before recording
    more, for code that needs every image at rest, such as the render graph.
  - `ImageAccess::Undefined` before a graphics barrier discards the contents but is ordered
    after the image's resting access, so it is always a legal start.
  - Buffers have no layouts and no resting access: ordering GPU accesses to them stays
    the caller's job, through barriers or the render graph. Host data the GPU reads each
    frame comes from frame memory, which needs no ordering.
- Dynamic state defaults: viewport and scissor default to the render area.
- Typestate presentation: `AcquiredFrame` -> submitted -> presented. Presenting before
  submit, or submitting twice, does not compile. No atomics tracking frame phase.

## Binding model: bindless, verified at compile time

- One global resource heap.
  - Vulkan: one descriptor set (set 0) with `UPDATE_AFTER_BIND | PARTIALLY_BOUND` arrays
    for sampled images, storage images, samplers, storage buffers (Vulkan 1.2 descriptor
    indexing). Freed slots are rewritten to a valid default descriptor before reuse.
  - Metal: one tier-2 argument buffer with the same arrays.
- Every `ImageView`/`Sampler`/`Buffer` created with a shader-visible usage gets a slot at
  creation and returns it through the deferred free path. `view.index() -> TextureIndex`.
- One global pipeline layout: the heap + a push-constant block (<= 128 bytes).
- **Vertex pulling.** Pipelines have no vertex input layouts; shaders read vertices from
  storage buffers by index. Index buffers are still bound directly.
- No bind groups, bind group layouts, per-set pipeline layouts, descriptor pools per group,
  dynamic offsets, or set-tuple macros.
- **Compile-time verification.** `dirk_shaders` defines typed indices (`TextureIndex`,
  `SamplerIndex`, `BufferIndex<T>`) and each pipeline's push-data struct (`#[repr(C)]`,
  `NoUninit`). Shaders and host share these exact Rust types.
  `GraphicsPipeline<P: PushData>`; `pass.draw(&pipeline, &P, ...)` only accepts `P`.
  The shader build reflects each entry point's push block size and emits constants so a
  `const` assertion fails the build if `size_of::<P>()` disagrees.
- Stale indices are memory-safe (they read a valid default or a live resource); in debug
  builds a generation check may report them.

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
