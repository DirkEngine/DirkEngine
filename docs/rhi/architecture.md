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
- `Completion` is `#[derive(Clone, Copy)] struct Completion { queue: QueueType, value: u64 }`
  with `rhi.wait(c, timeout)` and `rhi.is_complete(c)`.

## Frames

```rust
let mut frame = rhi.begin_frame()?;          // waits for this slot's previous work, frees garbage
scene_data.get_mut(&frame).write(&scene);    // PerFrame<T>: safe per-slot host writes
let mut cmd = frame.encoder::<Graphics>("scene")?;
// ... record ...
frame.submit(cmd)?;                           // or submit several, across queues
frame.present(acquired)?;                     // presentation is part of the frame
frame.finish()?;                              // closes the frame (also on Drop)
```

- `FRAMES_IN_FLIGHT` is a single constant in `dirk_rhi`.
- `PerFrame<T>` replaces hand-rolled `[T; MAX_FRAMES_IN_FLIGHT]` arrays and their
  `// SAFETY: waited for this slot` comments. Holding `&Frame` proves the slot is idle.
- Headless users (tests, tools) use frames too. A frame is a recording scope, not a window.

## Queues and recording

- Typed queues stay: `Graphics` and `Copy` (sealed `QueueKind`), with the `compile_fail`
  doctests proving a copy encoder cannot begin a render pass and a render pass borrows its
  encoder. Copy-queue ownership transfer is handled by the RHI/uploads, not callers.
- Vulkan copies share the graphics family for now: the transfer-only family path (queue
  ownership transfers, concurrently shared buffers) is kept but not selected
  (`QueueFamilies::DEDICATED_COPY`) until it runs on hardware that has such a family, since
  the devices it was tested on have none.
- No device-wide lock. Creation is lock-free (Vulkan creation needs no external sync);
  only queue submission/present take that queue's lock.
- Safe by default. `unsafe` remains only on: raw `barrier()` (bypassing the graph), raw
  host writes outside `PerFrame`, and creating shaders from unvalidated bytes.
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
