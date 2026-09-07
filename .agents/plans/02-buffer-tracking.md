# Plan 02 — Buffer tracking in the RHI & frame graph

Make buffers first-class frame graph resources so passes can declare buffer
accesses and get ordering for free, mirroring what textures have today.

## Status

The RHI already has most of the resource surface: `Buffer` trait with
host-visible `write`, `BufferDesc`, `copy_buffer`/`copy_buffer_to_image`,
`BufferBarrier`, and binding-level storage (`BindingType::UniformBuffer`,
`StorageBuffer`). What is missing is *graph integration*: buffers cannot be
declared, tracked, or resolved by the render graph, and upload paths bypass
ordering entirely (egui `prepare()` records copies before `graph.run()`).

## Dependencies

None — this is a foundation plan. Plans 01 (compute) and 03 (transient
allocator) build on it.

## Changes

### 1. Data-shaped buffer accesses (`frame_graph.rs`)

Extend the access enums alongside the texture variants:

```rust
pub enum BufferRead {
    Uniform { stages: ShaderStages },
    Storage { stages: ShaderStages },
    CopySource,
    Vertex,
    Index,
}

pub enum BufferWrite {
    Storage { stages: ShaderStages },   // ReadWrite when compute lands
    CopyDestination,
}
```

Add `BufferHandle`, graph-level `BufferDesc { size }` (memory domain stays an
allocation concern, like texture usage today), and
`create_buffer` / `import_buffer`. Builder methods gain
`read_buffer(handle, BufferRead[, range])` / `write_buffer(handle, BufferWrite[,
range])`.

### 2. Tracking model

Buffers have no layouts; hazards are pure RAW/WAR/WAW on byte ranges.

- Track one state per buffer initially: `(last_write: Option<BufferWrite>,
  last_reads: stage set)` — simpler than textures because there is no layout
  dimension. A barrier is needed when a read follows a write from a *different*
  pass, or write follows anything.
- Byte ranges: reuse `SubresourceRange`-style thinking with `BufferRange {
  offset, size }`; start whole-buffer, keep range fields in declarations.
- Emit existing `dirk_rhi::BufferBarrier`s into the pass pre-barrier list.

### 3. RHI gap: `BufferBarrier` carries no access modes

`command.rs::BufferBarrier` has only `src_stages`/`dst_stages`; Vulkan's
`vkCmdPipelineBarrier2` needs src/dst access masks. Either:

- add `src_access: AccessType` / `dst_access: AccessType` semantic fields
  (preferred; mirrors `ImageState` philosophy), or
- document that backends derive conservative full-memory access masks.

Prefer the first: extend the Vulkan backend's lowering with a small
`convert::buffer_access` mapping (`Storage{COMPUTE} → SHADER_WRITE`, etc.).

### 4. Resolution & callbacks

`PassContext::resolve_buffer(handle)` returns `&ResolvedBuffer { buffer:
&ActiveBuffer }`. Debug-validate declared-vs-used direction like textures.

### 5. Migrate upload paths

Move egui font/atlas uploads and staging copies through graph-declared buffer
writes where practical; short-lived `begin_single_time` uploads may stay
immediate but must be documented as outside graph ordering.

## Validation

- Unit tests in `frame_graph.rs`: WAW on same buffer emits barrier; RAW across
  passes emits barrier; read-after-read does not; disjoint byte ranges don't.
- Vulkan validation layers clean on an upload + draw frame.

## Non-goals

Indirect buffers/dispatch, persistent descriptor caching, host-write hazard
tracking (documented CPU-side discipline only).
