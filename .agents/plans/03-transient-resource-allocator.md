# Plan 03 — Transient resource allocator (heaps & placed resources)

Give the graph a real memory strategy: compute lifetimes from compiled passes,
pack compatible transients into backend-provided heaps, and support aliasing —
without the graph ever naming memory types or heaps' native representations.

## Dependencies

- Landed: usage derivation (transient inference from declared accesses).
- Recommended first: plan 05 (registry provides the persistent home so
  allocations survive across frames instead of per-frame churn).
- Interacts with plan 04: multi-queue lifetimes need interval-per-timeline
  care (noted below).

## RHI surface additions

Follow the "graph knows *that* memory can alias; backend knows *how*" split:

```rust
pub struct MemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    /// Opaque backend compatibility class (e.g. same memory-type index on
    /// Vulkan, same heap mode on Metal). Equal classes may alias.
    pub class: MemoryClass,
}

// Backend:
fn memory_requirements(&self, desc: &ImageDesc<'_>) -> MemoryRequirements;
fn create_heap(&self, desc: &HeapDesc { size, class }) -> Result<Self::Heap>;
```

Placement is a field on existing descs, not parallel constructors:

```rust
pub enum Placement<'a, B: Backend> {
    Auto,                    // today's behaviour: dedicated allocation
    Placed { heap: &'a B::Heap, offset: u64 },
}
```

Backends:

- Vulkan: allocate `VkDeviceMemory` per heap class, `vkBindImageMemory2` at
  offsets (the current gpu_allocator path becomes the `Auto` implementation;
  heap-backed binding replaces per-image allocations).
- Metal: `MTLHeap` with aliasing-friendly placement.
- D3D12 (future): placed resources are the default anyway — this surface is a
  prerequisite for that backend.

### Aliasing barrier

```rust
pub enum Barrier<'a, B: Backend> {
    Transition(ImageBarrier<'a, B>),
    Aliasing { before: Option<&'a B::Image>, after: &'a B::Image },
}
```

Vulkan lowers to an ownership/layout reset barrier between aliased images;
Metal/D3D12 have native semantics. The graph emits `Aliasing` when a heap
range is recycled within one submission.

## Graph-side algorithm

1. From `CompiledPass` order, derive per-texture lifetime intervals
   `[first_used_by, last_used_by]`.
2. Query `memory_requirements` per transient desc; group by `class`.
3. Within a group, bin-pack non-overlapping intervals onto offsets (simple
   first-fit decreasing is fine initially).
4. Emit `HeapDesc`s + placement table into the compiled graph.
5. `GraphExecutor` asks the allocator instead of `device.create_image`;
   allocator caches heaps across frames keyed by `(class, size-bucket)`.

The executor seam already isolates provisioning — no call-site changes beyond
it.

## Multi-queue note

With plan 04, lifetimes become per-queue-segment intervals and recycling a
range across queue boundaries requires an `Aliasing` barrier plus timeline
ordering. Keep v1 conservative: only alias within a single segment.

## Validation

- Unit tests: interval packing (overlap detection, offset assignment), class
  grouping.
- RenderDoc/Vulkan validation: aliased images get correct barriers; no
  overlapping bound ranges reported by `vkBindImageMemory2` validation.

## Non-goals

Defragmentation, heap growth/shrinking policies, budget reporting UI, buffer
placements before plan 02.
