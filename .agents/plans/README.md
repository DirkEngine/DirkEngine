# Renderer Evolution Plans

Ordered plans for growing the RHI + render graph. Each plan is self-contained;
dependencies between them are explicit.

| Plan | Topic | Depends on |
|------|-------|------------|
| [01 — Compute shaders](01-compute-shaders.md) | `ComputePipeline`, dispatch, shader build support | 02 |
| [02 — Buffer tracking](02-buffer-tracking.md) | Buffers as first-class frame graph resources | — |
| [03 — Transient allocator](03-transient-resource-allocator.md) | Heaps, placed resources, aliasing | landed usage derivation |
| [04 — Multi-queue scheduling](04-multi-queue-scheduling.md) | Per-pass queue assignment & cross-queue sync | 01 (03 recommended) |
| [05 — Resource registry](05-resource-registry.md) | Long-lived tracked imports across graph runs | — |

Dependency graph:

```text
02 ──► 01 ──► 04
 ▲             ▲
 │             │ (recommended)
(landed) ─► 03 ┘
        05 ────┘
```

## Ground rules established by the current design

These invariants must hold for every plan below:

1. **Semantic states live in the RHI** (`ImageState`, `QueueType`); native sync
   lives in the backends (`convert::image_state`). The graph never names a
   layout, stage mask, or access mask.
2. **The graph computes abstract before→after transitions; backends lower
   them.** No backend re-derives dependency analysis.
3. **Declarations are data-shaped**: accesses are enum values carrying stages
   and subresource ranges; the compiler may deliberately under-use the detail.
4. **Passes declare usages up front; usage capabilities (`ImageUsages`) are
   derived by the compiler and validated for imported resources.**
5. Pass callbacks receive a `PassContext` with validated resource resolution,
   not raw device + slice access.
