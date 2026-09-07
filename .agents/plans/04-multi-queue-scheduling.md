# Plan 04 — Multi-queue scheduling in the render graph

Let the graph compiler assign passes to semantic queues and emit cross-queue
synchronization, so compute/transfer work can overlap graphics instead of
serializing behind the single graphics submission.

## Dependencies

- Plan 01 (compute): without compute passes there is nothing to schedule.
- Strongly recommended: plan 05 (registry) — imported-resource last-use
  tracking must become queue/timeline-aware when handoffs cross submissions.
- RHI needs no structural change: `submit(queue, Submission { waits, signals })`
  with timelines already supports N batches per frame.

## Design

### 1. Queue affinity

Default affinity derives from access kinds:

- any attachment write or vertex/index draw → Graphics
- storage-only accesses with `ShaderStages::COMPUTE` → Compute
- copy/blit-only passes → Transfer (opt-in; copies on dedicated queues have
  ownership-transfer costs — make it opt-in via `PassBuilder::on(QueueType)`)

### 2. Compilation

Today's strictly-linear pass list becomes a dependency DAG (edges already
implied by read/write declarations). The compiler:

1. Builds edges from resource hazards (reuse the range-overlap logic).
2. Assigns each pass its queue; splits into **segments**: maximal runs of
   passes executable on one queue between required handoffs.
3. Emits per-segment timeline signal + next-segment wait pairs at queue-change
   boundaries. The graph owns a small timeline pool (or accepts one from the
   renderer) rather than signaling per pass — intra-submission ordering stays
   free; only genuine cross-queue edges pay.

```rust
pub struct GraphBatch<'a> {
    pub queue: QueueType,
    pub command_buffer: CommandBuffer,
    pub waits: Vec<TimelinePoint>,   // filled from prior segments
    pub signals: Vec<TimelinePoint>,
}
// RenderGraph::run becomes record() -> CompiledSubmission { batches }
```

### 3. Execution & submission

- Executor records one command buffer per segment (pools per `QueueType`).
- Renderer's `submit_frame` generalizes to iterate batches; presentation keeps
  its existing shape — present-queue work and final `Present` transitions stay
  on graphics in v1 regardless of what offscreen graphs do.

### 4. Imported resources

`final_state` exports that are consumed by a *later* segment must carry the
signal value; plan 05's registry stores `(timeline, value)` per entry so the
consuming graph's import can emit the matching wait. This replaces today's
implicit "same-submission ordering" assumption for viewport outputs.

## Validation

- Unit tests: DAG construction, segmentation (no cross-queue edge missed),
  timeline pairing (every wait has a matching earlier signal).
- Manual: profile a frame with an overlapping compute pass; confirm overlap in
  RenderDoc/GPU captures and clean validation-layer output.

## Non-goals

Automatic workload balancing/rebalancing, host-copy engine heuristics,
frame-overslapping scheduling, per-pass timelines.
