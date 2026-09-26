# Rendering contract

Rendering has three layers:

- `dirk_renderer` owns one `Rhi`, frame slots, scene snapshots, cameras, asset uploads,
  windows, graph construction, submission, and presentation.
- `dirk_render_utils` supplies graphs, batched uploads, typed host buffers, bindings,
  and pipelines. It depends on the public RHI, never renderer state or native backends.
- `dirk_rhi` selects private Metal implementations on Apple targets and Vulkan
  implementations elsewhere at compile time. Public resource types are concrete.

## Ownership and retirement

Resources have unique, non-cloneable owners. CPU writes require `&mut`; reads use
`&`. Native allocations share only an internal device/context owner. Image views
and bind groups are non-owning references: keep their source allocations valid
through the last recording, or retire them in that recording's submission cycle.
Do not record later work through a binding after its source owner was dropped.

Dropping a resource places its native payload in the current retirement bucket.
`Rhi::finish_cycle` seals that bucket with all graphics and transfer submissions.
Normal collection waits at least three cycles and checks completion of all work
through the retiring cycle. Frame count alone never proves completion. Record
and submit within one cycle; recorded commands are consumed on submission.

`wait_idle` waits for submitted work and collects sealed retirements, preserving
resources retired in the active cycle because commands may still be unsubmitted.
`flush` seals the active cycle and waits, allowing offscreen callers to drain
without producing more frames. Submit or discard all recordings before flushing.
Device teardown waits for submitted work before destroying native allocations.

The renderer has two frame slots and waits for a slot before updating its GPU
buffers. Completed command allocations are reused. Completion handles can be
waited on by the CPU or supplied as GPU dependencies; dropping one does not
abandon submitted commands. One queue object exists per supported semantic type
(graphics and transfer); hardware queues may alias. Compute is deferred.

## Explicit GPU obligations

This is a practical Rust abstraction, not a proof of arbitrary CPU/GPU ownership.
Creation validates ranges, usages, formats, and interfaces. Unsafe recording,
submission, and host-access methods require correct synchronization, resource
states, shader bounds, device agreement, and non-owning binding lifetimes.
There is no resource reference tracker, busy count, persistent identity registry,
or submission-time state reconciliation. Commands record native operations
immediately; GPU execution starts after queue submission.

Barriers are explicit. The graph derives them from declared accesses; small
callers can use the RHI directly. Uploads use one transfer batch and a matching
graphics acquisition with a GPU completion dependency. Shader artifacts remain
trusted imports with reflected interface checks in the utility/build layers.

## Graph

The graph executes every pass in insertion order, including clear-only passes.
Imports borrow allocations and provide actual initial and desired final states.
Repeated imports of the same allocation share one graph-local handle; conflicting
states are errors. Buffers are tracked as whole allocations. Images are tracked
per mip, covering all aspects and layers; partial aspects/layers are rejected.
The graph catches obvious uninitialized first reads and attachment loads. Coverage
of partial writes remains the caller's responsibility. Stable read-only assets
outside the graph require an explicit unsafe external-read callback contract.
Graph execution is unsafe because external state and synchronization cannot be
verified by the compiler. There is no transient allocator or cross-frame registry.

## Windows and scene data

A platform window has one shared native owner, also retained by the surface.
Mutable focus, occlusion, and theme state stay in the platform wrapper. Retiring a
native surface keeps its window owner until actual native destruction. Zero-size
or occluded windows skip acquisition; resize and out-of-date results recreate
at the next usable acquisition. Device/submission failures remain fatal.

Universe observers mark dirty IDs. After the universe tick, the renderer extracts
one final-state delta per changed entity. CPU proxy updates do not write GPU
memory. Camera uniforms belong to each viewport, not the world. Missing or invalid
cameras make a viewport unavailable; duplicate player cameras report the conflicting
IDs when the conflict changes. Missing-transform and unloaded renderables are skipped.
Camera space is +Y up, +Z forward, with full quaternion rotation including roll;
view transforms ignore scale. Projection uses left-handed Vulkan depth 0..1.
