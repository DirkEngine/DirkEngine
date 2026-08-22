# Plan 05 — Non-transient resource registry

Move long-lived imported-image bookkeeping (state tracking, submission
pairing) out of ad-hoc owners like `Viewport` into one renderer-owned registry
that graphs borrow from. The per-frame graph stays transient; only the
*knowledge about* long-lived resources gains a persistent home.

## Dependencies

None. Plan 04 consumes its timeline tracking; TAA-style history buffers will
be its first natural consumer.

## Problem being solved

Today `Viewport` manually tracks `output_state`, `last_render_value`,
`output_has_rendered`, and the `next_render_value()` / `mark_render_submitted()`
protocol. That is synchronization state embedded in presentation state, updated
by convention — forgetting one call leaves stale tracked state silently. Every
future long-lived shared resource (history buffers, editor previews,
light-bake targets) would copy this boilerplate.

## Design

```rust
pub struct ResourceRegistry {
    entries: SlotMap<ResourceKey, Entry>,
}

struct Entry {
    image: ActiveImage,
    view: ActiveImageView,
    aspects: ImageAspects,
    /// Last known semantic state of the resource.
    state: ImageState,
    /// Submission that produced `state`, when known.
    last_use: Option<TimelinePoint>,
}
```

API surface used by graphs:

```rust
impl ResourceRegistry {
    fn register(&mut self, image: ..., aspects: ...) -> ResourceKey;
    fn remove(&mut self, key: ResourceKey);
    /// initial_state comes from tracking; caller declares the export state.
    fn import(&self, key: ResourceKey, final_state: ImageState) -> ImportedTexture;
    /// Single choke point: called once per completed submission with the
    /// resources it touched.
    fn on_submitted(&mut self, key: ResourceKey, point: TimelinePoint, final_state: ImageState);
}
```

## Migration

1. `Renderer` owns the registry; creates it alongside `RenderDevice`.
2. `Viewport::new` registers its output; `resize`/`reconfigure` does
   `remove` + `register` (state resets to `Undefined` naturally).
3. `record_viewport_graph` imports via `registry.import(key, ShaderRead)`.
4. The fence-wait / submit path calls `on_submitted` — replacing
   `mark_render_submitted`. `Viewport` keeps only settings/camera/world.
5. Swapchain frames join later if ever: their boundary semantics are fixed
   (`Undefined → Present`) so they gain little.

## Interaction notes

- Plan 03's allocator may live beside the registry (registry = logical
  long-lived resources; allocator = physical transient memory); both are
  renderer-owned, graph-consumed.
- Plan 04 reads `last_use` to emit consumer waits across segments/submissions.

## Validation

- Unit tests: import→submit→re-import cycles produce correct initial states;
  remove+register resets; missing `on_submitted` is debug-asserted (turning
  today's silent staleness into a loud contract violation).

## Non-goals

Cross-process sharing, residency/priority management, automatic eviction.
