# Rendering hardware interface

The RHI selects Vulkan or Metal at compile time. Public resource aliases in
[src/lib.rs](src/lib.rs) expose the selected backend through the shared ownership
and validation layer.

## Resource types

The internal [`Api` trait](src/backend.rs) defines resource families for borrowed
descriptors such as `ImageViewDesc` and `BindGroupDesc`. Those descriptors are used
at two boundaries:

- `Rhi<B>` supplies the public wrappers, such as `GpuImage<B>`.
- The native backend supplies its own types, such as `VulkanImage` or `MetalImage`.

Associated types let one descriptor definition describe both boundaries while
preserving the concrete resource type. A single concrete alias would select only
one family; replacing the associated types would require separate descriptor
families or a different conversion API. Callers already use concrete aliases such
as `dirk_rhi::Image`; the generic family stays inside the implementation. This uses
static dispatch.

## Apple dependencies and allocation

These choices were checked against `gpu-allocator 0.28.0`, `metal 0.33.0` and
`core-graphics-types 0.2.0`.

`gpu-allocator 0.28.0` **does support Metal**. Its `metal` module uses `objc2-metal`
0.3 and accepts `Retained<ProtocolObject<dyn MTLDevice>>`, with allocations backed
by `MTLHeap`. This backend uses the separate `metal 0.33`/`objc` resource types.
The allocator therefore requires a binding migration or an audited interoperability
boundary, plus heap allocation and retirement handling.

The current [Metal resource implementation](src/native/metal/resource.rs) creates
buffers and textures directly through `metal::Device` and retires them through the
RHI. Device buffers use private storage; upload/readback buffers use shared storage.
This keeps allocation within the existing binding and ownership model, at the cost
of having no application-managed heap suballocation. The workspace enables only
`gpu-allocator`'s `std` and `vulkan` features for the Vulkan backend. Reconsider the
Metal allocator together with a binding migration and measured allocation needs.
The pinned `metal 0.33.0` crate itself recommends `objc2-metal` for new development.

`core-graphics-types` supplies the C-compatible `CGSize` used by
[`MetalLayerRef::set_drawable_size`](src/native/metal/presentation.rs) when creating
or resizing the presentation layer. `metal 0.33.0` takes that same package's type in
its method signature, so this is a direct dependency for drawable dimensions.

## Scope of the safety boundary

The public wrappers validate descriptors, ranges and recording lifecycle, and the
RHI defers resource destruction until submitted work completes. Recording, host
access and submission still have explicit `unsafe` contracts for GPU
synchronization, image states, non-owning bindings, shader/resource bounds and
submission within the recording cycle; see [the crate contract](src/lib.rs).
Rust's exclusive host borrow alone cannot prove that the GPU has finished using
an allocation.

An unsafe-free renderer remains a separate API design requirement. Safe entry
points must enforce the relevant ownership, completion, access-state and shader
contracts for the renderer's operations. That work should begin with narrow
recording, upload and submission operations, while scene and frame orchestration
remain in the renderer. Keep the explicit caller contracts until those guarantees
are enforced. Documenting the current boundary does not fulfill the unsafe-free
renderer requirement.
