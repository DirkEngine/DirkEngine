# Rendering follow-ups

The [current contract](../../docs/rhi/runtime-contract.md) takes precedence over old
exploration notes. Whole-buffer graph tracking and batched transfer uploads are
implemented. Compute, renderer threading, and a proper transient allocator remain
future work. A resource registry and reference tracker are not required.

- [Compute](01-compute-shaders.md)
- [Buffer tracking status](02-buffer-tracking.md)
- [Transient allocator](03-transient-resource-allocator.md)
- [Queue scheduling status](04-multi-queue-scheduling.md)
- [Registry decision](05-resource-registry.md)
