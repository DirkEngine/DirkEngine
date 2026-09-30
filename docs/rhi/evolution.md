# Deferred rendering work

The implemented contract is documented in [runtime-contract.md](runtime-contract.md).
Keep follow-up work separate and motivated by a real consumer:

- A proper transient resource allocator, with reuse/aliasing and measured budgets.
- Compute dispatch and graph queue scheduling beyond the current upload handoff.
- Moving renderer execution to another thread using the final-state delta boundary.
- Shader compilation/reflection improvements described in [shaders.md](shaders.md).
- Indirect rendering and device-loss recovery when needed.

A global resource registry and resource reference tracker are not planned requirements.
The current graph already tracks whole buffers and mip spans. Do not introduce
repeated-allocation caching as a substitute for the planned transient allocator.
