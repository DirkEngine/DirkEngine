# Transient allocator (deferred)

The user has a proper transient resource system planned. Do not add a temporary
allocation cache or resource registry as a prerequisite. The present graph derives
creation usages and owns transient allocations for a single recording; retirement
is handled by the RHI.

A future design should address allocation reuse, lifetime intervals, memory
compatibility, budgets, and explicit aliasing dependencies on both Metal and
Vulkan. Validate pixel/buffer results and memory behavior under actual workloads.
Choose its API and placement separately; these notes do not authorize implementation.
