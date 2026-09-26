# Compute shaders (deferred)

Add compute only with an actual workload. Extend shader generation/reflection and
introduce a compute pipeline, recording scope, and one semantic compute queue.
Use explicit barriers and completion dependencies; native queues may alias.
The graph already tracks whole buffers and image mips. General pass scheduling
is a separate decision. Keep renderer data out of the utility and RHI layers.
