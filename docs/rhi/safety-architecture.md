# RHI safety boundaries

The [runtime contract](runtime-contract.md) is authoritative. Rust unique owners,
exclusive CPU writes, borrowed recording scopes, typed graphics/transfer encoders,
and typed utility bindings prevent common mistakes without a resource tracker.

GPU dependencies and persistent non-owning bindings cannot be proven by ordinary
borrowing across asynchronous submissions. They are explicit unsafe obligations,
backed by a simple device retirement queue. Do not add object `Arc`s, busy counters,
state reconciliation, or a resource registry to imply guarantees this API does not
provide. Any stronger model needs a concrete workload and a separate design decision.

Native implementations remain private. Shader bytecode is a trusted boundary;
host records use `bytemuck::NoUninit` and checked shader interfaces. Safe byte
conversion does not prove GPU synchronization or shader bounds.
