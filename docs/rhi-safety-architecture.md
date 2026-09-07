# Proposed RHI safety architecture

Status: implementation authorized across the PR stack. This document preserves
the design discussion; [the runtime contract](rhi-runtime-contract.md) describes
the shared layer and its initial limitations. See [the RHI evolution notes](rhi-evolution.md) for the
related agreed work.

## Direction

Combine a shared safe RHI layer with a small set of types expressing recording
lifecycles and queue capabilities. Implement portable ownership and validation
rules once, and use borrowing and consuming operations to prevent common misuse.
Keep the type system approachable; encoding every resource state in types is
not a goal.

```text
Renderer / render graph
          ↓
Safe RHI: resources, recording scopes, validation, lifetime tracking
          ↓
Native backend: allocation, encoding, barriers, submission, completion
          ↓
Vulkan / Metal
```

## Responsibilities

The native backend supplies more than factories. Its operations cover native
resource creation/destruction, allocation, command encoding, barriers, mapping
and cache management, submission, completion observation, and presentation.
Expose a small, explicitly unsafe contract whose requirements the shared layer
must uphold. Backend code remains responsible for correct native translation.

The shared safe layer owns resource identity and metadata, portable validation,
command lifecycle, resource retention, pending submissions, and coordination
between host access and GPU execution. A generic `Rhi<B>` or `Device<B>` has a
clear purpose when it implements these responsibilities.

The renderer and graph retain pass scheduling, queue assignment, transient
allocation policy, and dependency planning. Avoid independently scheduling the
same workload in both the graph and RHI. Validating the graph's output is a
different responsibility from planning it.

If safe RHI operations expose explicit synchronization, define how incorrect or
missing dependencies are handled. A path trusting graph-generated commands
would require an explicit internal proof boundary or unsafe contract; ordinary
safe callers must not be able to bypass safety checks merely by claiming their
commands were validated.

## Resource ownership extends through GPU completion

RAII describes CPU ownership, but GPU use can continue after a caller drops a
handle. The intended lifecycle is:

1. Recording retains all resources required by its commands, including those
   reached through bindings and views.
2. Submission transfers that retained ownership into pending work.
3. Callers may release their own resource and command handles.
4. The RHI observes native completion.
5. Pending work releases its retained resources; native destruction occurs
   once all required ownership has ended.

Dropping a completion token must remain safe. Pending work must retain what it
needs independently of whether the caller waits. Recycling command storage or
destroying native resources requires completion evidence, not a fixed number
of elapsed frames.

Native object dependencies, such as resources retaining their device and views
retaining images, must also be preserved. Specify cleanup on failed recording,
failed submission, and device loss before implementing the lifecycle.

## Use types for structural facts

| Fact | Preferred mechanism |
|------|---------------------|
| Drawing occurs inside a render pass | Graphics methods on `RenderPass` |
| Encoder cannot be modified while its pass is active | Exclusive borrow |
| Only finished recordings can be submitted | Consuming `finish` returns a command buffer |
| Copy-only encoders cannot draw | Queue/capability marker |
| Resource belongs to a particular device | Runtime identity initially |
| Runtime-supplied range fits an allocation | Runtime validation |
| GPU has finished a submission | Native completion observation |
| CPU access is compatible with aliased GPU use | Ownership restrictions and/or runtime tracking |

Types can represent evidence obtained at runtime. For example, observing frame
completion could grant a token permitting reuse of that frame's allocations.
The underlying completion observation still has to happen.

### Recording and render-pass scopes

Illustrative API only; these names and signatures are not final:

```rust,ignore
let mut encoder = device.create_encoder::<Graphics>()?;

{
    let mut pass = encoder.begin_render_pass(&attachments)?;
    pass.bind_pipeline(&pipeline)?;
    pass.draw(0..3, 0..1)?;
}

let commands = encoder.finish()?;
let completion = graphics.submit(commands)?;
completion.wait()?;
```

The pass borrows its encoder exclusively. Finishing consumes the encoder, and
submission consumes the recorded command buffer. The RHI may recycle native
storage after completion; moving Rust values does not require fresh native
allocation. Repeated submission of one recording is a separate capability to
consider when a real consumer needs it.

Define how scopes end and how abandonment and errors behave. Safety must not
depend solely on a guard's destructor running: safe Rust can forget a guard,
so finalization/submission must still enforce the necessary native preconditions.

### Queue capabilities

Prefer one generic implementation, such as `CommandEncoder<B, Graphics>` or
`CommandEncoder<B, Compute>`, with a small sealed family of capability markers.
Shared operations are implemented once; graphics operations require graphics
capability. Establish which operations each semantic queue guarantees.

A marker describes a role, not a particular native queue or device. Different
semantic queues may alias one native queue, and different devices may expose
the same marker type. Preserve runtime identity and synchronization for those
cases. Use macros only for repetitive declarations after the design is clear.

Be cautious about `Image<ShaderRead>` and `Buffer<Idle>` types: cloned handles,
independent mip/layer states, and separately recorded command buffers make a
single value's type an awkward description of global resource state.

## Host access policy to settle first

Initial recommendation: reject direct host writes while a buffer is busy, and
offer an explicit staging/upload path for asynchronous updates. Avoid hidden
blocking or backend-dependent choices between waiting, rejection, and staging.

Define equivalent rules for reads, overlapping host access, and submission while
host access is active. Checks and access must be coordinated so a concurrent
submission cannot invalidate a writability check before the write occurs.
Frame-owned upload allocations may later offer a convenient borrowing-based
interface tied to completed frame slots.

## Performance and implementation scope

Generic wrappers retain static dispatch. Marker types and ordinary borrowing do
not themselves require runtime bookkeeping. Resource retention, completion
tracking, and dynamic validation have costs, but the existing safe contract
already requires mechanisms providing those guarantees.

Validate immutable descriptors during creation, reuse validated relationships,
and retain resources at sensible command/submission granularity. Measure actual
recording overhead before adding complex types to eliminate individual checks.
The shared layer should make these costs consistent and optimizable in one place.

## Crate organization

Initial preference:

- `dirk_rhi`: safe wrappers and the low-level backend contract.
- `dirk_rhi_vulkan` / `dirk_rhi_metal`: native implementations depending on that
  contract.
- Renderer: selects and constructs the concrete `Rhi<Backend>`.

This avoids a dependency cycle if `dirk_rhi` does not depend on its implementation
crates. Feature-gated backend modules inside `dirk_rhi` are another option,
especially if the low-level interface should remain private. Decide based on
dependency isolation and visibility after settling the safety responsibilities.

## Further design and validation

First settle host access and submission/completion ownership. Then specify the
small recording/pass API, native unsafe obligations, and failure behavior.
Coordinate access tracking with [the semantic synchronization work](rhi-evolution.md).
Use compile-fail tests for structural guarantees and the deferred
[backend conformance suite](rhi-backend-conformance.md) for native behavior.

## Related design references

- [wgpu architecture](https://wgpu.rs/doc/wgpu/documentation/internals/architecture/index.html):
  a shared safe core over an unsafe portable native abstraction provides a useful
  precedent for the responsibility split. Dirk's required feature scope is smaller.
- [wgpu command encoder](https://docs.rs/wgpu/latest/wgpu/struct.CommandEncoder.html):
  render-pass borrowing illustrates using lifetimes for local recording rules.
