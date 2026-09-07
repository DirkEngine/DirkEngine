# RHI evolution notes

These notes preserve the architecture discussion of PR #77, informed by the
Vulkan (#78), renderer integration (#79), Metal (#80), and egui (#82) PRs.
The implementation is now authorized. See [the runtime contract](rhi-runtime-contract.md)
for the shared layer's implemented guarantees and explicit initial limitations.
The remaining notes retain the design rationale and deferred scope.

## Proposed: ownership and safety architecture (review point 2)

Recording and render-pass scopes are agreed directions. The preferred direction
is a shared safe RHI layer over unsafe native backend operations, combined with
focused types for command lifecycles and queue capabilities. See
[the proposed safety architecture](rhi-safety-architecture.md) for responsibilities,
ownership, tradeoffs, and open decisions. The user subsequently authorized implementation across the PR stack; broader
features explicitly marked deferred remain outside this change.

The design must assign responsibility for resource retention, GPU completion,
host/GPU access exclusion, command-buffer reuse, and native external
synchronization. Distinguish Rust memory safety from rendering correctness;
document any caller obligations that cannot be checked. Define whether host
access to a busy resource rejects, waits, or stages, rather than leaving its
operational behavior arbitrarily backend-dependent.

## Agreed: consistent resource access semantics (review point 3)

- Use a consistent semantic access model for buffers and images. Preserve
  shader stages, access kind, and byte/subresource ranges through the RHI.
- Distinguish sampled reads, storage reads, storage writes/read-writes,
  attachments, and copies where their synchronization or usage differs.
- The graph determines dependencies; backends lower them to native barriers
  and ownership transfers. Avoid separate dependency analysis in each backend.
- Conservative lowering is acceptable initially, but must not require callers
  to discard information needed for later precision or multiple queues.
- Coordinate this work with the upper-stack buffer-tracking and compute plans.
  Queue/timeline types alone do not establish correct cross-queue handoffs.

## Agreed: define portable support and fallback (review point 4)

- Document the guaranteed baseline, optional queryable operations, and explicit
  preferences that may fall back.
- An accepted operation must preserve its specified effects, including the
  exact subresources affected. A regional blit cannot silently become an
  operation over unrelated mip levels.
- Add focused support queries for real callers, including relevant filtering,
  blending, blitting, and resource/binding limits. Let callers select fallback
  paths before command recording begins.
- Keep preferences explicit and expose selected results where needed. Avoid
  silently ignoring behavior that was requested as a requirement.
- Specify error and command-state behavior when recording or submission fails.

## Agreed: shader/layout agreement (review point 5)

See [the shader refactor notes](rhi-shaders.md). Preserve the existing planned
translation, specialization, and reflection work while giving shader binding
mapping and coordinate conventions one authoritative definition.

## Agreed: presentation lifecycle (review point 6)

- Preserve owned surface targets, acquired frame objects, selected metadata,
  and consuming present/discard operations.
- Clarify the intended acquire/submit/present protocol and exactly-once frame
  association. The current model fits offscreen work followed by a final
  submission touching the surface; work touching a surface image across several
  submissions needs an explicitly defined protocol.
- Specify what remains usable after failed acquisition, recording, submission,
  presentation, or discard, and how callers recover or recreate the surface.
- Keep native presentation synchronization inside the RHI/backend boundary.

## Agreed: reduce repetitive caller mechanics (review point 7)

- Introduce reusable image subresource ranges and buffer slices/ranges.
- Add a few common descriptor constructors without a large builder framework.
- Provide a renderer upload helper for staging, pitch alignment, copies, and
  transitions. Keep upload scheduling and batching above the RHI.
- Expose portable image metadata needed by imports and validation: format,
  extent, usages, mip/layer counts, and sample count. Avoid independently
  supplied metadata that can disagree with the actual image.
- Use the asset and egui upload paths as consumers to verify that these changes
  remove duplicated decisions while retaining explicit graphics behavior.

## Agreed: grow through complete features (review point 8)

Keep the current raster/transfer scope honest. Add operations across the API,
backends, and a real renderer consumer together. Compute dispatch, indirect
execution, binding arrays, and transient allocation are future work; vocabulary
alone does not make them supported. Preserve the separation between broad RHI
capabilities and narrower typed renderer conveniences.

The upper-stack transient allocator plan will need an allocation/placement
interface; current backend-owned allocations do not expose graph-controlled
aliasing. Reconcile these notes with the existing renderer plans when restacking.

## Deferred: native backend conformance (review point 9)

See [the backend conformance plan](rhi-backend-conformance.md). Actual-backend
testing is intentionally deferred while GPU availability and OS/CI constraints
are investigated. Mock contract tests remain useful but do not prove native
backend behavior.
