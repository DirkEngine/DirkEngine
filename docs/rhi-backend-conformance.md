# Deferred RHI backend conformance plan

Status: agreed future work; do not introduce GPU-dependent CI jobs as part of
the architecture discussion. Revisit after the ownership/completion contract
and portable baseline are defined in [the RHI evolution notes](rhi-evolution.md).

## Objective

Run a small shared suite against actual Vulkan and Metal backends to establish
that the portable contract has the same observable behavior. Keep existing mock
tests for API composition; do not use them as evidence of native correctness.

## 1. Establish the execution environment

- Determine supported development and CI hosts, native API availability, and
  validation/debug facilities without assuming hosted runners expose a GPU.
- Investigate a software Vulkan implementation for headless Linux coverage;
  validate its suitability before selecting dependencies or CI configuration.
- Use a suitable macOS host for native Metal. A Vulkan implementation on macOS
  does not substitute for exercising Metal.
- Separate headless resource/render tests from windowed presentation tests.
  Investigate display-session and event-loop/thread requirements for the latter.
- Start with explicit local or opt-in runs. In optional discovery mode, report
  unavailable backends as skipped with a reason. In jobs explicitly assigned a
  backend, missing backend support must fail rather than silently pass.

## 2. Build a minimal shared harness

Parameterize scenarios over the RHI backend (or its replacement after the safety
design settles). Keep backend/OS setup at the edges. Record backend, adapter,
capabilities, and validation diagnostics in failure reports. Use bounded waits
and an outer test/process timeout so device or presentation failures do not hang
the suite. Avoid a large mock GPU or an engine-wide integration harness.

## 3. Add focused headless scenarios

1. Upload, copy, and read back known buffer data; verify bytes after completion.
2. Upload and partially update a texture with a layout requiring row padding;
   verify copied/read-back pixels and untouched regions.
3. Render a simple offscreen result and verify selected pixels with appropriate
   tolerances. Cover shader/layout agreement and a real resource transition.
4. Record and submit work, release caller-owned resource handles, wait for
   completion, and verify the result using a retained readback resource. Cover
   the contract's required retention before submission as well as in flight.
5. Check command reuse, host access, and completion according to the chosen
   safety policy. Use compile-fail tests for misuse excluded by types; exercise
   only the documented safe rejection paths for remaining dynamic cases.

Avoid timing-dependent assertions that assume a short GPU job is still pending.
If a scenario needs controlled pending work, design a bounded synchronization
fixture with a guaranteed release/cleanup path. Do not deliberately violate the
unsafe native contract to test validation.

## 4. Add a separate presentation suite

On supported windowed hosts, exercise acquire/submit/present, explicit discard,
abandonment of an acquired frame, resize/recreation, and stale-frame handling
where the public API permits it. Verify outcomes and recovery rather than
specific native swapchain counts or exact platform error timing. Keep these
tests independently runnable from the headless suite.

## 5. Extend coverage with real features

Add targeted timeline/multiple-queue handoff tests when such consumers exist.
Distinguish semantic queues aliasing one native queue from actual independent
queues. Test optional operations only when advertised, and verify the specified
unsupported behavior otherwise.

## Completion criteria

The same portable scenarios have documented successful runs on Vulkan and Metal;
headless and presentation requirements are explicit; failures include useful
native context; and unavailable environments are distinguishable from successful
validation. Enable required CI coverage only after runner support is established.
