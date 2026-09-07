# RHI shader refactor notes

This expands the planned shader rework described on `ShaderSource` in
`crates/dirk_rhi/src/resource.rs`. It records agreed future work, not a change
to the currently accepted native shader representations.

## Existing direction

- Normalize source languages through a unified translation pipeline.
- Add specialization support.
- Derive bind-group layouts from reflection instead of hand-maintained
  declarations.
- Keep shader authoring, build orchestration, and hot-reload policy outside
  the low-level RHI.

## Findings from the upper PRs

In PR #80, the renderer's `build.rs` computes Metal buffer/texture/sampler
indices during SPIR-V translation. `MetalPipelineLayout` independently computes
the runtime binding indices. Their agreement is an implicit shader/backend
contract that can drift as layouts become more complex. Coordinate conversion
also lives in shader translation and needs an explicit engine convention.

## Required design outcomes

- Establish one authoritative representation of shader resource bindings and
  their backend mapping. Generated binding metadata consumed by the backend or
  shared layout-lowering code are possible designs; choose during the refactor.
- Define how independently compiled stages agree with a merged pipeline layout,
  including sparse binding numbers, unused bindings, and different stage usage.
- Document vertex/clip-space and framebuffer coordinate conventions and which
  layer performs each required conversion, including its relationship to winding.
- Make reflected resource layouts and the selected shader artifact a verifiable
  pair. Report mismatches with shader, group, and binding context.
- Keep backend-native artifacts acceptable at the RHI boundary; portability does
  not require moving the full shader compiler into that layer.

## Scope to decide explicitly

The present binding vocabulary combines sampled textures and samplers and has
no binding arrays or comparison samplers. Decide when separate texture/sampler
bindings, arrays, comparison sampling, and richer storage-image metadata are
needed. Add each with reflection, backend support, and an actual consumer;
these notes do not require implementing every extension immediately.

## Validation during the refactor

Use focused metadata/translation tests for stage-layout agreement, sparse
bindings, and coordinate conversion. Exercise at least one translated rendering
scenario through each actual backend under the separate
[backend conformance plan](rhi-backend-conformance.md). Compilation alone does
not establish that bindings or rendered orientation agree.
