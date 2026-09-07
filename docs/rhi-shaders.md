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

## Implemented binding and coordinate contract

`dirk_rhi::BindingMap` now allocates the compact per-stage buffer, texture, and
sampler slots used by both SPIR-V-to-MSL translation and Metal pipeline layouts.
Only bindings visible to that stage occupy slots; sparse group/binding numbers
and declarations used only by another stage therefore cannot shift its mapping.
The shared RHI checks the merged bind-group layouts and device slot limits.

Engine shaders use Vulkan-style clip coordinates (depth 0..1), with positive
viewport dimensions and framebuffer/scissor coordinates measured from the top
left. MSL translation flips vertex clip-space Y exactly once to preserve the
same framebuffer positions under Metal's viewport transform. Texture upload row
zero is the top row. Front-face settings are interpreted in framebuffer space;
the backend uses the selected native winding without a second shader-side flip.
GPU orientation and culling verification remains part of the deferred native
conformance plan; compilation does not verify pixels.

Metal format queries use conservative support from Apple's
[feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf),
with 32-bit filtering and MSAA queried from the device. Unsupported depth formats
are rejected rather than aliased to a different storage representation.
