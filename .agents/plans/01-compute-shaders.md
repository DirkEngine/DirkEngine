# Plan 01 — Compute shader support

Add compute pipelines and dispatch to the RHI and renderer so the frame graph
can schedule compute-only passes (and later async-compute queues per plan 04).

## Dependencies

- Plan 02 (buffer tracking): real compute wants storage-buffer declarations;
  image-only compute would make the first consumer artificial.
- Landed work: data-shaped declarations already carry `ShaderStages::COMPUTE`
  and map to `ImageState::ShaderRead`/`ShaderWrite`.

## RHI changes

### 1. Pipelines

```rust
pub struct ComputePipelineDesc<'a, B: Backend> {
    pub label: &'a str,
    pub layout: &'a B::PipelineLayout,
    pub shader: &'a B::Shader,
}
// Backend: type ComputePipeline; fn create_compute_pipeline(...)
```

Vulkan: `VkComputePipelineCreateInfo` (single stage, `VK_PIPELINE_BIND_POINT_COMPUTE`).
Metal: `MTLComputePipelineState` from the MSL function.

### 2. Command recording

```rust
fn bind_compute_pipeline(&mut self, pipeline: &B::ComputePipeline);
fn dispatch(&mut self, x: u32, y: u32, z: u32);
```

`bind_groups` is already bind-point agnostic. Deliberately out of scope until a
consumer exists: dispatch indirect, push constants, shared memory/slab sizes
(`threadgroup_memory_length`), subgroups. Keep the honest-partial-surface rule:
add dispatch together with pipeline creation, not after.

### 3. Shader build (`dirk_shaders` / build.rs)

- Accept compute entry points in the shader collection macro/build step.
- SPIR-V → MSL via spirv-cross2 for Metal (compute entry maps to
  `kernel void`); reflection already uses rspirv-reflect — extend descriptor
  metadata extraction to `SPV_ENV_COMPUTE` stages.
- Generated Rust side gains a `ComputePipelineSpec` analogue of the graphics
  specs.

## Frame graph changes

Compute passes need no new node type — a pass without attachments already
skips `begin_rendering`. What changes:

- `PassBuilder::read_storage_image(handle, stages)` /
  `write_storage_image(handle, stages)` convenience methods over the existing
  `TextureWrite::Storage` variant (`ImageState::ShaderWrite` →
  `GENERAL` layout, correct sync2 masks already present in
  `convert::image_state`).
- Buffer storage accesses come from plan 02.

## Renderer changes

- `pipeline/compute.rs`: `ComputePipeline<Spec>` mirroring
  `GraphicsPipeline<Spec>`, including bind-group plumbing.
- First consumer: replace the blit-based mip generation or add a trivial
  gradient/storage-image pass exercised in tests + viewport debug view.

## Validation

- `cargo nextest run -p dirk_rhi -p dirk_renderer`.
- Manual run on both backends (lavapipe + MoltenVK) with validation enabled;
  confirm storage-image barriers appear as expected in RenderDoc.

## Non-goals

Async queue scheduling (plan 04), indirect dispatch, specialization constants,
ray tracing.
