# Native backend validation

Device tests are ignored by default and enabled explicitly:

```sh
cargo nextest run -p dirk_rhi -p dirk_render_utils -p dirk_renderer --run-ignored only
```

`dirk_rhi/tests/native_validation.rs` runs under Vulkan 1.3 with Khronos
validation: headless devices and views, binding limits, write masks, depth-stencil
pipelines, timeouts, rejected submissions, and view lifetimes.
`dirk_render_utils/tests/upload.rs` and `tests/graph.rs` exercise transfer upload
and graphics acquisition with readback, layered uploads, dropped resource
retirement before submission, clear-only graphs, nonzero-mip attachments, and
multisample resolves. `dirk_renderer` checks presentation encoding, GPU mip
generation, and glTF model uploads with zero validation errors. The utility and
renderer tests can run on Metal or Vulkan; Vulkan software drivers are sufficient. CPU tests cover graph dependencies, uninitialized reads, mip splitting,
retirement buckets, binding maps, padded uploads, scene deltas, and camera transforms.

CI compiles, lints, and tests on Linux and macOS. Native tests are ignored by default
because ordinary CI runners do not promise GPU access. Cross-compilation establishes
build compatibility, not rendered correctness. Additional native validation should
cover window resize/minimize, presentation recovery, textured draws and winding,
MSAA scene rendering, and a dedicated transfer family under the native
validation/debug layers.
