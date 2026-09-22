# Native backend validation

`dirk_render_utils/tests/upload.rs` contains explicitly enabled device tests:

```sh
cargo nextest run -p dirk_render_utils --run-ignored only
```

They exercise transfer upload and graphics acquisition with byte readback, dropped
resource retirement before submission, clear-only graph execution with pixel
readback, and device-idle collection while current-cycle commands remain pending.
They can run on Metal or Vulkan; Vulkan software drivers are sufficient for these
checks. CPU tests cover graph dependencies, uninitialized reads, mip splitting,
retirement buckets, binding maps, padded uploads, scene deltas, and camera transforms.

CI compiles, lints, and tests on Linux and macOS. Native tests are ignored by default
because ordinary CI runners do not promise GPU access. Cross-compilation establishes
build compatibility, not rendered correctness. Additional native validation should
cover window resize/minimize, presentation recovery, textured draws and winding,
MSAA, and a dedicated transfer family under the native validation/debug layers.
