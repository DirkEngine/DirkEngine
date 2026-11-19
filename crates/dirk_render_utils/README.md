# Render utilities

Rendering utilities built on the RHI (`dirk_rhi`).

[`Uploads`] stage host data in frame memory and copy it into new
device-local buffers and images, handing them to graphics work from the
copy queue when the device has a dedicated one.

The design lives in `docs/rhi/architecture.md` at the repository root.

## Tests

Like `dirk_rhi`'s, the integration tests in `tests/` run on a real device
with validation enabled, through the setup in `dirk_rhi_test_shaders`, and
skip on machines without one unless `DIRK_REQUIRE_GPU=1`.
