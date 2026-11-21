# rhi

The engine's render hardware interface (RHI).

One native backend is compiled per target and wrapped by the safe types in
this crate. There are no backend traits or generic backend parameters: each
native module exposes the same crate-private types and methods, and the
public types hold a native object plus the immutable information needed to
validate its use. Requests are validated once, when objects are created.

`Rhi::new` creates the instance and device. This crate is the only one that
calls the native graphics API, so it is where most of the engine's `unsafe`
code lives; every unsafe block says why it is sound.

The binding design lives in `docs/rhi/architecture.md` at the repository
root.

## Tests

The integration tests in `tests/` run on a real device with validation
enabled, and fail if validation reports an error. On machines without a
graphics driver or a suitable device they print why and pass without running;
set `DIRK_REQUIRE_GPU=1` to make them fail instead, as CI does on Linux, where
they run on Mesa's software driver (lavapipe).
