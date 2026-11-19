# DirkEngine

A pure Rust Vulkan game engine, built from plugins (`EnginePlugin`) that
configure an `EngineBuilder` and runtime subsystems that the engine loop
starts, ticks and shuts down. The root package is the `dirkengine` binary;
everything else is a crate in `crates/`:

- `dirk_engine`: the engine, its builder, plugins, subsystems and editor support.
- `dirk_universe`: the ECS: worlds, entities, components, systems, command buffers.
- `dirk_events`: the frame-buffered pub/sub event bus (`docs/events.md`).
- `dirk_threads`: the worker runtime and threading primitives.
- `dirk_assets`: asset handles, `.dirkasset` descriptors and the asset registry.
- `dirk_platform`: windows and OS events, the only crate that uses `winit`.
- `dirk_input`: the input API, and its egui mapping (`egui` feature).
- `dirk_world`: the components and systems that populate a world.
- `dirk_player`: players, their input and movement.
- `dirk_renderer`: the Vulkan renderer, the only crate that uses `ash` and `unsafe`.
- `dirk_shaders`: types shared by the engine and its shaders, which live in
  `crates/dirk_renderer/shaders` and compile to SPIR-V with rust-gpu.
- `dirk_editor`: the editor's menus and windows.
- `dirk_proc`: the procedural macros (`#[derive(Event)]`, `#[derive(Component)]`,
  `#[derive(System)]`, `#[system_trait]`).
- `dirk_utils`, `dirk_build`: small shared utilities, and build script helpers.

## Conventions

Follow `docs/dev/rust-conventions.md`. It applies the shared
[engineering standard](https://gist.github.com/PiquelChips/04330e52c03efbe9ce4c889c88673a27)
to DirkEngine. Keep changes focused and follow the style of neighbouring code.

## Skills

Skills live in `.agents/skills`, linked from `.claude/skills`.
`dirk-cleanup` audits the repository and runs a cleanup pass across it.

## Validation

`just validate` (or `just` alone) formats the sources and runs every check CI
runs: Clippy with warnings denied on every target, with every feature, with no
default features and on each crate alone, the tests with every feature and with
none, the docs and doc tests, the dependency audit and the unused dependency
check. Run it before handing work back. Mixed feature sets, such as `editor`
without `cli`, aren't checked: build them yourself when you change that code.

While iterating, run only what you changed, such as `just test -p <crate>`.
When you change feature-gated code (`editor`, `cli`, `egui`), check it both
with every feature and with none.

If a validation step cannot be run because a required tool or system dependency
is unavailable, report the exact command that failed and the reason.

## Issues, commits & pull requests

Issues live in GitHub Issues. Titles of issues and pull requests follow
Conventional Commits, such as `fix(renderer): recreate the swapchain on resize`,
and say what changes for the reader. Pull requests are squash-merged, so the
title becomes the commit on `main`.

Keep pull requests small and focused: one change each, which a reviewer can
read in one sitting. Split refactors from behaviour changes, and stack
dependent pull requests, each on the last. A pull request that closes an
issue says `Fixes #123`.

Open pull requests as drafts while iterating: CI then skips macOS, its slowest
runner. Mark them ready for review to run the full matrix.

## Documentation

Each crate's README is its crate documentation. Update it, the doc comments,
and `docs/` in the same pull request as the change they describe.
