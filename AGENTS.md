# DirkEngine

A pure Rust Vulkan game engine. Plugins (`EnginePlugin`) configure an
`EngineBuilder`; the subsystems they register are started, ticked and shut
down by the engine loop. The root `Cargo.toml` is a virtual workspace:
libraries live in `crates/` and binaries in `apps/`. The `dirkengine` binary
is `apps/dirkengine` (its `src/lib.rs` holds `DefaultPlugins` and its
`src/cli.rs` the CLI); every other member is a crate in `crates/`:

- `dirk_engine`: the engine, its builder, plugins, subsystems, resources and
  the editor's dock and commands.
- `dirk_universe`: the ECS: the universe, its worlds, entities, components,
  systems and command buffers.
- `dirk_events`: the channel-backed pub/sub event bus, routed on worker
  threads (see its README).
- `dirk_threads`: the worker pool and threading primitives, on tokio.
- `dirk_assets`: asset handles, `.dirkasset` descriptors and the asset registry.
- `dirk_platform`: windows and OS events, the only crate that uses `winit`.
- `dirk_input`: input actions, axes and maps, and their egui mapping (`egui`
  feature).
- `dirk_world`: the components and systems that populate a world (transforms,
  renderables, model uploads).
- `dirk_player`: players, their input and movement.
- `dirk_renderer`: the Vulkan renderer, the only crate that uses `ash` and
  `unsafe`.
- `dirk_shaders`: `no_std` types shared by the engine and its shaders. The
  shaders are `dirk_shaders_gpu` in `crates/dirk_renderer/shaders`, compiled
  to SPIR-V by the renderer's build script with rust-gpu.
- `dirk_editor`: the editor's menus, settings and universe windows.
- `dirk_proc`: the procedural macros (`#[derive(Event)]`,
  `#[derive(Component)]`, `#[derive(System)]`, `#[system_trait]`).
- `dirk_utils`, `dirk_build`: small shared utilities, and build script helpers.

`docs/dev/rust-conventions.md` describes how the crates depend on each other.

## Glossary

- **Universe / world / `dirk_world`:** the `Universe` is the whole ECS and
  owns every `World`; a `World` holds entities and their components.
  `dirk_world` is not the ECS world: it holds the components and systems that
  fill one.
- **Subsystem / system:** a subsystem (`dirk_engine`) is runtime state the
  engine loop drives. A system (`dirk_universe`) is ECS logic the universe
  runs over entities or components.
- **Plugin / resource:** a plugin only configures the builder and is never
  ticked. A resource is a cheap, cloneable handle a subsystem factory
  publishes so later factories can find it.
- **Command buffer:** `dirk_universe::CommandBuffer` records edits to the
  universe; a Vulkan command buffer records GPU work. They are unrelated.
- **Scene / proxy:** the renderer's copy of ECS state. `SceneManager` mirrors
  the universe, a `Scene` one world and a `SceneProxy` one entity; the
  renderer's ECS systems keep them in sync.
- **Frame graph / render commands:** the frame graph (`RenderGraph`, in
  `frame_graph.rs`) runs a frame's passes in the order they were added and
  inserts the barriers between them. Render commands are closures other
  threads queue to run on the renderer; they are internal to it.
- **`AssetHandle` / `Handle<T>`:** an `AssetHandle` names an asset (a path
  under `assets/` and a type); a `Handle<T>` is loaded, ref-counted data, and
  dropping the last one unloads it.
- **Editor:** the `editor` feature, the `EditorSubsystem` trait and dock in
  `dirk_engine::editor`, and the `dirk_editor` crate of default windows.

## Hazards

Other agents work in parallel worktrees on this machine, and the developer
uses it too. Before anything destructive, ask what else it touches.

- Stop only a process you started, by the PID you captured. Never kill by
  name or pattern (`pkill -f`, `killall dirkengine`): your own agent process
  has the worktree's path in its arguments, and other worktrees run the same
  binary.
- Never delete `~/.cache/rust-gpu` (`~/Library/Caches/rust-gpu` on macOS) or
  the nightly toolchains rustup lists. The renderer's build script installs a
  pinned nightly and compiles rust-gpu's codegen backend into that cache once
  per rust-gpu version, which takes minutes; every worktree shares them.
- Run the engine from the worktree's root (`just run` or `cargo run`, not from
  `apps/dirkengine`). It finds `assets/` and writes logs to `saved/logs`
  relative to the current directory, so running it from another checkout reads
  and writes that checkout's files. `saved/` is
  ignored by git: never commit or delete another checkout's.
- Running the engine opens a window and needs a Vulkan device; debug builds
  also need the Khronos validation layer (see the README). Tests never need a
  GPU or a window, since CI can't count on either: test the logic around GPU
  code, as the frame graph's tests do.
- Don't set `RUSTFLAGS`: it replaces the `.cargo/config.toml` rustflags that
  set the `platform_*` cfgs. macOS-only code then silently compiles out: the
  build passes and the engine fails at run time on MoltenVK.

## Conventions

Follow `docs/dev/rust-conventions.md`, which applies the shared
[engineering standard](https://gist.github.com/PiquelChips/04330e52c03efbe9ce4c889c88673a27)
to DirkEngine, and the style of neighbouring code. Clippy enforces most rules
(`[workspace.lints]` in `Cargo.toml`, `clippy.toml`), and `deny.toml` keeps
`ash`, `winit` and `egui` behind the crates that wrap them.

When a dependency's API matters, read its source in `~/.cargo/registry/src`
at the version in `Cargo.lock`, not what you remember: ash, winit, egui and
rust-gpu change fast.

If a rule fights the task at hand, say so plainly and get the owner's
sign-off before breaking it.

## Validation

While iterating, check only the crates you changed:

```sh
just fmt
cargo clippy --locked -p <crate> --all-targets --all-features -- -D warnings
just test -p <crate>
```

If you changed a doc comment, a README or a public API, also run `just doc`
and `just doc-test`: nextest runs no doc tests, and CI denies rustdoc
warnings.

Before handing back a change that spans crates or touches the workspace,
run `just validate` (or `just` alone): it formats the sources and runs every
check CI runs, through the same recipes. `just lint` runs Clippy with
`--all-targets -- -D warnings` with every feature, with no default features
and on each crate alone.

Code behind a feature (`editor`, `cli`, `egui`) must build and pass its tests
both with every feature and with none (`just lint`, `just test`,
`just test-minimal`). Mixed sets, such as `editor` without `cli`, aren't
checked: build them yourself when you change that code.

If a step can't run, such as for a missing tool or system dependency, report
the exact command that failed and why.

## Surfaces

Before handing back a feature, walk this list and say which entries applied:

- Its plugin is registered in `DefaultPlugins` (`apps/dirkengine/src/lib.rs`),
  and in the CLI (`apps/dirkengine/src/cli.rs`) and the root README's example,
  which repeat the list.
- It works with the `editor` feature and without it; an editor window or
  command sits behind the feature.
- Its input goes through `dirk_input` bindings, and its events are
  registered on the event bus.
- What it creates it can remove: entities despawn and worlds are destroyed
  (`CommandBuffer::despawn`, `destroy_world`) along with their renderer
  proxies, and a loaded asset unloads when its last `Handle` drops. What it
  changes can be seen, in the editor or the logs.
- Its crate README, doc comments and `docs/` describe it.

## Issues, commits and pull requests

Issues live in GitHub Issues. Titles of issues and pull requests follow
Conventional Commits, such as `fix(renderer): recreate the swapchain on resize`,
and say what changes for the reader. Pull requests are squash-merged, so the
title becomes the commit on `main`.

Keep pull requests small and focused: one change each, which a reviewer can
read in one sitting. Split refactors from behaviour changes, and stack
dependent pull requests, each on the last. A pull request that closes an
issue says `Fixes #123`.

Open pull requests as drafts while iterating: CI then skips macOS, its
slowest runner. Mark them ready for review to run the full matrix.

## Documentation

Each crate's README is its crate documentation. Update it, the doc comments
and `docs/` in the same pull request as the change they describe.
