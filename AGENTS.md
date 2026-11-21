# DirkEngine

A pure Rust Vulkan game engine. Plugins (`EnginePlugin`) configure an
`EngineBuilder`; the subsystems they register are started, ticked and shut
down by the engine loop. The root `Cargo.toml` is a virtual workspace:
libraries live in `crates/` and binaries in `apps/`. The `dirkengine` binary
is `apps/dirkengine` (its `src/lib.rs` holds `DefaultPlugins` and its
`src/cli.rs` the CLI), and `xtask` is `tools/xtask`, the development tasks
`cargo xtask` runs; every other member is a crate in `crates/`:

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
- `dirk_rhi`: the render hardware interface: the device and what is
  recorded on it, over one native graphics backend per target: Metal on
  Apple targets, the only crate that uses `objc2-metal`, and Vulkan
  elsewhere. Its native tests' shaders are `crates/dirk_rhi/test-shaders`
  and its rust-gpu crate `dirk_rhi_test_shaders_gpu`, built like the
  engine's.
- `dirk_render_utils`: rendering utilities built on `dirk_rhi`: the render
  graph and staged uploads.
- `dirk_renderer`: the Vulkan renderer. It and `dirk_rhi` are the only crates
  that use `ash`; `docs/dev/rust-conventions.md` says which use `unsafe`.
- `dirk_shaders`: `no_std` types shared by the engine and its shaders. The
  shaders are `dirk_shaders_gpu` in `crates/dirk_renderer/shaders`, compiled
  to SPIR-V by the renderer's build script with rust-gpu.
- `dirk_editor`: the editor's menus, settings and universe windows.
- `dirk_proc`: the procedural macros (`#[derive(Event)]`,
  `#[derive(Component)]`, `#[derive(HeapIndices)]`).
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
  per rust-gpu version, which takes minutes; every worktree shares them. Two
  shader builds at once can race on the target spec under
  `target/spirv-builder`, failing with "error loading target specification:
  EOF while parsing": rerun before calling it a failure.
- Run the engine from the worktree's root (`just run` or `cargo run`, not from
  `apps/dirkengine`). It finds `assets/` and writes logs to `saved/logs`
  relative to the current directory, so running it from another checkout reads
  and writes that checkout's files. `saved/` is
  ignored by git: never commit or delete another checkout's.
- Running the engine opens a window and needs a Vulkan device; debug builds
  also need the Khronos validation layer (see the README). Where the loader or
  the layer isn't installed system-wide, point `LD_LIBRARY_PATH` at the loader
  and `VK_ADD_LAYER_PATH` at the layer. Tests never need a GPU or a window,
  since CI can't count on either: test the logic around GPU code, as the frame
  graph's tests do. The native tests of `dirk_rhi` and `dirk_render_utils` are
  the exception: they use a device when there is one and skip otherwise;
  `DIRK_REQUIRE_GPU=1` makes a missing device fail them (see `dirk_rhi`'s
  README).

## Conventions

The engineering standard is the `sourceorigin-standard` crate, whose README
lists its rules: `just standard` checks them, and the `standard` skill says
how to fix a violation or challenge a rule. DirkEngine keeps its comments:
`rust-comments`, `doc-length` and `config-comments` are switched off in
`standard.toml`. The synced `docs` skill's doc-comment rules and the
`cleanup` skill's check for doc comments that narrate or justify (item 12 of
its `references/checks.md`) don't apply to DirkEngine's existing comments,
which agents must not trim or rewrite. Follow `docs/dev/rust-conventions.md`
for what is particular to DirkEngine, and the style of neighbouring code.
Clippy enforces most rules
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
and on each crate alone. `just check-changed` runs only the checks a branch's
changes since `origin/main` need, and `just standard` the engineering
standard's rules, which CI checks on every change.

Code behind a feature (`editor`, `cli`, `egui`) must build and pass its tests
both with every feature and with none (`just lint`, `just test`,
`just test-minimal`). Mixed sets, such as `editor` without `cli`, aren't
checked: build them yourself when you change that code.

macOS-only code (`target_os = "macos"`) compiles only for Apple targets, and draft
CI skips macOS. When you change it, run `cargo clippy --locked --target
aarch64-apple-darwin -p <crate> --all-features -- -D warnings`, and say in the
pull request what to run on a Mac before merging, such as `just run`.

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
and `docs/` in the same pull request as the change they describe; the `docs`
skill says how. `docs/` holds loose Markdown pages, with no book to build, and
`just doc` checks the rustdoc with warnings denied.
