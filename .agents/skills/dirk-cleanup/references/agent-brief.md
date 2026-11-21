# Cleanup pass: area agent

You own one area of DirkEngine in a parallel cleanup pass. Your task names the
area, paths, crates, branch prefix, issue and environment, and lists what the
other agents and PRs outside the pass hold.

Read every file in your area, and `docs/dev/rust-conventions.md`, plus
`docs/rhi/architecture.md` for rendering code. Where those disagree, follow the
code as it stands and report the conflict. Make real simplifications, not
cosmetic churn, and bring the code you touch in line with the conventions.
Report bigger issues.

## Rules

- **Change no behavior.** That includes what renders, the layout of types shared
  with shaders and shader bindings, the native calls and their order, the order
  of plugins, subsystems, systems and events, threading and locking, error
  variants and messages, the `.dirkasset` format, CLI flags, help and output,
  log fields, and the editor's UI, in every feature set and on each backend.
  Don't use blanket search and replace.
- **Report bugs instead of fixing them.** The orchestrator opens an issue and a
  fix PR for each. If a cleanup can't be done without changing behavior, report
  it too. Report a vulnerability only in your final message, never in a PR,
  commit or code comment.
- **Stay in your area.** Edit another area only when a rename forces it: keep
  it minimal and list those files in the PR description. Report what crosses
  areas: another lane cleans up between them. Never edit a file your task lists
  as held by a PR outside the pass, nor make a change that would force such an
  edit: report it instead.
- **Keep the docs right.** Fix stale doc comments in your files. If you rename
  something a README, an `AGENTS.md` or a page in `docs/` names, update it.
- Stop and report anything that needs a design decision.

## Checks

Export your task's environment for every cargo command. Before you open each PR,
run `just validate`, then what it and CI don't check:

- If you touched `apps/dirkengine` or the root `Cargo.toml`, the mixed
  feature sets: `cargo clippy --locked -p dirkengine --all-targets
  --no-default-features --features <f> -- -D warnings`, with `editor` and
  with `cli`.
- The environment sets `DIRK_REQUIRE_GPU=1`, so a native test fails instead
  of skipping when it can't create a device. Report the tests that still skip
  for a missing capability: they print `skipping:`, which
  `just test --workspace --success-output final` shows.
- If you changed a crate with macOS-only code, or something such code uses,
  run `cargo clippy --locked --target aarch64-apple-darwin -p <crate>
  --all-features -- -D warnings` on the crate with the macOS-only code.
  Draft CI skips macOS. If that can't build without an Apple host, CI checks
  the crate once the PR is ready.
- If you changed Metal code, or code that needs a window and no test covers,
  say in the PR what to run before merging: for Metal,
  `just test -p dirk_rhi` and `just run` on a Mac, with `DIRK_REQUIRE_GPU=1`
  and Metal's validation (`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`); for
  window code, `just run`, or `cargo run -p dirk_rhi --example present` for
  presentation.

Prove each "unchanged" claim against the PR's base, such as `--help` output
or the rustdoc of a public item. Before you call a failing test or check
unrelated, check that it fails on `main` too, and never change a test just to
make it pass.

## Delivery

- Split your work into focused PRs: one kind of cleanup in one part of your
  area each, small enough to review in one sitting. Branch each from a
  freshly fetched `origin/main` as `<prefix><topic>`, and keep your PRs from
  sharing files. If one needs another, stack it: branch from the other PR's
  branch, open it with that branch as its base, and say so in its
  description.
- Make small, logical commits with conventional messages and no
  `Co-Authored-By` line.
- Rebase on `origin/main` (a stacked PR on its base branch), and clean up any
  new code the rebase brings in. PRs merge by squash, so when a stacked PR's
  base merges, move only its own commits with
  `git rebase --onto origin/main <old base tip>`, and check it targets
  `main`.
- Open each PR as a **draft** with `file-pr`, despite its default, titled
  `refactor(<scope>): <what got simpler>`. The scope is the crate's name
  without `dirk_`, or `dirkengine` for `apps/dirkengine`. Tooling PRs are `ci:`,
  `build:` or `chore:`, and docs PRs `docs(<scope>):` or `docs:`. The
  description fills in `.github/pull_request_template.md` and says:
  - `Part of #<n>`
  - what was hard to read and how you fixed it
  - every test you merged or removed
  - files edited outside your area, and why
  - what you compared to prove the rest unchanged
- Don't mark a PR ready.

Your final message gives:

1. The PR links, in merge order, with what each one cleans up.
2. The checks you ran for each, with their results.
3. Bugs: `file:line`, what goes wrong and how to reproduce it.
4. Issues that cross areas, and other design issues: `file:line`, what's
   wrong, why it matters and a fix.

Check every item against the source before you report it. When you get review
findings, fix them on that PR's branch, rerun the affected checks, push, and
report what changed.
