# Cleanup pass: area agent

You own one area of DirkEngine in a parallel cleanup pass. Your task names the
area, paths, crates, branch prefix, title scope, issue and `CARGO_BUILD_JOBS`,
and lists what the other agents own.

Read every file in your area, and `docs/dev/rust-conventions.md`. Make real
simplifications, not cosmetic churn, and bring the code you touch in line
with the conventions. Report bigger issues.

## Rules

- **Change no behavior.** That includes what renders, the order of plugins,
  subsystems, systems and events, threading and locking, error variants and
  messages, the `.dirkasset` format, CLI flags, help and output, log fields,
  and the editor's UI, in every feature set. Don't use blanket search and
  replace.
- **Report bugs instead of fixing them.** The orchestrator opens an issue and a
  fix PR for each. If a cleanup can't be done without changing behavior, report
  it too. Report a vulnerability only in your final message, never in a PR,
  commit or code comment.
- **Stay in your area.** Edit another area only when a rename forces it:
  keep it minimal and list those files in the PR description. Report what
  crosses areas: another lane cleans up between them.
- **Keep the docs right.** Fix stale doc comments in your files. If you rename
  something a README, an `AGENTS.md` or a page in `docs/` names, update it.
- Stop and report anything that needs a design decision.

## Checks

Export `CARGO_BUILD_JOBS` for every cargo command. Before you open each PR,
run `just validate`, then what it and CI don't check:

- If you touched `src`, the mixed feature sets: `cargo clippy --locked
  --all-targets --no-default-features --features <f> -- -D warnings`, with
  `editor` and with `cli`.
- If you touched code behind `cfg(validation)`, which release builds leave
  out, `cargo clippy --locked --release -p <crate> --all-features -- -D
  warnings`.
- If you touched `platform_macos` code, `cargo clippy --locked --target
  aarch64-apple-darwin -p <crate> --all-targets --all-features -- -D
  warnings`. Draft CI skips macOS.
- No test or CI job renders. If you changed how the renderer, the platform
  layer or the editor runs, say in the PR that it needs a `just run`.

Prove each "unchanged" claim against the PR's base, such as `--help` output
or the rustdoc of a public item. Before you call a failing test unrelated,
check that it fails on `main` too, and never change a test just to make it
pass.

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
  `refactor(<scope>): <what got simpler>`. Its description fills in
  `.github/pull_request_template.md` and says:
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
