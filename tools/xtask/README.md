# xtask

The workspace's development tasks, run with `cargo xtask`: the engineering
standard's rules (`standard`), and CI's quick checks for only what changed
(`check-changed`). The justfile runs both, as `just standard` and
`just check-changed`.

## `cargo xtask standard`

Checks the repository against the rules of the engineering standard, the
`sourceorigin-standard` crate, whose README lists them. It prints each
violation with its location, rule and fix, and fails when there is one.
`standard.toml` and `budget.toml`, at the root, hold how the repository applies
the rules: its settings, its deviations with their reasons, and each crate's
budget of exceptions. `just standard` runs it, and CI runs it in a job of its
own on every change.

- `cargo xtask standard --fix` first makes the fixes that need no judgement,
  such as lowering a budget to what its crate holds, and prints each.
- `cargo xtask standard report` prints each rule's violations and deviation,
  with the `#[expect]`s, `// INVARIANT:` lines and allowed lints, as JSON.
- `cargo xtask standard rules` lists the rules.
- `cargo xtask standard skills` writes the standard's shared skills into
  `.agents/skills`.

## `cargo xtask check-changed`

Runs CI's checks for only what changed since the merge base with a branch,
`origin/main` unless named (`cargo xtask check-changed <base>`), counting
uncommitted and untracked files. It matches the changed paths against the
filters of `.github/filters.yml`, runs the recipes of each CI job they start,
on the crates the change reaches when a recipe takes crates, leaves the slow
ones to CI, and prints a summary of every check. The `builds` edges of
`standard.toml`'s `[check-changed]` table add the crates whose build script
compiles a changed crate without depending on it, such as the renderer, which
compiles its shaders. `just check-changed` runs it, passing its arguments on.

`src/ci.rs` describes CI's jobs as a table, and its `ci_matches_the_workflow`
test fails until that table, `.github/workflows/ci.yml` and
`.github/filters.yml` change together.
