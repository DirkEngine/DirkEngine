---
name: standard
description: >
  Work with the engineering standard: the rules `just standard` checks, the
  Dylint lints CI runs, and the pull-request checks. Read a violation and fix
  it, make the mechanical fixes with `--fix`, and know what to do when a rule
  gets in the way. Use when `just standard`, a Dylint lint or a PR check
  fails, before adding a deviation, an `#[expect]`, a workspace `allow`, an
  `// INVARIANT:` or a `// wall-clock:` comment, or when adding or changing a
  rule in sourceorigin-standard.
---

<!-- Synced from the sourceorigin-standard crate by `cargo xtask standard skills`: change it there, not here. -->

# Work with the standard

The standard is the `sourceorigin-standard` crate. Its README lists every
rule, and so does `cargo xtask standard rules`. It reaches a repository in
three ways:

- **Rules:** `just standard` runs them (`cargo xtask standard`), and so does
  CI, on every change.
- **Dylint lints:** the rules that need types, in one library,
  `standard_lints`. CI runs them, and a warning fails the build.
- **PR checks:** the `standard / pr-checks` workflow checks the pull request
  itself: its title and description, no `Co-authored-by` trailers, a test
  with each fix, a Dependencies section for a new dependency, and approval
  for each exception it adds.

A repository has two required checks: `ci`, for the code, and
`standard / pr-checks`, for the pull request. `standard.toml` at its root
holds how it applies the rules, and its `[pr-checks]` table the PR checks'
settings, such as the scopes a title may take. `budget.toml` holds each
crate's exact number of `#[expect]` lints, `// INVARIANT:` lines and
`// wall-clock:` lines, and each manifest's number of lints set to `allow`.

## Fix a violation

A violation reads:

```text
crates/core/src/lib.rs:12: [rust-comments] <what is wrong, and how to fix it>
  If the rule is wrong here, ask the user, then add `rust-comments = { reason = "<why>", paths = ["crates/core/src/lib.rs"] }` …
```

1. Run `cargo xtask standard --fix` first. It makes the fixes that need no
   judgement and prints each one: `[lints] workspace = true`, a workspace
   lint at `deny` or `forbid` lowered to `warn`, a budget lowered to what a
   crate holds, a stale deviation removed, a synced skill rewritten. It never
   turns an `allow` into `warn`. Read its diff.
2. Fix each remaining violation as its message says, in the code it names.
3. Run `just standard` again until it passes.

A Dylint lint reports like a Clippy lint, with its fix in its message.

## When a rule gets in the way

Never work around a rule. A workaround passes the check and breaks what the
rule is for:

- splitting a function or a doc comment only to stay under a cap;
- moving prose into a string literal, or a comment into a name;
- `let _ =`, a dummy use or a rename only to silence a lint;
- an `#[expect]`, a workspace `allow` or a raised budget in `budget.toml`
  without asking.

Stop instead, and ask the user. Name the rule, the place, and why the rule is
wrong there. Offer the narrowest way out first:

1. **A deviation for the one file or crate:** under `[deviations]` in
   `standard.toml`, `<rule> = { reason = "<why>", paths = ["<file or
   glob>"] }`, or `crates = ["<crate>"]`. The violation's message prints this
   form for its file.
2. **A deviation for the whole repository:** `<rule> = "<why>"`, only when
   the rule is wrong everywhere in it.
3. **A change to the rule for every repository:** a pull request to
   sourceorigin-standard, which each repository picks up with the new
   version.

A deviation, or an entry of its scope, that excuses no violation fails as
stale, and `--fix` removes it.

A lint has its own exceptions, each on the smallest item that needs it, with
a reason saying why the lint is wrong there:

- a Clippy or rustc lint: `#[expect(<lint>, reason = "<why>")]`;
- a Dylint lint: `#[cfg_attr(dylint_lib = "standard_lints",
  expect(<lint>, reason = "<why>"))]`;
- a lint outside the set the standard requires: `allow` in
  `[workspace.lints]`, the one reviewed place for a lint the repository
  doesn't want at all.

With the user's approval, make the exception or deviation in the same pull
request as the code that needs it, and say why in its description. A new
`#[expect]`, `// INVARIANT:`, `// wall-clock:` or workspace `allow` raises
its number in `budget.toml` (`[allow]` counts a manifest's by its path): the
same question.

## Approval of an exception

The PR checks' `lint-exceptions` check fails a pull request that loosens a
rule: any change to `standard.toml`, `clippy.toml`, `deny.toml`,
`dylint.toml` or `.cargo/config.toml`, or a budget that rises, as one does for a lint newly
`allow`ed. A change to `standard.toml` that only removes deviations, removes
entries from their `paths` or `crates`, or rewords a reason passes without
approval. It passes once one of the `approvers` in `[pr-checks]` adds the
`approved-exception` label after the latest push; a later push, a
force-push included, needs it added again.

**Never add the `approved-exception` label yourself**, nor ask another agent
to. You act with the owner's GitHub token, so the check can't tell your
label from theirs: adding it forges their approval. Ask the user instead:
name each exception the job's summary lists, and why it is needed, and leave
the label to them.

## The synced skills

The `standard`, `cleanup` and `docs` skills in `.agents/skills/` are copies
of the crate's. Don't edit them here: change them in sourceorigin-standard.
`cargo xtask standard skills` rewrites them from the crate, and the
`skills-synced` rule fails while a copy differs.

## Add or change a rule

Rules live in sourceorigin-standard. A rule goes to the first of these that
can express it: the types, rustc and Clippy settings, a rule in the crate, a
Dylint lint when it needs types, a PR check, the `cleanup` skill.

1. Add a variant to `rules!` in `crates/sourceorigin-standard/src/rules/mod.rs`.
   Its doc comment is the rule's text: what it reads and what fails. Its
   string is its name.
2. Write its check as a method on `Repository`, in a module of its own under
   `rules/`. It returns a `Finding` for each violation, whose message says
   how to fix it, and, when the fix needs no judgement, a `Fix` that
   `--fix` makes. Load what it reads in `Repository::load`.
3. Test it on a `Fixture` in `rules/tests/<rule>.rs`.
4. Run `just readme`, which writes the rule's text into the crate's README.
5. In the pull request, say what each repository must change when it moves
   to the new version.
