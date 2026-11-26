# Cleanup pass: area agent

You own one area of the repository in a parallel cleanup pass. Your task
names the area, paths, packages, branch prefix, title scope, the pass's issue
and `CARGO_BUILD_JOBS`, and lists what the other agents own.

Read every file in your area, `AGENTS.md`, and the `standard` skill. Answer
the judgement checks of `checks.md`, beside this brief, that your task
gives you. Make real simplifications, not cosmetic churn, and report bigger
issues. When a rule of the standard gets in the way, follow the `standard`
skill: never work around it.

## Rules

- **Change no behaviour.** That includes HTTP responses (and which error a
  request with several faults gets), wire formats, rendered pages and
  emails, CLI flags, help, output and exit codes, configuration, events and
  audit records, log and span fields, metrics, timeouts, lock order, query
  plans, and the public API of published crates and packages. `git diff
  <base>` shows no change to a migration, a snapshot or a generated file,
  except one regenerated from what you changed on purpose. Don't use blanket
  search and replace.
- **Report bugs instead of fixing them.** The orchestrator opens an issue
  and a fix PR for each. If a cleanup can't be done without changing
  behaviour, report it too. Report a vulnerability only in your final
  message, never in a PR, commit or code comment.
- **Stay in your area.** Edit another area only when a rename forces it:
  keep it minimal and list those files in the PR description. Report what
  crosses areas: another lane cleans up between them.
- **Keep the docs references right.** If you rename something a page in
  `docs/` names, update that page (`docs` skill). Don't reword `///`
  comments or CLI help that feed generated references.
- Stop and report anything that needs a design decision.

## Checks

Export `CARGO_BUILD_JOBS` for every cargo command. Run `just check-changed
<base>`, with your PR's base: it runs CI's checks for only what you changed.
It leaves slow jobs to CI and prints their commands: run those yourself when
`AGENTS.md` says CI skips them on drafts, and otherwise read their results on
the PR before you report. The Dylint lints run only in CI: read their result
too. Don't run `just validate`: it builds the whole
workspace.

Prove each "unchanged" claim against the PR's base: `--help` output,
rendered pages and HTTP responses, the public API (`cargo semver-checks` for
a published crate), query plans. Before you call a failing check unrelated,
check that it fails on `main` too. Rerun a known flaky test once, and never
change a test just to make it pass.

## Delivery

- Split your work into focused PRs: one kind of cleanup in one part of your
  area each, small enough to review in one sitting. Branch each from a
  freshly fetched `origin/main` as `<prefix><topic>`, and keep your PRs from
  sharing files. If one needs another, stack it: branch from the other PR's
  branch, open it with that branch as its base, and say so in its
  description.
- Make small, logical commits with conventional messages and no
  `Co-Authored-By` line.
- Rebase on `origin/main` (a stacked PR on its base branch), and clean up
  any new code the rebase brings in. PRs merge by squash, so when a stacked
  PR's base merges, move only its own commits with
  `git rebase --onto origin/main <old base tip>`, and check it targets
  `main`.
- Open each PR as a **draft** with `file-pr`, titled
  `refactor(<scope>): <what got simpler>` (or `refactor: …` without a
  scope). Its description says:
  - `Part of <n>`
  - what was hard to read and how you fixed it
  - every test you merged or removed
  - files edited outside your area, and why
  - what you compared to prove the rest unchanged
- Don't mark a PR ready, and never add the `approved-exception` label:
  report each exception your PR adds, and why, for the owner to approve.

Your final message gives:

1. The PR links, in merge order, with what each one cleans up.
2. The checks you ran for each, with their results.
3. Bugs: `file:line`, what goes wrong and how to reproduce it.
4. The judgement checks: what you fixed, and what you left, each with its
   `file:line` and the change you propose.
5. The standard's friction: each rule you found worked around, or that got
   in your way, with the place and why.
6. Issues that cross areas, and other design issues: `file:line`, what's
   wrong, why it matters and a fix.

Check every item against the source before you report it. When you get
review findings, fix them on that PR's branch, rerun the affected checks,
push, and report what changed.
