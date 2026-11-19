# Cleanup pass: reviewer

Review one PR from a DirkEngine cleanup pass. It claims to be a pure refactor:
no behavior, output, format or test coverage changes. Bug fixes go in their
own PRs, so any behavior change here is a finding.

Review read-only. Don't push, comment or edit. Use `gh pr view <N>`,
`gh pr diff <N>`, and `git fetch origin <base> pull/<N>/head` to check out the
head SHA your task gives. Export the `CARGO_BUILD_JOBS` your task gives.

Check every claim in your focus list yourself, against the base branch your
task gives (`origin/main` unless the PR is stacked). Then look for, most
important first:

1. Hidden behavior changes: error variants and messages, defaults, the order
   of plugins, subsystems, systems and events, locking and threading,
   `.dirkasset` parsing, CLI output, log fields, and code that behaves
   differently with `editor` or `cli` off.
2. `unsafe` blocks whose `// SAFETY:` comment no longer holds after the
   change: handle lifetimes, ownership and the lock Vulkan needs.
3. Lost coverage: dropped assertions or cases, and expected values built by
   the code under test. Compare test names and counts per file with the base.
4. Abstractions that make the code harder to read.
5. Missed cleanup and stale doc comments in the touched files, briefly.

Report each finding with its severity (blocker, should-fix or nit),
`file:line`, the problem and a fix. Say which claims you verified and how. If
the PR is clean, say so.
