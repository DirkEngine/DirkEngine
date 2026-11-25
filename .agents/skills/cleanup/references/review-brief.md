# Cleanup pass: reviewer

Review one PR from a cleanup pass. It claims to be a pure refactor: no
behaviour, output, schema, format or test coverage changes. Bug fixes go in
their own PRs, so any behaviour change here is a finding.

Review read-only. Don't push, comment or edit. Use `gh pr view <N>`,
`gh pr diff <N>`, and `git fetch origin <base> pull/<N>/head` to check out
the head SHA your task gives. Export the `CARGO_BUILD_JOBS` your task gives.

Check every claim in your focus list yourself, against the base branch your
task gives (`origin/main` unless the PR is stacked). Then look for, most
important first:

1. Hidden behaviour changes: error codes, messages, statuses and precedence,
   defaults, serialization (tokens, events, audit records, wire types),
   rendered pages and emails, log and span fields, timeouts, lock order and
   transaction boundaries, query plans, and the published crates' public
   API.
2. Lost coverage: dropped assertions or cases, and expected values built by
   the code under test. Compare test names and counts per file with the
   base.
3. Workarounds of the standard: code contorted to pass a rule instead of
   following it (the `standard` skill lists them).
4. Abstractions that make the code harder to read.
5. Missed cleanup and stale doc comments in the touched files, briefly.

Report each finding with its severity (blocker, should-fix or nit),
`file:line`, the problem and a fix. Say which claims you verified and how.
If the PR is clean, say so.
