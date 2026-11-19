---
name: dirk-cleanup
description: >
  Run a repository-wide cleanup of DirkEngine: an audit that measures the code
  and suggests fixes, then a cleanup pass where one agent per area opens
  focused, reviewed refactor PRs, a boundary lane cleans up between crates, a
  docs lane checks the docs against the code, and every bug found gets a
  GitHub issue and a fix PR. Use when asked for a cleanup or refactor pass,
  codebase stats or an audit, or to deduplicate or simplify DirkEngine as a
  whole. Not for cleaning up a single PR or file.
---

# DirkEngine cleanup

You orchestrate a cleanup of the whole repository in two parts: an audit, then
a cleanup pass. Run both unless the owner asks for one. Delegates write the
code. You split the work, brief them, route reviews and report.

Work from a checkout of `origin/main` that contains this skill, since the
delegates read their briefs from it. Keep a ledger in your scratchpad: each
agent, branch, PR, review state and finding, and the files each task holds.
Files are free once the task holding them has reported and its PRs touching
them have merged or closed. Start no task on files that aren't free. A forced
edit in another lane's files (a renamed item's callers, a docs reference) is
the exception: that lane rebases over it. After compaction, a usage limit or a
T3 restart, check the ledger and restart only what didn't finish.

## Audit

1. Measure: `python3 -I .agents/skills/dirk-cleanup/scripts/stats.py .
   --json <scratchpad>/stats.json`. It needs `cloc`. Chart the result with
   `html_preview` and `html_render`, including growth over time from the
   JSON.
2. Read the output, then check what the script can't count:
   - coverage, with `cargo llvm-cov nextest --locked --workspace --all-features`
   - whether each `#[expect]` reason still holds, and which `expect` calls
     can really panic
   - the deviations `docs/dev/rust-conventions.md` lists, and code that
     breaks its layering or its rules for errors and unsafe code
   - duplication built up since the last pass
3. Send the owner the charts and a ranked list of suggestions, each with the
   numbers and paths behind it. Open nothing yet.
4. For each suggestion the owner picks, start a thread with
   `t3_thread_launch`. Use Claude Opus 5.5 in a worktree off `origin/main`.
   Its message stands alone: the goal, the numbers, the scope and what to
   report. A thread that checks many sites reports a table of each site, its
   verdict and why.

Changes that move code in many areas, such as file splits, merge before the
cleanup pass starts or go into the area agents' tasks.

## Cleanup pass

### Split

Start from this split and adjust it to the code. Split an area over about
10,000 lines of source and test code (`stats.py` by crate) along a module
boundary, and merge small areas. Every file has exactly one owner.

| Area | Paths |
| --- | --- |
| Renderer | the rendering crates, today `crates/dirk_renderer` (with its `shaders`) and `crates/dirk_shaders` |
| Core | `crates/dirk_engine`, `crates/dirk_universe`, `crates/dirk_events`, `crates/dirk_threads`, `crates/dirk_utils`, `crates/dirk_proc`, `crates/dirk_build` |
| Features and tooling | the other crates, `src`, `build.rs`, `.cargo`, `.config`, `.github` and the root files |

Nobody edits `assets` or `.agents`. `docs/`, every README and the `AGENTS.md`
files belong to the docs lane. Each area agent opens several focused PRs, on
branches named `refactor/cleanup-<YYYY-MM-DD>-<area>-<topic>`.

PRs opened outside the pass hold their files too, with no forced-edit
exception, until they merge or close or the owner releases them. List them
all with `gh pr list --limit 200 --json number,title,files`.

Builds run in parallel on a machine with no swap. Pick one
`CARGO_BUILD_JOBS` value for every agent that compiles, such as 4, and pass
it to each. Run `just build` here first, so every worktree finds rust-gpu's
backend cached in `~/.cache/rust-gpu`. Run at most 4 reviews at a time.

Open one GitHub issue for the pass, titled as `AGENTS.md` describes, such as
`refactor: cleanup pass, November 2026`. Each cleanup PR says `Part of #<n>`.
Tell the owner the split in one message, with the files held by PRs outside
the pass in each area. Start an area no such PR touches without waiting. For
the rest, ask the owner whether to start now, leaving the held files alone, or
after those PRs merge. A held file that frees up goes to its area's agent if
that agent hasn't reported; otherwise its cleanup waits for the next pass.

### Start the agents

Call `orchestrator_capabilities` once. For each area, start a `delegate_task`
on Claude Opus 5.5 with `isolation: "worktree"` and `role: "implementation"`.
The task says to follow `.agents/skills/dirk-cleanup/references/agent-brief.md`.
It names the area, paths, crates, branch prefix, issue, what the other areas
own, the files held by PRs outside the pass, and `CARGO_BUILD_JOBS`.

Start a docs lane the same way, with the docs lane's files as its area and
`docs/cleanup-<YYYY-MM-DD>-` as its branch prefix. It fixes drift between the
docs and the code, and reports stale doc comments: send each to the agent that
owns it.

### Review

Review each focused PR on its own as its agent reports it. Call
`link_pull_request` and start a review: a `delegate_task` on GPT 6.1 Sol with
`role: "review"`, `isolation: "worktree"` and
`target.options: { reasoningEffort: "high" }`.
It follows `.agents/skills/dirk-cleanup/references/review-brief.md`.
Give it the PR number, the head SHA, the base branch (the parent PR's branch
for a stacked PR), `CARGO_BUILD_JOBS`, and a focus list:
every "unchanged" claim in the PR description, every merged helper or
test that could have changed behavior, and every moved `unsafe` block.

- Check each finding against the code before acting on it.
- Send real findings to the implementer's thread with `t3_thread_send`.
- After a fix that rewrites code, run a new, narrower review.
- Judgment calls go to the owner, with your recommendation.
- If Sol is over its limit, review with Opus instead and say so.

When a PR's review is clean and it is rebased, mark it ready, which also runs
CI on macOS. Follow its CI with `babysit-pr`, sending fixes to its agent, and
update its `[!NOTE]` line to name who reviewed it. The owner merges.

When something lands on `main`, have each agent rebase and apply its cleanup
to the new code, and start the boundary themes and fixes whose files are now
free. Review any logic the rebase merged. Suggest merging from the bottom of
the crate layering up (`docs/dev/rust-conventions.md`).

### Boundary lane

The area agents each see one area. Alongside them, start one more Opus
delegate, `role: "research"`, to look at how the crates fit together:

- the dependency direction against the layering in
  `docs/dev/rust-conventions.md`, and logic that sits in the wrong crate
- public items that no other crate needs, and internals that leak across a
  boundary
- types, conversions and helpers duplicated across crates
- features that bypass plugins and subsystems, and `editor` gating spread
  across crates
- errors that lose their context at a crate boundary

It edits nothing. It reports themes, each with the files involved and the
change it proposes. Add the cross-area issues the area agents report to that
list. Themes that need a design decision go to the owner first.

Once a theme's files are free, start one Opus implementation delegate per
theme. Merge themes that share files, or run them one after another. Each
follows the agent brief, with the theme's files as its area, and opens
focused PRs on `refactor/cleanup-<YYYY-MM-DD>-boundary-<topic>`. Review them
like the others.

### Bugs

Each bug an agent or reviewer reports goes through these steps:

1. Check it against current `main`, with a Sol delegate when there are many.
   Merge duplicate reports, and ask the owner about anything that might be
   intended. A vulnerability stops here: tell the owner only, and open no
   issue or PR. The owner can open a private security advisory.
2. Open a GitHub issue with the `bug` label, titled as `AGENTS.md` describes,
   or reuse the one that exists (`gh issue list --search`). Include the
   `file:line`, what goes wrong and how to reproduce it. If the existing issue
   is assigned or has a linked PR, note it in the ledger and start no fix.
3. Once its files are free, start an Opus implementation delegate in a worktree,
   on `fix/<topic>` from a freshly fetched `origin/main`, with the issue and
   `CARGO_BUILD_JOBS`. It writes a test, or names the check, that fails
   without the fix, makes the smallest fix, runs the agent brief's checks, and
   opens a draft PR with `file-pr` that says `Fixes #<n>`.
4. Review and ship it as in Review, but without the review brief, which treats
   any behavior change as a finding. The reviewer's task says to work at the
   head SHA without pushing or commenting, revert the fix locally to check the
   test or check then fails, and check the fix is correct and minimal and
   nothing else changed. It reports findings with severity and `file:line`.

Run at most 3 fixes at a time.

### Report

When every area and docs PR is reviewed and green, send the owner one
report:

1. The PRs by area: link, lines added and removed, merge order, and the PRs
   that are stacked.
2. The PRs whose description asks for a `just run` before merging.
3. Decisions only the owner can make, with your recommendation.
4. The bugs: each one's issue, fix PR and state.
5. The boundary themes, and which ones wait for the area PRs.
6. Other design issues.

Number the items `NEW-1`, `NEW-2` and so on, never `#1`, which GitHub links to
an issue or PR.

### After the merge

Rerun the stats and compare them with the audit. Report again when the
boundary and fix PRs are reviewed and green, and close the pass's issue once
they merge.
