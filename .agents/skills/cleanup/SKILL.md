---
name: cleanup
description: >
  Run the monthly cleanup of the whole repository: an audit that measures the
  code, checks the standard's judgement rules and finds the rules that cause
  the most friction, then a pass where one agent per area opens focused,
  reviewed refactor PRs, a boundary lane cleans up between the areas, and
  every bug found gets an issue and a fix PR. Use when asked for a cleanup or
  refactor pass, an audit, a conformance check or codebase stats, or to
  deduplicate or simplify the repository as a whole. Not for cleaning up a
  single PR or file.
---

<!-- Synced from the sourceorigin-standard crate by `cargo xtask standard skills`: change it there, not here. -->

# Cleanup

You orchestrate a cleanup of the whole repository in two parts: an audit,
then a cleanup pass. Run both unless the owner asks for one. Delegates write
the code. You split the work, brief them, route reviews and report.

The repository's own facts come from its files: `AGENTS.md` names the issue
tracker, the surfaces, the hazards, the files nobody edits and how to
validate a change; `standard.toml` and `budget.toml` how it applies the
standard; `cargo metadata` its crates.

Work from a checkout of `origin/main` that contains this skill, since the
delegates read their briefs from it. Keep a ledger in your scratchpad: each
agent, branch, PR, review state and finding, and the files each task holds.
Files are free once the task holding them has reported and its PRs touching
them have merged. Start no task on files that aren't free. A forced edit
outside an area (a renamed item's callers, a docs reference) is the
exception: the agent holding those files rebases over it. After compaction, a
usage limit or a T3 restart, check the ledger and restart only what didn't
finish.

## Audit

1. Measure: `python3 -I .agents/skills/cleanup/scripts/stats.py .
   --json <scratchpad>/stats.json`. It needs `cloc`. Chart the result with
   `html_preview` and `html_render`, including growth over time from the
   JSON.
2. Measure the standard's friction: `cargo xtask standard report >
   <scratchpad>/report.json`. Rank the rules by their deviations (a
   whole-repository one above one scoped to `paths` or `crates`) and
   violations, and the lints by how many `#[expect]` attributes name them and
   whether `[workspace.lints]` allows them. Read the deviations' reasons and
   a sample of the `#[expect]` reasons. Each rule or lint near the top is a
   candidate: change it, delete it, narrow its deviation, or keep it and fix
   the code.
3. Read the output, then check what the scripts can't count:
   - coverage, as `AGENTS.md` says to measure it
   - unused code and dependencies, with `just unused`
   - generated files missing from `.gitattributes`
   - which `expect` calls can really panic
   - duplication built up since the last pass
   - the judgement checks in `references/checks.md`, sampled across the
     repository, to rank where the pass should look first
4. Send the owner the charts, the friction ranking and a ranked list of
   suggestions, each with the numbers and paths behind it. Open nothing yet.
5. For each suggestion the owner picks, start a thread with
   `t3_thread_launch`. Use Claude Opus 5.5 in a worktree off `origin/main`.
   Its message stands alone: the goal, the numbers, the scope and what to
   report. A thread that checks many sites reports a table of each site, its
   verdict and why. A rule the owner wants changed or deleted gets an issue
   in sourceorigin-dev/standard instead, since the rule lives
   there.

Changes that move code in many areas, such as file splits, merge before the
cleanup pass starts or go into the area agents' tasks.

## Cleanup pass

### Split

Split the repository into areas: one per workspace member (`cargo metadata`)
or top-level directory, as `stats.py` groups them. Split an area over about
30,000 lines of source and tests along a module boundary (`stats.py` lists
the modules of large areas), and merge small areas. A last area, tooling,
holds every other hand-written file: CI, root files, scripts. Every file has
exactly one owner. Each area takes the Conventional Commit scope its crate
uses, or none (`refactor: …`).

Nobody edits generated files (`.gitattributes` marks them), migrations or
the other files `AGENTS.md` says never to change. `docs/` and Markdown belong
to the docs lane.

Before you start the agents, run `just validate` on `origin/main`. If it
fails, as it can under a new stable Clippy, have one Opus delegate fix it in
a PR, review and ship it as in Review, and ask the owner to merge it first.

Open one issue for the pass, where `AGENTS.md` says issues live, and note its
id `<n>`. Each PR says `Part of <n>`. Each area agent opens several focused
PRs, on branches named `refactor/cleanup-<YYYY-MM>-<area>-<topic>`. Tell the
owner the split in one message and start without waiting.

Every worktree compiles the workspace in its own target directory, on a
machine with no swap. Pick one `CARGO_BUILD_JOBS` value for every agent that
compiles, and pass it to each. 4 suits 8 implementers and their reviewers on
16 cores and 58 GB. Run at most 8 reviews at a time.

### Start the agents

Call `orchestrator_capabilities` once. For each area, start a `delegate_task`
on Claude Opus 5.5 with `isolation: "worktree"` and `role: "implementation"`.
The task says to follow `.agents/skills/cleanup/references/agent-brief.md`.
It names the area, paths, packages, branch prefix, title scope, the pass's
issue, what the other areas own, and `CARGO_BUILD_JOBS`.

Start a docs lane the same way, on `docs/cleanup-<YYYY-MM>`. It follows the
`docs` skill, answers the docs and skills checks of `references/checks.md`,
fixes drift between the docs and the code, runs the repository's docs check
and opens a draft PR. It reports stale `///` comments and CLI help instead of
editing them: those sit in the area agents' files and may feed generated
references.

### Review

Review each focused PR on its own as its agent reports it. Call
`link_pull_request` and start a review: a `delegate_task` on GPT 6.1 Sol with
`role: "review"`, `isolation: "worktree"` and
`target.options: { reasoningEffort: "high" }`. It follows
`.agents/skills/cleanup/references/review-brief.md`. Give it the PR number,
the head SHA, the base branch (the parent PR's branch for a stacked PR),
`CARGO_BUILD_JOBS`, and a focus list: every "unchanged" claim in the PR
description, and every merged helper or test that could have changed
behaviour.

- Check each finding against the code before acting on it.
- Send real findings to the implementer's thread with `t3_thread_send`.
- After a fix that rewrites code, run a new, narrower review.
- Judgement calls go to the owner, with your recommendation.
- If Sol is over its limit, review with Opus instead and say so.

When a PR's review is clean and it is rebased, mark it ready. A PR that adds
an exception waits for the owner's `approved-exception` label: ask for it,
and never add it yourself or have a delegate add it. Follow its two
required checks, `ci` and `standard / pr-checks`, with `babysit-pr`, and
update its `[!NOTE]` line to name who reviewed it. The owner merges.

When something lands on `main`, have each agent rebase and apply its cleanup
to the new code, and start the boundary themes and fixes whose files are now
free. Review any logic the rebase merged. Suggest merging the crates others
depend on first.

### Boundary lane

The area agents each see one area. Alongside them, start one more Opus
delegate, `role: "research"`, to look at how the areas fit together, against
the architecture pages of the developer guide:

- the dependency direction between the crates, and logic in the wrong crate
- public items no other crate needs, and the published crates' public API
- types, conversions and helpers duplicated across crates or languages
- what each surface in `AGENTS.md` reimplements instead of sharing
- errors that lose their context at a crate boundary
- the surfaces and duplication checks of `references/checks.md`, across areas

It edits nothing. It reports themes, each with the files involved and the
change it proposes. Add the cross-area issues the area agents report to that
list. Themes that need a design decision, or change a published crate's API,
go to the owner first.

Once a theme's files are free, start one Opus implementation delegate per
theme. Merge themes that share files, or run them one after another. Each
follows the agent brief, with the theme's files as its area, and opens
focused PRs on `refactor/cleanup-<YYYY-MM>-boundary-<topic>`. Review them
like the others.

### Bugs

Every bug an agent or reviewer reports gets an issue and a fix PR:

1. Check it against current `main`, with a Sol delegate when there are many.
   Merge duplicate reports, and ask the owner about anything that might be
   intended. A vulnerability, anything an attacker could exploit, stops
   here: tell the owner only, and open no issue or PR (`SECURITY.md`).
2. Open an issue of type bug where `AGENTS.md` says issues live, or reuse
   the one that exists. Include the `file:line`, what goes wrong and how to
   reproduce it. If the existing issue is assigned or has a linked PR, note
   it in the ledger and start no fix.
3. Once its files are free, start an Opus implementation delegate in a
   worktree, on `fix/<topic>` from a freshly fetched `origin/main`, with the
   issue and `CARGO_BUILD_JOBS`. It writes a test that fails without the
   fix, makes the smallest fix, updates the docs it affects, runs the agent
   brief's checks, and opens a draft PR with `file-pr` that says
   `Fixes <issue>`.
4. Review and ship it as in Review, but without the review brief, which
   treats any behaviour change as a finding. The reviewer's task says to
   work at the head SHA without pushing or commenting, revert the fix
   locally to check the test then fails, and check the fix is correct and
   minimal and nothing else changed. It reports findings with severity and
   `file:line`.

Run at most 3 fixes at a time.

### Report

When every area and docs PR is reviewed and green, send the owner one
report:

1. The PRs by area: link, lines added and removed, merge order, and the PRs
   that are stacked.
2. Decisions only the owner can make, with your recommendation.
3. The bugs: each one's issue, fix PR and state.
4. The boundary themes, and which ones wait for the area PRs.
5. The standard's friction: the rules and lints to change or delete, the
   workarounds the agents found, and the issues opened in
   sourceorigin-standard.
6. Other design issues, and judgement checks the pass left open.
7. Features missing from a surface in `AGENTS.md`, checked against current
   `main`.

Number the items `NEW-1`, `NEW-2` and so on, never `#1`, which GitHub links
to an issue or PR.

### After the merge

Rerun the stats and the report, and compare them with the audit. Report
again when the boundary and fix PRs are reviewed and green. Then offer these
follow-ups, and start each one only when the owner asks:

1. A doc-comment PR for what the docs lane reported. It changes generated
   references on purpose.
2. Issues for the surface gaps.
