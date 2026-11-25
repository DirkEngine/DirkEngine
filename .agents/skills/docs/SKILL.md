---
name: docs
description: >
  Write or change documentation: a page of the user or developer guide, a
  README, a screenshot, or a doc comment, including those that feed `--help`
  and generated references. Use before writing or reviewing documentation,
  when a change needs its docs updated, or when asked to document a feature.
---

<!-- Synced from the sourceorigin-standard crate by `cargo xtask standard skills`: change it there, not here. -->

# Write the documentation

First read the repository's documentation page, which `AGENTS.md` names
(usually `docs/dev/documentation.md`). It holds what differs per repository:
where pages go, which pages are generated and from what, how screenshots are
taken, and the command that builds and checks the docs. This skill holds the
rest.

Check every claim against the code before you write it: the documentation
says what the code does now. Update it in the same pull request as the change
it describes.

## What each kind of documentation holds

- **User guide:** complete and detailed. It covers every feature a user can
  reach, through each of its surfaces, from what it is for to what goes wrong.
  It says what the reader can see and do, never how it is implemented.
- **Developer guide:** short recipes, one per thing a contributor adds: a
  setting, an endpoint, a page, a rule. Each names the files to create or
  change, the code that goes in each, and the command that checks it. Where
  `cargo xtask new <thing>` scaffolds it, the recipe is that command and what
  to fill in. An architecture page says how the parts fit together. Rules
  aren't prose here: the standard's checks hold them.
- **Doc comments:** what goes in, and what comes out or happens, including
  errors and panics. No justification, no history, no narration of the code
  below. A doc comment on a type users configure, or on a CLI argument, is
  user documentation: write it for them. There are no developer notes: what a
  developer must know, the code enforces.
- **READMEs:** what the directory or crate is and how to use it, linking to
  the guides rather than repeating them.

Don't document what explains itself elsewhere: a migration in its SQL
comments, a test in its name, a change in its commit.

## Writing

- Keep a page to one task or topic, and link to the reference rather than
  repeating it.
- Describe a feature through each of its surfaces, in the order `AGENTS.md`
  lists them. When a task can be done on more than one, show each way, and
  never present one as the only way. If a surface can't do it, say so.
- Describe how it works now, not how it changed.
- Write placeholders in prose as `` `<name>` `` or `\<name>`. A bare
  `<name>` is an HTML tag: it hides the text.

## Style

Write so that someone new to the project can follow the page from top to
bottom.

- **Start with the reader's goal.** Open each page with one or two sentences
  saying what it helps them do.
- **Keep sentences short, one idea per paragraph.** A sentence that needs a
  parenthesis and a semicolon is two sentences.
- **Show, then explain.** Put the command or code first, then say what it
  does and what the reader should see.
- **Number the steps of a procedure.** Give each step one action.
- **Say where each command runs** when it could run in more than one place.
- **Warn before a command that deletes data** or can't be undone, not after
  it.
- **Cover the main path first.** Limits, edge cases and failure modes go at
  the end of the page, under *Troubleshooting* or *Limits*, or in the
  reference.
- Address the reader as "you", use the present tense, and write the labels
  of buttons and fields in **bold**, exactly as they appear.
- Link a concept where it is first mentioned instead of explaining it again.

For example, rather than:

> `backup prune` deletes old snapshots; keeps `--keep` (default 7, `0`
> keeps none).

write:

> Delete all but the last seven snapshots:
>
> ```console
> mytool backup prune --keep 7
> ```
>
> The command prints each snapshot it deletes. Deleted snapshots can't be
> restored.

The user guide's feature guides share one outline:

```markdown
# <Feature>

What it does and when to use it, in a sentence or two.

## Before you start

What the reader needs first: permissions, other features, accounts.

## <Task>

Numbered steps, through each surface that can do it. Give each task its own
section, named after what the reader does, such as "Rotate a secret".

## Check that it works

## Troubleshooting
```

Reference pages list every setting instead, each with what it does and its
default.

## Images

- Show the interface where it helps the reader find their way, such as the
  first time a guide uses a page, or a step done through a form. The text
  still makes sense without the image.
- Screenshots of the project's own interface are generated, never taken by
  hand, so they match it. Retake them after changing a page they show, with
  the repository's screenshot command.
- Give each image alt text that says what it shows, for readers who can't
  see it:

  ```markdown
  ![The overview: one healthy application and its last deployment](../images/overview.png)
  ```

## Links

- Link to another page with its path relative to the current page. These
  links work both on GitHub and in the book.
- Link to a repository file outside `docs/` with its GitHub URL on `main`
  (`tree` instead of `blob` for a directory). A relative path would break in
  the book.
- A heading's anchor is its text in lower case, with spaces replaced by
  hyphens and punctuation removed. Renaming a heading breaks the links to
  it.

## Before you finish

Run the command that builds and checks the docs, and retake the screenshots
if you changed a page they show. Check the commands, labels and output you document
against the running app, with the skill that runs it.
