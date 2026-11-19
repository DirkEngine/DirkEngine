# Agent Instructions

This repository is a Rust workspace. Keep changes focused, follow the existing
crate style.

## Validation

`just validate` (or `just` alone) formats the sources and runs every check CI
runs: Clippy with warnings denied on every target and feature set, the tests
with every feature and with none, the docs and doc tests, the dependency audit
and the unused dependency check. Run it before handing work back.

While iterating, run only what you changed, such as `just test -p <crate>`.

If a validation step cannot be run because a required tool or system dependency
is unavailable, report the exact command that failed and the reason.

## Pull requests

Open pull requests as drafts while iterating: CI then skips macOS, its slowest
runner. Mark them ready for review to run the full matrix.
