# Agent Instructions

This repository is a Rust workspace. Keep changes focused, follow the existing
crate style.

## Validation

Before handing work back, run the same validation steps consistently:

```sh
cargo fmt --all
cargo clippy --workspace
cargo nextest run --workspace
```

If working on just one specific crate, replace `--workspace` with `-p <crate>`.

If a validation step cannot be run because a required tool or system dependency
is unavailable, report the exact command that failed and the reason.

If working on specific cargo features, run with `--no-default-features`
& `--all-features`.

## Pull requests

Open pull requests as drafts while iterating: CI then skips macOS, its slowest
runner. Mark them ready for review to run the full matrix.
