# DirkEngine's development tasks. `just` alone runs `validate`, the checks every
# change passes before review. CI runs the same recipes (.github/workflows/ci.yml).

default: validate

# Formats the sources, then runs every check CI runs.
validate: fmt lint test test-minimal doc doc-test audit unused

# Runs the engine from the workspace root, where it finds `assets/` and writes
# `saved/`; arguments go on to its CLI.
[positional-arguments]
run *ARGS:
    @cargo run --locked -p dirkengine -- "$@"

build:
    @cargo build --locked --workspace

fmt:
    @cargo fmt --all

# CI checks formatting without modifying sources; `validate` uses `fmt`.
fmt-check:
    @cargo fmt --all -- --check

# The no-default-features pass leaves out dirk_editor, which always enables the
# engine's editor support. The workspace passes unify features across crates,
# so a crate can compile there only because another crate enables a feature it
# needs: linting each crate alone catches that.
# Clippy with warnings denied: every target, both feature sets, each crate alone.
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo clippy --locked --workspace --exclude dirk_editor --all-targets --no-default-features -- -D warnings
    for manifest in apps/*/Cargo.toml crates/*/Cargo.toml; do
        cargo clippy --locked --manifest-path "$manifest" --all-targets -- -D warnings
    done

# Arguments replace `--workspace`, such as `just test -p dirk_events -E 'test(flush)'`.
# The tests with every feature; arguments go on to nextest.
[positional-arguments]
test *ARGS='--workspace':
    @cargo nextest run --locked --all-features "$@"

# The tests again with no default features.
test-minimal:
    @cargo nextest run --locked --workspace --exclude dirk_editor --no-default-features

# Rustdoc warnings, such as broken intra-doc links, fail the build.
doc:
    @RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps --all-features

doc-test:
    @cargo test --locked --workspace --doc --all-features

# Advisories, licenses, bans and sources, as deny.toml configures them, for
# every workspace member, including crates the binary doesn't depend on.
audit:
    @cargo deny --workspace check -D warnings

# Dependencies a crate declares but never names.
unused:
    @cargo machete
