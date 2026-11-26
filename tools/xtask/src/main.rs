#![doc = include_str!("../README.md")]

mod ci;
mod standard;

use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sourceorigin_standard::check_changed::DEFAULT_BASE;

/// The repository's root, which is the workspace's.
const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// `cargo xtask <task>`.
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

/// A development task.
#[derive(Subcommand)]
enum Task {
    /// Check the repository against the engineering standard's rules
    /// (`sourceorigin-standard`), or report on, list or sync them.
    Standard(standard::Standard),
    /// Run CI's checks for only what changed since the merge base with
    /// `base`, counting uncommitted and untracked files.
    CheckChanged {
        /// The branch to compare with.
        #[arg(default_value = DEFAULT_BASE)]
        base: String,
    },
}

fn main() -> Result<ExitCode> {
    let root = Path::new(ROOT);
    match Cli::parse().task {
        Task::Standard(standard) => standard.run(root),
        Task::CheckChanged { base } => ci::CI
            .run(root, &base)
            .with_context(|| format!("check what changed since the merge base with {base}")),
    }
}
