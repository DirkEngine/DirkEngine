//! `cargo xtask standard`: the engineering standard's rules, from the
//! `sourceorigin-standard` crate, whose README lists them. `standard.toml`
//! and `budget.toml` hold how this repository applies them.

use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use sourceorigin_standard::rules;

/// `cargo xtask standard [--fix | report | rules | skills]`.
#[derive(Args)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct Standard {
    /// First make the fixes that need no judgement, and print each.
    #[arg(long)]
    fix: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

/// What `cargo xtask standard` does instead of checking the rules.
#[derive(Clone, Copy, Subcommand)]
enum Command {
    /// Print each rule's violations and deviation, with the `#[expect]`s,
    /// `// INVARIANT:` lines and allowed lints, as JSON.
    Report,
    /// List the rules.
    Rules,
    /// Write the standard's shared skills into `.agents/skills`.
    Skills,
}

impl Standard {
    /// Checks the repository at `root` against the rules, printing each
    /// violation and failing when there is one, or runs the command.
    pub(crate) fn run(self, root: &Path) -> Result<ExitCode> {
        match self.command {
            Some(Command::Report) => {
                let report = rules::report(root).context("report on the standard's rules")?;
                println!("{}", report.to_json().context("serialize the report")?);
            }
            Some(Command::Rules) => print!("{}", rules::describe()),
            Some(Command::Skills) => {
                for fix in rules::sync_skills(root).context("sync the standard's skills")? {
                    println!("synced {fix}");
                }
            }
            None => {
                if self.fix {
                    for fix in rules::fix(root).context("fix the standard's rules")? {
                        println!("fixed {fix}");
                    }
                }
                let violations = rules::check(root).context("check the standard's rules")?;
                for violation in &violations {
                    eprintln!("{violation}");
                }
                if !violations.is_empty() {
                    eprintln!(
                        "{} violations of the standard; `cargo xtask standard rules` lists the rules",
                        violations.len()
                    );
                    return Ok(ExitCode::FAILURE);
                }
            }
        }
        Ok(ExitCode::SUCCESS)
    }
}
