//! The jobs of .github/workflows/ci.yml, as `just check-changed` runs them.
//! The `ci_matches_the_workflow` test fails until this table, ci.yml and
//! .github/filters.yml change together.

use sourceorigin_standard::check_changed::{self, Ci, Job, Recipe, Run, Trigger};

/// A filter of .github/filters.yml.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Filter {
    /// Everything but the docs and files nothing builds from.
    Rust,
    /// The manifests, the lockfile, `deny.toml` and the workflows.
    Deny,
}

impl check_changed::Filter for Filter {
    const ALL: &'static [Self] = &[Self::Rust, Self::Deny];

    fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Deny => "deny",
        }
    }
}

/// A recipe of the justfile, run here on the whole workspace.
const fn here(command: &'static [&'static str]) -> Recipe {
    Recipe {
        command,
        run: Run::Workspace,
    }
}

/// CI's jobs, in the workflow's order; `ci` only sums up the others.
pub(crate) const CI: Ci<Filter> = Ci {
    jobs: &[
        Job {
            id: "checks",
            trigger: Trigger::AnyOf(&[Filter::Rust, Filter::Deny]),
            recipes: &[
                here(&["just", "fmt-check"]),
                here(&["just", "unused"]),
                here(&["just", "audit"]),
            ],
        },
        Job {
            id: "rust",
            trigger: Trigger::AnyOf(&[Filter::Rust]),
            recipes: &[
                here(&["just", "lint"]),
                Recipe {
                    command: &["just", "test"],
                    run: Run::Crates {
                        libraries_only: false,
                    },
                },
                here(&["just", "test-minimal"]),
                here(&["just", "doc-test"]),
                here(&["just", "doc"]),
                Recipe {
                    command: &["just", "build"],
                    run: Run::InCi("bundles the debug build and its assets as an artifact"),
                },
            ],
        },
        Job {
            id: "standard",
            trigger: Trigger::EveryChange,
            recipes: &[here(&["just", "standard"])],
        },
    ],
    ignored: &["ci"],
};

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// Fails until the table, ci.yml and filters.yml agree.
    #[test]
    fn ci_matches_the_workflow() {
        super::CI
            .check_workflow(Path::new(crate::ROOT))
            .expect("the CI table matches ci.yml and filters.yml");
    }
}
