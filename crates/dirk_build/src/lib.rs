#![doc = include_str!("../README.md")]

use std::{collections::HashMap, path::PathBuf};

/// Returns the directory of the current cargo workspace
///
/// # Panics
///
/// Will panic if it fails to run `cargo metadata`
#[must_use]
pub fn workspace_dir() -> PathBuf {
    let metadata = cargo_metadata::MetadataCommand::new()
        .exec()
        .expect("failed to run cargo metadata");

    metadata.workspace_root.into_std_path_buf()
}

/// Will add usefull relative paths to the compilation env.
///
/// Paths can be added in the function body inserting them into the [`HashMap`]
pub fn setup_paths() {
    println!(
        "cargo:rustc-env=WORKSPACE_ROOT={}",
        workspace_dir().display()
    );

    let workspace_root = PathBuf::from(".");

    let mut paths = HashMap::new();
    paths.insert("SAVED_PATH", "saved");
    paths.insert("ASSETS_PATH", "assets");

    for (name, path) in paths {
        println!(
            "cargo:rustc-env={name}={}",
            workspace_root.join(path).display()
        );
    }
}
