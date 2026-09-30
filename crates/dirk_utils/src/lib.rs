#![doc = include_str!("../README.md")]

use std::path::{Path, PathBuf};

mod version;
pub use version::*;

/// The up direction used for all world and
/// renderer coordinate calcualtions.
///
/// Y-up
pub const UP_DIRECTION: glam::Vec3 = glam::Vec3::Y;
/// The forward direction used for all world and
/// renderer coordinate calcualtions.
/// We use Z-forward because that is how Vulkan does it.
pub const FORWARD_DIRECTION: glam::Vec3 = glam::Vec3::Z;

/// Returns a canonicalized path relative to `base`.
///
/// # Errors
///
/// Returns an error if either path cannot be canonicalized or `path` is outside `base`.
pub fn format_path(base: &Path, path: &Path) -> std::io::Result<PathBuf> {
    let root = base.canonicalize()?;
    Ok(path
        .canonicalize()?
        .strip_prefix(&root)
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Path '{}' is not relative to base '{}'",
                    path.display(),
                    root.display()
                ),
            )
        })?
        .to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_path_checks_its_base() -> std::io::Result<()> {
        let root = std::env::temp_dir().join(format!("dirk-format-path-{}", std::process::id()));
        let base = root.join("base");
        let outside = root.join("outside");
        std::fs::create_dir_all(&base)?;
        std::fs::create_dir_all(&outside)?;
        let result = (|| {
            assert_eq!(format_path(&base, &base)?, PathBuf::new());
            assert!(format_path(&base, &outside).is_err());
            Ok(())
        })();
        std::fs::remove_dir_all(root)?;
        result
    }
}
