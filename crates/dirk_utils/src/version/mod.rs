use std::{fmt, str::FromStr};

use thiserror::Error;

/// A packed Vulkan-style version with 10-bit major/minor and 12-bit patch fields.
/// This type does not implement Semantic Versioning or prerelease identifiers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Version(u32);

impl Default for Version {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Version {
    /// Version 0.0.0
    pub const ZERO: Self = Self(0);
    /// Create a new version object with specified numbers
    ///
    /// # Panics
    ///
    /// Will panic if the specified `major`, `minor` & `patch`
    /// do not fit the storage.
    /// `major`: version must fit in 10 bits (0–1023).
    /// `minor`: version must fit in 10 bits (0–1023).
    /// `patch`: version must fit in 12 bits (0–4095).
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        assert!(
            major < (1 << 10),
            "major version must fit in 10 bits (0–1023)"
        );
        assert!(
            minor < (1 << 10),
            "minor version must fit in 10 bits (0–1023)"
        );
        assert!(
            patch < (1 << 12),
            "patch version must fit in 12 bits (0–4095)"
        );
        Self(((major) << 22) | ((minor) << 12) | (patch))
    }
    /// Creates a version, or returns `None` if a component does not fit the
    /// storage limits documented on [`Version::new`].
    fn checked_new(major: u32, minor: u32, patch: u32) -> Option<Self> {
        (major < (1 << 10) && minor < (1 << 10) && patch < (1 << 12))
            .then(|| Self::new(major, minor, patch))
    }
    /// Returns the major number of the [`Version`]
    #[must_use]
    pub fn major(&self) -> u32 {
        self.0 >> 22
    }
    /// Returns the minor number of the [`Version`]
    #[must_use]
    pub fn minor(&self) -> u32 {
        (self.0 >> 12) & 0x3ff
    }
    /// Returns the patch number of the [`Version`]
    #[must_use]
    pub fn patch(&self) -> u32 {
        self.0 & 0xfff
    }
    /// Increments major, resets minor and patch to 0.
    ///
    /// Returns `None` if the major version is already at its maximum (1023).
    #[must_use]
    pub fn bump_major(self) -> Option<Self> {
        Self::checked_new(self.major() + 1, 0, 0)
    }
    /// Increments minor, resets patch to 0.
    ///
    /// Returns `None` if the minor version is already at its maximum (1023).
    #[must_use]
    pub fn bump_minor(self) -> Option<Self> {
        Self::checked_new(self.major(), self.minor() + 1, 0)
    }
    /// Increments patch.
    ///
    /// Returns `None` if the patch version is already at its maximum (4095).
    #[must_use]
    pub fn bump_patch(self) -> Option<Self> {
        Self::checked_new(self.major(), self.minor(), self.patch() + 1)
    }
    /// Returns true if both versions share a major version and this version is
    /// at least as new as `required`. This is not `SemVer` compatibility.
    #[must_use]
    pub fn is_compatible_with(self, required: Self) -> bool {
        self.major() == required.major() && self >= required
    }
    /// Returns whether the major version is zero. This is a project convention,
    /// not a `SemVer` prerelease identifier.
    #[must_use]
    pub fn is_prerelease(self) -> bool {
        self.major() == 0
    }
}

impl From<u32> for Version {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<Version> for u32 {
    fn from(v: Version) -> Self {
        v.0
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Comparing the raw packed `u32` is equivalent to comparing
/// (major, minor, patch) lexicographically because the fields are
/// stored in significance order.
impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major(), self.minor(), self.patch())
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Version({self})")
    }
}

/// An error occured in parsing the version string.
#[derive(Debug, PartialEq, Error)]
pub struct ParseVersionError(String);

impl fmt::Display for ParseVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid version string: '{}'", self.0)
    }
}

impl FromStr for Version {
    type Err = ParseVersionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || ParseVersionError(s.to_string());
        let mut parts = s.trim_start_matches('v').splitn(3, '.');
        let major = parts
            .next()
            .ok_or_else(err)?
            .parse::<u32>()
            .map_err(|_| err())?;
        let minor = parts
            .next()
            .ok_or_else(err)?
            .parse::<u32>()
            .map_err(|_| err())?;
        let patch = parts
            .next()
            .ok_or_else(err)?
            .parse::<u32>()
            .map_err(|_| err())?;
        Self::checked_new(major, minor, patch).ok_or_else(err)
    }
}

#[cfg(test)]
mod tests;
