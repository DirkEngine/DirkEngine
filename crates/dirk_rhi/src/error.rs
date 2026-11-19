use std::{error::Error as StdError, time::Duration};

use crate::Completion;

/// Result of a fallible RHI operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors reported by the RHI. Every variant carries the detail needed to act
/// on it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No device met the RHI's requirements. Lists each rejected device and
    /// why it was rejected.
    #[error("no suitable graphics device: {0}")]
    NoSuitableDevice(String),
    /// A native graphics API call failed.
    #[error("{operation} failed")]
    Native {
        /// What the RHI was doing, e.g. `vkCreateBuffer`.
        operation: &'static str,
        /// The native error.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
    /// The device was lost. Every object created from it is unusable.
    #[error("the graphics device was lost")]
    DeviceLost,
    /// A wait ended before its submission completed.
    #[error("timed out after {timeout:?} waiting for {completion}")]
    Timeout {
        /// The completion waited for.
        completion: Completion,
        /// How long the wait lasted.
        timeout: Duration,
    },
    /// A request was rejected before reaching the native API.
    #[error("invalid {object} `{label}`: {reason}")]
    Invalid {
        /// Kind of object or request, e.g. `buffer`.
        object: &'static str,
        /// The object's debug label.
        label: String,
        /// Why the request was rejected.
        reason: String,
    },
}

impl Error {
    /// Wraps a native error with the operation that produced it.
    pub(crate) fn native(
        operation: &'static str,
        source: impl Into<Box<dyn StdError + Send + Sync>>,
    ) -> Self {
        Self::Native {
            operation,
            source: source.into(),
        }
    }

    /// Rejects a request for `object` labelled `label`.
    pub(crate) fn invalid(object: &'static str, label: &str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            object,
            label: label.to_owned(),
            reason: reason.into(),
        }
    }
}
