//! Portable validation and unique ownership above private native operations.
mod device;
mod encoder;
mod presentation;
mod resource;
pub use device::*;
pub use encoder::*;
pub use presentation::*;
pub use resource::*;

use crate::{Backend, InvalidResourceKind as Ir, Result};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Unique native resource and immutable allocation metadata.
/// Dropping the native object retires it through its device's garbage queue.
pub struct Object<B: Backend, T, M> {
    raw: T,
    metadata: M,
    marker: std::marker::PhantomData<fn() -> B>,
}
impl<B: Backend, T, M> std::fmt::Debug for Object<B, T, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource").finish_non_exhaustive()
    }
}
impl<B: Backend, T, M> Object<B, T, M> {
    /// Immutable allocation or layout information.
    #[must_use]
    pub fn info(&self) -> &M {
        &self.metadata
    }
    pub(super) fn raw(&self) -> &T {
        &self.raw
    }
}
