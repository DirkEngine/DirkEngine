//! Shared ownership and validation above native backend operations.
#![allow(
    clippy::missing_errors_doc,
    reason = "operations share the typed RHI error contract"
)]
mod device;
mod encoder;
mod presentation;
mod resource;
pub use device::*;
pub use encoder::*;
pub use presentation::*;
pub use resource::*;

use crate::{Backend, InvalidResourceKind as Ir, Result};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn identity() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| Ir::BadState.with_detail("RHI state lock poisoned").into())
}

/// Shared native resource, immutable metadata, and device identity.
pub struct Object<B: Backend, T, M>(pub(super) Arc<ObjectInner<B, T, M>>);
pub(super) struct ObjectInner<B: Backend, T, M> {
    raw: T,
    metadata: M,
    backend: Arc<B>,
    gate: Arc<Mutex<()>>,
    id: u64,
    busy: Arc<AtomicUsize>,
}
impl<B: Backend, T, M> Clone for Object<B, T, M> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<B: Backend, T, M> std::fmt::Debug for Object<B, T, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resource").field("id", &self.0.id).finish()
    }
}
impl<B: Backend, T, M> Object<B, T, M> {
    /// Stable identity shared by cloned handles.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.0.id
    }
    /// Immutable allocation or layout information.
    #[must_use]
    pub fn info(&self) -> &M {
        &self.0.metadata
    }
    pub(super) fn raw(&self) -> &T {
        &self.0.raw
    }
    pub(super) fn require_device(&self, backend: &Arc<B>) -> Result<()> {
        if !Arc::ptr_eq(&self.0.backend, backend) {
            return Err(Ir::ForeignInstance.into());
        }
        Ok(())
    }
}
