//! Allocates stable handles for worlds and entities.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::{Entity, WorldId};

/// Shared allocator for [`Entity`] handles, including the entities behind
/// [`WorldId`]s.
///
/// Cloning this type is cheap. Every clone points at the same counters, so
/// handles allocated from any clone stay unique within that allocator.
#[derive(Clone, Debug, Default)]
pub struct Allocator {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    next_entity: AtomicU64,
}

impl Allocator {
    /// Creates a new empty allocator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder can be merged only while no handle or queued buffer from it survives.
    pub(crate) fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Allocates a new [`WorldId`].
    #[must_use]
    pub fn allocate_world(&self) -> WorldId {
        WorldId::new(self.allocate_entity())
    }

    /// Allocates a new [`Entity`].
    #[must_use]
    pub fn allocate_entity(&self) -> Entity {
        Entity::new(self.inner.next_entity.fetch_add(1, Ordering::Relaxed))
    }
}
