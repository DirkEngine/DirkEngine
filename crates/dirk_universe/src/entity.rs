//! This struct holds everything to do with [`Entity`]s.
//! Entities are just simple handles.
//! You can spawn them by creating an entity builder

use std::{any::TypeId, collections::HashMap};

use crate::components::{AnyComponent, MutableComponent};

/// A unique, opaque identifier for a spawned entity.
#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub struct Entity(u64);

impl Entity {
    #[must_use]
    pub(crate) fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns an empty [`EntityBuilder`].
    #[must_use]
    pub fn builder() -> EntityBuilder {
        EntityBuilder::new()
    }
    /// Returns the raw entity ID
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// A builder struct to create a new entity. Allows adding of components.
#[derive(Default)]
pub struct EntityBuilder {
    pub(crate) components: HashMap<TypeId, Box<dyn AnyComponent>>,
}

impl EntityBuilder {
    #[must_use]
    fn new() -> Self {
        Self::default()
    }

    /// Adds a [`Component`](crate::components::Component) to this entity.
    ///
    /// If adding multiple components of the same type, only the last call
    /// to `with_component` will be kept.
    #[must_use]
    pub fn with_component<C: MutableComponent>(mut self, component: C) -> Self {
        self.components
            .insert(TypeId::of::<C>(), Box::new(component));
        self
    }
}
