//! Ordered changes produced when command buffers are applied.
//!
//! Changes are available to every system for one tick and are replaced at the
//! start of the next tick. They preserve command order, including intermediate
//! component values and entities created and destroyed within the same tick.
//! Queries describe the final live state; changes retain historical values.

use crate::{
    Entity, WorldId,
    components::{Component, ComponentValue},
};

/// A lifecycle or component change, in command application order.
#[derive(Debug)]
pub enum Change {
    /// A world was created, before its entities are spawned.
    WorldCreated {
        /// The created world.
        world: WorldId,
    },
    /// A world was destroyed, after all its entities were despawned.
    WorldDestroyed {
        /// The destroyed world.
        world: WorldId,
    },
    /// An entity was spawned, before its component additions.
    EntitySpawned {
        /// The spawned entity.
        entity: Entity,
        /// Its initial world.
        world: WorldId,
    },
    /// An entity moved between worlds.
    EntityMoved {
        /// The moved entity.
        entity: Entity,
        /// Its previous world.
        from: WorldId,
        /// Its destination world.
        to: WorldId,
    },
    /// An entity was despawned, after its component removals.
    EntityDespawned {
        /// The despawned entity.
        entity: Entity,
        /// Its last world.
        world: WorldId,
    },
    /// A component was attached.
    ComponentAdded {
        /// The affected entity.
        entity: Entity,
        /// The attached value.
        component: ComponentValue,
    },
    /// A component was replaced. Both values remain readable for this tick.
    ComponentUpdated {
        /// The affected entity.
        entity: Entity,
        /// The replaced value.
        old: ComponentValue,
        /// The replacement value.
        new: ComponentValue,
    },
    /// A component was removed, including by despawn or world destruction.
    ComponentRemoved {
        /// The affected entity.
        entity: Entity,
        /// The removed value.
        component: ComponentValue,
    },
}

/// A typed view of a component change. Values need not implement `Clone`.
#[derive(Debug)]
pub enum ComponentChange<'a, C: Component> {
    /// A component was attached.
    Added {
        /// The affected entity.
        entity: Entity,
        /// The attached value.
        component: &'a C,
    },
    /// A component was replaced.
    Updated {
        /// The affected entity.
        entity: Entity,
        /// The replaced value.
        old: &'a C,
        /// The replacement value.
        new: &'a C,
    },
    /// A component was removed.
    Removed {
        /// The affected entity.
        entity: Entity,
        /// The removed value.
        component: &'a C,
    },
}

impl Change {
    /// Borrows a typed component change, or returns `None` for other changes.
    #[must_use]
    pub fn component<C: Component>(&self) -> Option<ComponentChange<'_, C>> {
        match self {
            Self::ComponentAdded { entity, component } => Some(ComponentChange::Added {
                entity: *entity,
                component: component.get()?,
            }),
            Self::ComponentUpdated { entity, old, new } => Some(ComponentChange::Updated {
                entity: *entity,
                old: old.get()?,
                new: new.get()?,
            }),
            Self::ComponentRemoved { entity, component } => Some(ComponentChange::Removed {
                entity: *entity,
                component: component.get()?,
            }),
            _ => None,
        }
    }
}
