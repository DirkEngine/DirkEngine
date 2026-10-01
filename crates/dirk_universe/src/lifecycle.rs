//! Minimal structural notifications, retained for the current tick.

use crate::{Entity, WorldId};
use std::any::TypeId;

/// A structural transition, in command application order.
/// Component updates are tracked by query filters instead of events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleEvent {
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
    /// A component was removed, including by despawn or world destruction.
    ComponentRemoved {
        /// The affected entity.
        entity: Entity,
        /// The removed component's type. Its value is dropped immediately.
        type_id: TypeId,
    },
}
