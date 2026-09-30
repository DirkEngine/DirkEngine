use std::fmt::Display;

use crate::{
    Entity, EntityBuilder,
    components::{Component, MutableComponent, ReadOnly},
};

/// Identifies a [`World`]. Worlds are entities: the ID wraps the entity that
/// carries the world's [`World`] component and any other world-level data.
#[derive(Clone, Copy, Debug, Default, Hash, Eq, PartialEq)]
pub struct WorldId(Entity);

impl WorldId {
    #[must_use]
    pub(crate) fn new(entity: Entity) -> Self {
        Self(entity)
    }

    /// Returns the ID of the world represented by `entity`, such as a world
    /// reported by `Delta<World>`. Commands ignore IDs of entities that are
    /// not live worlds.
    #[must_use]
    pub fn from_entity(entity: Entity) -> Self {
        Self(entity)
    }

    /// Returns the entity representing this world.
    #[must_use]
    pub fn entity(self) -> Entity {
        self.0
    }

    /// Returns the raw ID of the world's entity.
    #[must_use]
    pub fn raw(self) -> u64 {
        self.0.raw()
    }
}

impl Display for WorldId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.raw())
    }
}

/// The component that makes an entity a world. Worlds hold entities, which
/// carry an [`InWorld`] component naming their world. Query `&World` to list
/// worlds, and `Delta<World>` to observe their creation and destruction.
#[derive(Clone, Debug)]
pub struct World {
    id: WorldId,
    name: String,
}

impl Component for World {
    type Mutability = ReadOnly;
}

impl World {
    /// Returns a [`WorldBuilder`].
    #[must_use]
    pub fn builder(name: impl Into<String>) -> WorldBuilder {
        WorldBuilder::new(name)
    }

    pub(crate) fn new(id: WorldId, name: String) -> Self {
        Self { id, name }
    }

    /// Returns the [`WorldId`] of the [`World`].
    #[must_use]
    pub fn id(&self) -> WorldId {
        self.id
    }

    /// Returns the name of the [`World`].
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The world an entity lives in. Spawning adds it, sending an entity replaces
/// it and despawning removes it, so `Delta<InWorld>` observes all three.
///
/// Only the engine writes it; move entities with
/// [`CommandBuffer::send`](crate::CommandBuffer::send) instead:
///
/// ```compile_fail
/// # use dirk_universe::prelude::*;
/// fn teleport(query: Query<&mut InWorld>) {}
/// ```
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct InWorld(pub WorldId);

impl Component for InWorld {
    type Mutability = ReadOnly;
}

/// Marks a world whose entities only `PerWorld` queries reach: queries
/// spanning worlds skip them, and entities cannot be sent into or out of it.
/// The world entity itself, with its world-level components, stays visible to
/// every query. Set it with [`WorldBuilder::isolated`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Isolated;

impl Component for Isolated {
    type Mutability = ReadOnly;
}

/// Builder struct for [`World`].
#[derive(Default)]
pub struct WorldBuilder {
    pub(crate) name: String,
    /// Components of the world entity itself.
    pub(crate) world: EntityBuilder,
    pub(crate) isolated: bool,
    pub(crate) entities: Vec<EntityBuilder>,
}

impl WorldBuilder {
    /// Creates a new empty [`WorldBuilder`].
    #[must_use]
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Adds an [`Entity`] that will be spawned on [`World`] creation.
    #[must_use]
    pub fn with_entity(mut self, entity: EntityBuilder) -> Self {
        self.entities.push(entity);
        self
    }

    /// Adds a world-level component, such as gravity, to the world entity.
    #[must_use]
    pub fn with_component<C: MutableComponent>(mut self, component: C) -> Self {
        self.world = self.world.with_component(component);
        self
    }

    /// Isolates the world: see [`Isolated`].
    #[must_use]
    pub fn isolated(mut self) -> Self {
        self.isolated = true;
        self
    }
}
