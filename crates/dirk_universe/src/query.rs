//! This module handles querying entities from a [`World`] based on what
//! components they have (or don't have).
//!
//! [`World`]: crate::World

pub mod experimental;
pub mod filter;

use std::any::TypeId;

use crate::{Entity, Universe, WorldId, components::Component};

/// A struct to query entities from a [`World`].
///
/// Conditions are evaluated as:
///   - ALL `with_component` types must be present, **and**
///   - NONE of the `without_component` types may be present, **and**
///   - the entity lives in **at least one** of the `with_world` worlds
///     (if any are specified — omitting `with_world` entirely matches all
///     worlds), **and**
///   - the entity does **not** live in any `without_world` world.
///
/// Note: unlike `with_component`, which uses AND across calls, `with_world`
/// uses OR across calls. This means the entity only needs to belong to *one*
/// of the specified worlds to match.
///
/// An empty query matches every entity.
///
/// # Example
/// ```rust
/// # use dirk_universe::components::Component;
/// # use dirk_universe::query::Query;
/// # #[derive(Component, Debug, serde::Deserialize, serde::Serialize)]
/// # struct Position;
/// # #[derive(Component, Debug, serde::Deserialize, serde::Serialize)]
/// # struct Velocity;
/// # #[derive(Component, Debug, serde::Deserialize, serde::Serialize)]
/// # struct Frozen;
/// let query = Query::empty()
///     .with_component::<Position>()
///     .with_component::<Velocity>()
///     .without_component::<Frozen>();
/// ```
///
/// [`World`]: crate::World
#[derive(Default)]
pub struct Query {
    /// Component types that must ALL be present on a matching entity.
    required_components: Vec<TypeId>,
    /// Component types that must ALL be absent on a matching entity.
    excluded_components: Vec<TypeId>,
    /// Worlds that the entity could be on.
    /// Matching is OR: the entity must live in at least one of these worlds.
    possible_worlds: Vec<WorldId>,
    /// Worlds that the entity must not be in (AND NOT across all entries).
    excluded_worlds: Vec<WorldId>,
}

impl Query {
    /// Creates a new, empty [`Query`] that matches every entity.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Require that matching entities have component `C`.
    ///
    /// Calling this multiple times with different types adds AND conditions.
    /// Adding the same type twice is a no-op; the condition is deduplicated.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `C` has already been added via
    /// [`without_component`], since such a query can never match anything.
    ///
    /// [`without_component`]: Query::without_component
    #[must_use]
    pub fn with_component<C: Component>(mut self) -> Self {
        let id = TypeId::of::<C>();
        debug_assert!(
            !self.excluded_components.contains(&id),
            "with_component and without_component called with the same type — \
             this query will never match any entity"
        );
        if !self.required_components.contains(&id) {
            self.required_components.push(id);
        }
        self
    }

    /// Require that matching entities do **not** have component `C`.
    ///
    /// Like [`with_component`], multiple calls add independent AND NOT
    /// conditions. Adding the same type twice is a no-op.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `C` has already been added via
    /// [`with_component`], since such a query can never match anything.
    ///
    /// [`with_component`]: Query::with_component
    #[must_use]
    pub fn without_component<C: Component>(mut self) -> Self {
        let id = TypeId::of::<C>();
        debug_assert!(
            !self.required_components.contains(&id),
            "without_component and with_component called with the same type — \
             this query will never match any entity"
        );
        if !self.excluded_components.contains(&id) {
            self.excluded_components.push(id);
        }
        self
    }

    /// Restrict matching entities to those that live in `world`.
    ///
    /// Multiple calls add OR conditions: an entity matches if it lives in
    /// **any** of the specified worlds. Calling with the same [`WorldId`]
    /// twice is a no-op.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `world` has already been added via
    /// [`without_world`], since such a query can never match anything.
    ///
    /// [`without_world`]: Query::without_world
    #[must_use]
    pub fn with_world(mut self, world: WorldId) -> Self {
        debug_assert!(
            !self.excluded_worlds.contains(&world),
            "with_world and without_world called with the same WorldId — \
             this query will never match any entity"
        );
        if !self.possible_worlds.contains(&world) {
            self.possible_worlds.push(world);
        }
        self
    }

    /// Require that matching entities do **not** live in `world`.
    ///
    /// Multiple calls add independent AND NOT conditions. Calling with the
    /// same [`WorldId`] twice is a no-op.
    ///
    /// # Panics (debug only)
    /// Panics in debug builds if `world` has already been added via
    /// [`with_world`], since such a query can never match anything.
    ///
    /// [`with_world`]: Query::with_world
    #[must_use]
    pub fn without_world(mut self, world: WorldId) -> Self {
        debug_assert!(
            !self.possible_worlds.contains(&world),
            "without_world and with_world called with the same WorldId — \
             this query will never match any entity"
        );
        if !self.excluded_worlds.contains(&world) {
            self.excluded_worlds.push(world);
        }
        self
    }

    /// Returns `true` if `entity` satisfies every condition in this [`Query`].
    ///
    /// This is the single source of truth; [`query`] is implemented in terms
    /// of it.
    ///
    /// [`query`]: Query::query
    #[must_use]
    pub(crate) fn matches(&self, universe: &Universe, entity: Entity) -> bool {
        let world_ok = self.possible_worlds.is_empty()
            || self
                .possible_worlds
                .iter()
                .any(|&world| universe.is_in_world(world, entity));

        if !world_ok {
            return false;
        }

        if self
            .excluded_worlds
            .iter()
            .any(|&world| universe.is_in_world(world, entity))
        {
            return false;
        }

        self.required_components
            .iter()
            .all(|&t| universe.components.contains(entity, t))
            && self
                .excluded_components
                .iter()
                .all(|&t| !universe.components.contains(entity, t))
    }

    /// Returns an iterator over all entities that satisfy this [`Query`].
    ///
    /// The iterator is lazy — no allocation occurs until the caller collects.
    /// Entities are sourced from [`Universe::entities`] so despawned IDs are
    /// never returned, even if their component data has not yet been cleaned up.
    ///
    /// [`World`]: crate::World
    pub(crate) fn query<'u>(&'u self, universe: &'u Universe) -> impl Iterator<Item = Entity> {
        universe
            .entities
            .keys()
            .copied()
            .filter(|&e| self.matches(universe, e))
    }
}
