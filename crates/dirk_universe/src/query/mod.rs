//! Typed queries over live entities.
//!
//! Queries read live entities across all worlds. A tuple of parameters fetches
//! every requested component, skipping entities missing any of them. Filter
//! tuples also use AND semantics; the default filter `()` matches every entity.
//! `&C` reads a component. `&mut C` edits a clone of a component immediately
//! for later systems in the same tick, preserving the ordered change log.
//!
//! ```rust
//! use dirk_universe::{
//!     Universe, components::Component,
//!     query::{QueryItem, filter::Without},
//! };
//!
//! #[derive(Component, Debug)]
//! struct Position(f32);
//! #[derive(Component, Debug)]
//! struct Velocity(f32);
//! #[derive(Component, Debug)]
//! struct Frozen;
//!
//! fn inspect_moving_entities(universe: &Universe) {
//!     for query in QueryItem::<(&Position, &Velocity), Without<Frozen>>::iter(universe) {
//!         let entity = query.entity();
//!         let (position, velocity) = query.into_params();
//!         println!("{entity:?}: position {}, velocity {}", position.0, velocity.0);
//!     }
//! }
//! ```

pub mod filter;

use std::{
    any::{TypeId, type_name},
    collections::HashMap,
    marker::PhantomData,
};

use self::filter::Filter;
use crate::{
    Entity, Universe, WorldId,
    components::{Component, ComponentMut},
};

/// One matching entity supplied to a system invocation.
///
/// The engine calls the system once for each match. Use `into_params()` to
/// access its components; no iteration is needed inside the system.
pub type Query<'u, P, F = ()> = QueryItem<'u, P, F>;

/// Read-only lookups for systems that run once per tick, including lifecycle
/// systems that must run when no live entities match. This parameter does not
/// cause per-entity invocation.
pub struct QueryView<'u, P: ReadOnlyQueryParameter, F: Filter = ()> {
    universe: &'u Universe,
    _marker: PhantomData<fn() -> (P, F)>,
}

impl<'u, P: ReadOnlyQueryParameter, F: Filter> QueryView<'u, P, F> {
    pub(crate) fn new(universe: &'u Universe) -> Self {
        Self {
            universe,
            _marker: PhantomData,
        }
    }

    /// Iterates over live entities matching this read-only query.
    pub fn iter(&self) -> impl Iterator<Item = QueryItem<'u, P, F>> + 'u {
        QueryItem::iter(self.universe)
    }

    /// Iterates over matching entities in one world.
    pub fn iter_in_world(&self, world: WorldId) -> impl Iterator<Item = QueryItem<'u, P, F>> + 'u {
        let universe = self.universe;
        universe
            .entities_in_world(world)
            .filter_map(move |entity| QueryItem::matches(entity, universe))
    }

    /// Fetches one entity if it is alive and matches this query.
    #[must_use]
    pub fn get(&self, entity: Entity) -> Option<QueryItem<'u, P, F>> {
        self.universe
            .is_alive(entity)
            .then(|| QueryItem::matches(entity, self.universe))
            .flatten()
    }
}

/// A matched entity and the data fetched for it by a query.
pub struct QueryItem<'u, P: QueryParameter, F: Filter = ()> {
    entity: Entity,
    params: P::Item<'u>,
    _filter: PhantomData<fn() -> (P, F)>,
}

impl<'u, P: QueryParameter, F: Filter> QueryItem<'u, P, F> {
    pub(crate) fn matches(entity: Entity, universe: &'u Universe) -> Option<Self> {
        if !F::matches(entity, universe) {
            return None;
        }

        Some(Self {
            entity,
            params: P::from_entity(entity, universe)?,
            _filter: PhantomData,
        })
    }

    /// Returns the entity matched by this query item.
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Returns the fetched query parameters.
    pub fn params(&self) -> &P::Item<'u> {
        &self.params
    }

    /// Consumes this query item and returns the fetched parameters.
    pub fn into_params(self) -> P::Item<'u> {
        self.params
    }
}

impl<'u, P: ReadOnlyQueryParameter, F: Filter> QueryItem<'u, P, F> {
    /// Iterates over every live entity for which `F` matches and every
    /// parameter of `P` fetches successfully. Entities from every world are
    /// included; iteration order is unspecified.
    pub fn iter(universe: &'u Universe) -> impl Iterator<Item = Self> + 'u {
        universe
            .entities
            .keys()
            .copied()
            .filter_map(move |entity| Self::matches(entity, universe))
    }
}

/// Describes the data fetched for each entity matched by a query.
pub trait QueryParameter: Sized {
    /// The concrete value borrowed or produced for one matched entity.
    type Item<'u>;

    /// Builds this parameter value for `entity`, returning `None` if it does not match.
    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>>;

    /// Registers component access for conflict validation.
    fn register_access(access: &mut QueryAccess);
}

/// Tracks the component access declared by one system.
#[doc(hidden)]
#[derive(Default)]
pub struct QueryAccess {
    components: HashMap<TypeId, AccessKind>,
    commands: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AccessKind {
    Read,
    Write,
}

impl QueryAccess {
    /// Registers a read of `C`.
    ///
    /// # Panics
    /// Panics when this system also declares mutable access to `C`.
    pub fn read<C: Component>(&mut self) {
        assert!(
            self.components.get(&TypeId::of::<C>()) != Some(&AccessKind::Write),
            "system reads and writes {} through overlapping queries",
            type_name::<C>()
        );
        self.components
            .entry(TypeId::of::<C>())
            .or_insert(AccessKind::Read);
    }

    /// Registers a write of `C`.
    ///
    /// # Panics
    /// Panics when this system already declares any access to `C`.
    pub fn write<C: Component>(&mut self) {
        assert!(
            self.components
                .insert(TypeId::of::<C>(), AccessKind::Write)
                .is_none(),
            "system accesses {} through overlapping mutable queries",
            type_name::<C>()
        );
    }

    /// Registers structural command access.
    ///
    /// # Panics
    /// Panics when this system declares `Commands` more than once.
    pub fn commands(&mut self) {
        assert!(
            !std::mem::replace(&mut self.commands, true),
            "system requests Commands more than once"
        );
    }
}

/// Query parameters that do not require mutable access.
pub trait ReadOnlyQueryParameter: QueryParameter {}

impl QueryParameter for () {
    type Item<'u> = ();

    fn from_entity(_: Entity, _: &Universe) -> Option<Self::Item<'_>> {
        Some(())
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryParameter for () {}

macro_rules! impl_query_parameter_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> QueryParameter for ($($name,)+)
        where
            $($name: QueryParameter),+
        {
            type Item<'u> = ($($name::Item<'u>,)+);

            fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
                Some(($($name::from_entity(entity, universe)?,)+))
            }

            fn register_access(access: &mut QueryAccess) {
                $($name::register_access(access);)+
            }
        }

        impl<$($name: ReadOnlyQueryParameter),+> ReadOnlyQueryParameter for ($($name,)+) {}
    };
}

impl_query_parameter_for_tuple!(A, B);
impl_query_parameter_for_tuple!(A, B, C);
impl_query_parameter_for_tuple!(A, B, C, D);
impl_query_parameter_for_tuple!(A, B, C, D, E);
impl_query_parameter_for_tuple!(A, B, C, D, E, F);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G, H);

/// Compatibility alias for a read-only component query.
pub type Read<C> = &'static C;

impl<C: Component> QueryParameter for &C {
    type Item<'u> = &'u C;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }

    fn register_access(access: &mut QueryAccess) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryParameter for &C {}

impl<C: Component + Clone> QueryParameter for &mut C {
    type Item<'u> = ComponentMut<'u, C>;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe
            .components
            .get_mut::<C>(entity, &universe.edit_order, &universe.prepared_order)
    }

    fn register_access(access: &mut QueryAccess) {
        access.write::<C>();
    }
}
