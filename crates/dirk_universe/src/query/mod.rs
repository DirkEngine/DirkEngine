//! Typed, read-only queries over live entities.
//!
//! Queries read live entities across all worlds. A tuple of parameters fetches
//! every requested component, skipping entities missing any of them. Filter
//! tuples also use AND semantics; the default filter `()` matches every entity.
//! Component references borrow the universe, so mutation remains deferred through
//! command buffers.
//!
//! ```rust
//! use dirk_universe::{
//!     Universe, components::Component,
//!     query::{QueryItem, Read, filter::Without},
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
//!     for query in QueryItem::<(Read<Position>, Read<Velocity>), Without<Frozen>>::iter(universe) {
//!         let entity = query.entity();
//!         let (position, velocity) = query.into_params();
//!         println!("{entity:?}: position {}, velocity {}", position.0, velocity.0);
//!     }
//! }
//! ```

pub mod filter;

use std::marker::PhantomData;

use self::filter::Filter;
use crate::{Entity, Universe, components::Component};

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

    /// Iterates over every live entity for which `F` matches and every
    /// parameter of `P` fetches successfully — e.g. `Read<C>` already skips
    /// entities without `C`, even when `F` is the default `()` filter.
    ///
    /// Entities from every world are included. The iteration order is unspecified.
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
}

impl QueryParameter for () {
    type Item<'u> = ();

    fn from_entity(_: Entity, _: &Universe) -> Option<Self::Item<'_>> {
        Some(())
    }
}

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
        }
    };
}

impl_query_parameter_for_tuple!(A, B);
impl_query_parameter_for_tuple!(A, B, C);
impl_query_parameter_for_tuple!(A, B, C, D);
impl_query_parameter_for_tuple!(A, B, C, D, E);
impl_query_parameter_for_tuple!(A, B, C, D, E, F);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G, H);

/// Fetches an immutable component reference for each matched entity.
pub struct Read<C: Component>(PhantomData<C>);

impl<C: Component> QueryParameter for Read<C> {
    type Item<'u> = &'u C;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }
}
