//! Experimental query API, planned to eventually replace [`super::Query`].
//!
//! This module is temporary; its API is subject to change.

use std::marker::PhantomData;

use super::filter::{DefaultFilter, Filter};
use crate::{Entity, Universe, components::Component};

/// A matched entity and the data fetched for it by an experimental query.
pub struct QueryItem<'u, P: QueryParameter, F: Filter = DefaultFilter> {
    entity: Entity,
    params: P::Item<'u>,
    _filter: PhantomData<(&'u Universe, P, F)>,
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
    /// entities without `C`, even when `F` is [`DefaultFilter`].
    ///
    /// The iteration order is unspecified.
    pub fn iter(universe: &'u Universe) -> impl Iterator<Item = Self> + 'u {
        universe
            .entities
            .keys()
            .copied()
            .filter_map(move |entity| Self::matches(entity, universe))
    }
}

/// Describes the data fetched for each entity matched by an experimental query.
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

impl_query_parameter_for_tuple!(A);
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
