//! Filters for the experimental query API.

use std::{any::TypeId, marker::PhantomData};

use crate::{Entity, Universe, components::Component};

/// A predicate that decides whether an entity is included in a query.
///
/// Filters are combined with tuples (AND semantics) and composed from
/// [`With`], [`Not`] and [`DefaultFilter`].
pub trait Filter {
    /// Returns `true` when `entity` satisfies this filter in `universe`.
    fn matches(entity: Entity, universe: &Universe) -> bool;
}

macro_rules! impl_filter_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> Filter for ($($name,)+)
        where
            $($name: Filter),+
        {
            fn matches(entity: Entity, universe: &Universe) -> bool {
                $($name::matches(entity, universe))&&+
            }
        }
    };
}

impl_filter_for_tuple!(A);
impl_filter_for_tuple!(A, B);
impl_filter_for_tuple!(A, B, C);
impl_filter_for_tuple!(A, B, C, D);
impl_filter_for_tuple!(A, B, C, D, E);
impl_filter_for_tuple!(A, B, C, D, E, F);
impl_filter_for_tuple!(A, B, C, D, E, F, G);
impl_filter_for_tuple!(A, B, C, D, E, F, G, H);

impl Filter for () {
    fn matches(_: Entity, _: &Universe) -> bool {
        true
    }
}

/// The default filter, which matches every entity.
pub struct DefaultFilter;
impl Filter for DefaultFilter {
    fn matches(_: Entity, _: &Universe) -> bool {
        true
    }
}

/// Matches entities that have component `C`.
pub struct With<C: Component>(PhantomData<C>);
impl<C: Component> Filter for With<C> {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        universe.components.contains(entity, TypeId::of::<C>())
    }
}

/// Matches entities that do **not** have component `C`.
pub struct Not<C: Component>(PhantomData<C>);
impl<C: Component> Filter for Not<C> {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        !universe.components.contains(entity, TypeId::of::<C>())
    }
}
