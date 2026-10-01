//! Filters for the typed query API.

use std::{any::TypeId, marker::PhantomData};

use crate::{Entity, Universe, components::Component};

/// A predicate that decides whether an entity is included in a query.
///
/// Filters are combined with tuples (AND semantics) and composed from
/// [`With`] and [`Without`]. The empty tuple `()` matches every entity.
pub trait Filter {
    /// Returns `true` when `entity` satisfies this filter in `universe`.
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool;
}

macro_rules! impl_filter_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> Filter for ($($name,)+)
        where
            $($name: Filter),+
        {
            fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
                $($name::matches(entity, universe, last_run))&&+
            }
        }
    };
}

impl_filter_for_tuple!(A, B);
impl_filter_for_tuple!(A, B, C);
impl_filter_for_tuple!(A, B, C, D);
impl_filter_for_tuple!(A, B, C, D, E);
impl_filter_for_tuple!(A, B, C, D, E, F);
impl_filter_for_tuple!(A, B, C, D, E, F, G);
impl_filter_for_tuple!(A, B, C, D, E, F, G, H);

impl Filter for () {
    fn matches(_: Entity, _: &Universe, _: u64) -> bool {
        true
    }
}

/// Matches entities that have component `C`.
pub struct With<C: Component>(PhantomData<C>);
impl<C: Component> Filter for With<C> {
    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        universe.components.contains(entity, TypeId::of::<C>())
    }
}

/// Matches entities that do **not** have component `C`.
pub struct Without<C: Component>(PhantomData<C>);
impl<C: Component> Filter for Without<C> {
    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        !universe.components.contains(entity, TypeId::of::<C>())
    }
}

/// Matches components added since this system last ran. Replacing an existing
/// component is a change, but does not count as an addition.
pub struct Added<C: Component>(PhantomData<C>);
impl<C: Component> Filter for Added<C> {
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.added::<C>(entity, last_run)
    }
}

/// Matches components added or mutably dereferenced since this system last ran.
/// Each system observes changes independently; reading does not clear a flag.
pub struct Changed<C: Component>(PhantomData<C>);
impl<C: Component> Filter for Changed<C> {
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.changed::<C>(entity, last_run)
    }
}
