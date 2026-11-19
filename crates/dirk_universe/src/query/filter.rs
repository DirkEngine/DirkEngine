//! Filters for the typed query API.

use std::{any::TypeId, marker::PhantomData};

use crate::{Entity, Universe, components::Component, macros::sealed, schedule::Access};

/// A predicate that decides whether an entity is included in a query.
///
/// Filters are combined with tuples (AND semantics) and composed from
/// [`With`], [`Without`] and [`Changed`]. The empty tuple `()`
/// matches every entity.
pub trait QueryFilter: sealed::QueryFilter {
    /// Returns `true` when `entity` satisfies this filter in `universe`.
    #[doc(hidden)]
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool;

    /// Registers the components this filter depends on.
    #[doc(hidden)]
    fn register_access(access: &mut Access);
}

impl QueryFilter for () {
    fn matches(_: Entity, _: &Universe, _: u64) -> bool {
        true
    }

    fn register_access(_: &mut Access) {}
}

macro_rules! impl_filter_for_tuple {
    ($first:ident $first_binding:ident $(, $ty:ident $binding:ident)*) => {
        impl<$first: QueryFilter $(, $ty: QueryFilter)*> QueryFilter for ($first, $($ty,)*)
        where
            ($($ty,)*): QueryFilter,
        {
            fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
                $first::matches(entity, universe, last_run)
                    $(&& $ty::matches(entity, universe, last_run))*
            }

            fn register_access(access: &mut Access) {
                $first::register_access(access);
                $($ty::register_access(access);)*
            }
        }
    };
}
all_tuples!(impl_filter_for_tuple);

/// Matches entities that have component `C`.
pub struct With<C: Component>(PhantomData<C>);
impl<C: Component> sealed::QueryFilter for With<C> {}
impl<C: Component> QueryFilter for With<C> {
    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        universe.components.contains(entity, TypeId::of::<C>())
    }

    fn register_access(access: &mut Access) {
        access.observe::<C>();
    }
}

/// Matches entities that do **not** have component `C`.
pub struct Without<C: Component>(PhantomData<C>);
impl<C: Component> sealed::QueryFilter for Without<C> {}
impl<C: Component> QueryFilter for Without<C> {
    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        !universe.components.contains(entity, TypeId::of::<C>())
    }

    fn register_access(access: &mut Access) {
        access.observe::<C>();
    }
}

/// Matches components added or mutably dereferenced since this system last ran.
/// Each system observes changes independently; reading does not clear a flag.
pub struct Changed<C: Component>(PhantomData<C>);
impl<C: Component> sealed::QueryFilter for Changed<C> {}
impl<C: Component> QueryFilter for Changed<C> {
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.changed::<C>(entity, last_run)
    }

    fn register_access(access: &mut Access) {
        access.observe::<C>();
    }
}
