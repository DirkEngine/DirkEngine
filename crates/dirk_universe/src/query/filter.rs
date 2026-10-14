//! Filters for the typed query API.

use std::{any::TypeId, marker::PhantomData};

use super::{AnyEntity, Join, LiveEntity, Liveness};
use crate::{Entity, Universe, components::Component, macros::sealed::Sealed};

/// A predicate that decides whether an entity is included in a query.
///
/// Filters are combined with tuples (AND semantics) and composed from
/// [`With`], [`Without`], [`Added`] and [`Changed`]. The empty tuple `()`
/// matches every entity.
pub trait QueryFilter: Sealed {
    /// Which entities this filter can describe.
    #[doc(hidden)]
    type Liveness: Liveness;

    /// Returns `true` when `entity` satisfies this filter in `universe`.
    #[doc(hidden)]
    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool;
}

impl QueryFilter for () {
    type Liveness = AnyEntity;

    fn matches(_: Entity, _: &Universe, _: u64) -> bool {
        true
    }
}

macro_rules! impl_filter_for_tuple {
    ($first:ident $first_binding:ident $(, $ty:ident $binding:ident)*) => {
        impl<$first: QueryFilter $(, $ty: QueryFilter)*> QueryFilter for ($first, $($ty,)*)
        where
            ($($ty,)*): QueryFilter,
            $first::Liveness: Join<<($($ty,)*) as QueryFilter>::Liveness>,
        {
            type Liveness =
                <$first::Liveness as Join<<($($ty,)*) as QueryFilter>::Liveness>>::Output;

            fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
                $first::matches(entity, universe, last_run)
                    $(&& $ty::matches(entity, universe, last_run))*
            }
        }
    };
}
all_tuples!(impl_filter_for_tuple);

/// Matches entities that have component `C`.
pub struct With<C: Component>(PhantomData<C>);
impl<C: Component> Sealed for With<C> {}
impl<C: Component> QueryFilter for With<C> {
    type Liveness = LiveEntity;

    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        universe.components.contains(entity, TypeId::of::<C>())
    }
}

/// Matches entities that do **not** have component `C`.
pub struct Without<C: Component>(PhantomData<C>);
impl<C: Component> Sealed for Without<C> {}
impl<C: Component> QueryFilter for Without<C> {
    type Liveness = AnyEntity;

    fn matches(entity: Entity, universe: &Universe, _: u64) -> bool {
        !universe.components.contains(entity, TypeId::of::<C>())
    }
}

/// Matches components added since this system last ran. Replacing an existing
/// component is a change, but does not count as an addition.
pub struct Added<C: Component>(PhantomData<C>);
impl<C: Component> Sealed for Added<C> {}
impl<C: Component> QueryFilter for Added<C> {
    type Liveness = LiveEntity;

    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.added::<C>(entity, last_run)
    }
}

/// Matches components added or mutably dereferenced since this system last ran.
/// Each system observes changes independently; reading does not clear a flag.
pub struct Changed<C: Component>(PhantomData<C>);
impl<C: Component> Sealed for Changed<C> {}
impl<C: Component> QueryFilter for Changed<C> {
    type Liveness = LiveEntity;

    fn matches(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.changed::<C>(entity, last_run)
    }
}
