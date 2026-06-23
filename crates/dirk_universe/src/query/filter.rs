use std::{any::TypeId, marker::PhantomData};

use crate::{Entity, Universe, components::Component};

pub trait Filter {
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

impl_filter_for_tuple!(A, B);
impl_filter_for_tuple!(A, B, C);
impl_filter_for_tuple!(A, B, C, D);
impl_filter_for_tuple!(A, B, C, D, E);
impl_filter_for_tuple!(A, B, C, D, E, F);
impl_filter_for_tuple!(A, B, C, D, E, F, G);
impl_filter_for_tuple!(A, B, C, D, E, F, G, H);

pub struct DefaultFilter;
impl Filter for DefaultFilter {
    fn matches(_: Entity, _: &Universe) -> bool {
        true
    }
}

impl<C: Component> Filter for C {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        universe.components.contains(entity, TypeId::of::<C>())
    }
}

pub struct Not<C: Component>(PhantomData<C>);
impl<C: Component> Filter for Not<C> {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        !universe.components.contains(entity, TypeId::of::<C>())
    }
}

pub struct Changed;
impl Filter for Changed {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        // TODO: implement change detection
        true
    }
}

pub struct Spawned;
impl Filter for Spawned {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        // TODO: detect if entity was just spawned
        true
    }
}

pub struct Despawned;
impl Filter for Despawned {
    fn matches(entity: Entity, universe: &Universe) -> bool {
        // TODO: detect if entity was just despawned
        true
    }
}
