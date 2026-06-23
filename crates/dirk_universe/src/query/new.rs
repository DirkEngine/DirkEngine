//! The new query system. It is here only temporarily to avoid breaking APIs during dev.

#![allow(unused, missing_docs)]

use std::{any::TypeId, marker::PhantomData};

use super::filter::*;
use crate::{Entity, Universe, components::Component};

pub struct Query<'u, P: QueryParameter<'u>, F: Filter = DefaultFilter> {
    entity: Entity,
    params: P,
    _filter: PhantomData<(&'u Universe, F)>,
}

impl<'u, P: QueryParameter<'u>, F: Filter> Query<'u, P, F> {
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

    pub fn entity(&self) -> Entity {
        self.entity
    }

    pub fn params(&self) -> &P {
        &self.params
    }
}

pub trait QueryParameter<'u>: Sized {
    fn from_entity(entity: Entity, universe: &'u Universe) -> Option<Self>;
}

macro_rules! impl_query_parameter_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<'u, $($name),+> QueryParameter<'u> for ($($name,)+)
        where
            $($name: QueryParameter<'u>),+
        {
            fn from_entity(entity: Entity, universe: &'u Universe) -> Option<Self> {
                Some(($($name::from_entity(entity, universe)?),+))
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

impl<'u, C: Component> QueryParameter<'u> for &'u C {
    fn from_entity(entity: Entity, universe: &'u Universe) -> Option<Self> {
        universe.component::<C>(entity)
    }
}
