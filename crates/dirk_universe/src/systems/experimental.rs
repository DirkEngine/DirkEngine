//! Experimental standalone systems, planned to eventually replace the
//! [`super::UniverseSystem`]/[`super::TickingSystem`] family.
//!
//! This module is temporary; its API is subject to change.

use std::{any::type_name, marker::PhantomData};

use crate::{
    Universe,
    query::{
        experimental::{QueryItem, QueryParameter},
        filter::{DefaultFilter, Filter},
    },
};

/// A standalone experimental system that can run against a [`Universe`].
pub trait StandaloneSystem: Send + Sync + 'static {
    /// Returns a static name for diagnostics.
    fn name(&self) -> &'static str;

    /// Runs the system over the provided universe.
    fn run(&self, universe: &Universe);
}

/// A system backed by a function or closure that receives matching query items.
pub struct FuncSystem<Func, P, F = DefaultFilter> {
    func: Func,
    _marker: PhantomData<fn(P, F)>,
}

impl<Func, P, F> FuncSystem<Func, P, F>
where
    P: QueryParameter,
    F: Filter,
    Func: for<'u> Fn(QueryItem<'u, P, F>) + Send + Sync + 'static,
{
    /// Creates a function-backed system.
    #[must_use]
    pub fn new(func: Func) -> Self {
        Self {
            func,
            _marker: PhantomData,
        }
    }
}

impl<Func, P, F> StandaloneSystem for FuncSystem<Func, P, F>
where
    P: QueryParameter + 'static,
    F: Filter + 'static,
    Func: for<'u> Fn(QueryItem<'u, P, F>) + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        type_name::<Func>()
    }

    fn run(&self, universe: &Universe) {
        for query in QueryItem::<P, F>::iter(universe) {
            (self.func)(query);
        }
    }
}

/// Creates a function-backed system with the default filter.
#[must_use]
pub fn system<P, Func>(func: Func) -> FuncSystem<Func, P>
where
    P: QueryParameter,
    Func: for<'u> Fn(QueryItem<'u, P>) + Send + Sync + 'static,
{
    FuncSystem::new(func)
}

/// Creates a function-backed system with an explicit filter.
#[must_use]
pub fn filtered_system<P, F, Func>(func: Func) -> FuncSystem<Func, P, F>
where
    P: QueryParameter,
    F: Filter,
    Func: for<'u> Fn(QueryItem<'u, P, F>) + Send + Sync + 'static,
{
    FuncSystem::new(func)
}
