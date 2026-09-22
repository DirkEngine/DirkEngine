//! Experimental systems that run once per matching entity on each tick.
//!
//! Register systems with [`crate::UniverseBuilder::with_system`], or invoke
//! [`StandaloneSystem::run`] directly. This API is subject to change and does
//! not yet replace the lifecycle hooks on [`super::UniverseSystem`].

use std::{any::type_name, marker::PhantomData};

use crate::{
    CommandBuffer, Universe,
    query::{
        experimental::{QueryItem, QueryParameter},
        filter::Filter,
    },
};

/// An experimental system that can run against a shared [`Universe`].
///
/// Systems run sequentially and may own mutable state without synchronization.
pub trait StandaloneSystem: 'static {
    /// Returns a static name for diagnostics.
    fn name(&self) -> &'static str;

    /// Runs the system using the current tick's component data and delta time.
    ///
    /// Commands are queued in `cmd`; the caller submits the buffer after running
    /// its systems. [`Universe::tick`] applies these commands on the next tick.
    fn run(&mut self, cmd: &mut CommandBuffer, universe: &Universe, delta_time: f64);
}

/// A system backed by a function or closure that receives matching query items.
pub struct FuncSystem<Func, P, F = ()> {
    func: Func,
    _marker: PhantomData<fn(P, F)>,
}

impl<Func, P, F> FuncSystem<Func, P, F>
where
    P: QueryParameter + 'static,
    F: Filter + 'static,
    Func: for<'u> FnMut(&mut CommandBuffer, QueryItem<'u, P, F>, f64) + 'static,
{
    /// Creates a system from a function or stateful closure.
    ///
    /// The callback receives the tick's command buffer, one matching query item,
    /// and delta time in seconds. Annotate the query item to infer its parameter
    /// and filter types. It is called once per matching entity, in unspecified
    /// entity order, and is not called when there are no matches.
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
    Func: for<'u> FnMut(&mut CommandBuffer, QueryItem<'u, P, F>, f64) + 'static,
{
    fn name(&self) -> &'static str {
        type_name::<Func>()
    }

    fn run(&mut self, cmd: &mut CommandBuffer, universe: &Universe, delta_time: f64) {
        for query in QueryItem::<P, F>::iter(universe) {
            (self.func)(cmd, query, delta_time);
        }
    }
}
