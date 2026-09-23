//! Systems run once per tick, in registration order.
//!
//! Implement [`System`] for a stateful type or use [`FuncSystem::new`] for a
//! function or closure. Both register through [`crate::UniverseBuilder::with_system`].

use crate::{CommandBuffer, Universe};
use std::any::type_name;

/// A sequential task with access to queries and this tick's changes.
///
/// Systems run even when the universe has no entities. Commands produced by a
/// system are applied on the next tick; component borrows cannot escape a run.
pub trait System: 'static {
    /// Returns the type name for diagnostics.
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Reads the current universe and queues changes for the next tick.
    /// `delta_time` is measured in seconds.
    fn run(&mut self, commands: &mut CommandBuffer, universe: &Universe, delta_time: f64);
}

/// A system backed by a function or stateful closure.
pub struct FuncSystem<Func> {
    func: Func,
}

impl<Func> FuncSystem<Func>
where
    Func: FnMut(&mut CommandBuffer, &Universe, f64) + 'static,
{
    /// Creates a system called once per tick, including when no entities match.
    /// Use typed queries or change iterators inside the callback.
    #[must_use]
    pub fn new(func: Func) -> Self {
        Self { func }
    }
}

impl<Func> System for FuncSystem<Func>
where
    Func: FnMut(&mut CommandBuffer, &Universe, f64) + 'static,
{
    fn name(&self) -> &'static str {
        type_name::<Func>()
    }
    fn run(&mut self, commands: &mut CommandBuffer, universe: &Universe, delta_time: f64) {
        (self.func)(commands, universe, delta_time);
    }
}
