//! Systems run in registration order, once per tick.
//!
//! Functions and stateful [`System`] implementations declare the data they
//! receive. The universe fetches those parameters before each run.

use std::{
    any::{TypeId, type_name},
    cell::{RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    CommandBuffer, Entity, Universe,
    components::Component,
    lifecycle::LifecycleEvent,
    query::{Query, QueryAccess, QueryParameter, filter::Filter},
};

/// Data supplied to one system invocation.
pub trait SystemParam {
    /// The value supplied while the universe is borrowed for this invocation.
    type Item<'u>;

    /// Fetches a parameter using this system's previous execution counter.
    fn fetch<'u>(
        universe: &'u Universe,
        last_run: u64,
        delta_time: f64,
        commands: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u>;

    /// Registers this parameter's accesses before the system runs.
    fn register_access(access: &mut QueryAccess);
}

impl SystemParam for () {
    type Item<'u> = ();

    fn fetch<'u>(_: &'u Universe, _: u64, _: f64, _: &'u RefCell<CommandBuffer>) -> Self::Item<'u> {
    }

    fn register_access(_: &mut QueryAccess) {}
}

macro_rules! impl_system_param_tuple {
    ($($param:ident),+) => {
        impl<$($param: SystemParam),+> SystemParam for ($($param,)+) {
            type Item<'u> = ($($param::Item<'u>,)+);

            fn fetch<'u>(
                universe: &'u Universe,
                last_run: u64,
                delta_time: f64,
                commands: &'u RefCell<CommandBuffer>,
            ) -> Self::Item<'u> {
                ($($param::fetch(universe, last_run, delta_time, commands),)+)
            }

            fn register_access(access: &mut QueryAccess) {
                $($param::register_access(access);)+
            }
        }
    };
}

impl_system_param_tuple!(A);
impl_system_param_tuple!(A, B);
impl_system_param_tuple!(A, B, C);
impl_system_param_tuple!(A, B, C, D);

impl<P: QueryParameter, F: Filter> SystemParam for Query<'_, P, F> {
    type Item<'u> = Query<'u, P, F>;

    fn fetch<'u>(
        universe: &'u Universe,
        last_run: u64,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u> {
        Query::for_system(universe, last_run)
    }

    fn register_access(access: &mut QueryAccess) {
        P::register_access(access);
    }
}

/// Optional structural commands for a system. Component values can be edited
/// directly through `Query<&mut C>`.
pub struct Commands<'u> {
    buffer: RefMut<'u, CommandBuffer>,
}

impl Deref for Commands<'_> {
    type Target = CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl DerefMut for Commands<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl SystemParam for Commands<'_> {
    type Item<'u> = Commands<'u>;

    fn fetch<'u>(
        _: &'u Universe,
        _: u64,
        _: f64,
        commands: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u> {
        Commands {
            buffer: commands.borrow_mut(),
        }
    }

    fn register_access(access: &mut QueryAccess) {
        access.commands();
    }
}

/// Structural notifications for this tick. Ordinary component updates use
/// `Changed<C>` queries. Records contain IDs and types, never component values.
pub struct Lifecycle<'u> {
    universe: &'u Universe,
}

impl<'u> Lifecycle<'u> {
    /// Iterates over structural transitions in command order.
    pub fn iter(&self) -> impl Iterator<Item = &'u LifecycleEvent> {
        self.universe.lifecycle()
    }
}

impl SystemParam for Lifecycle<'_> {
    type Item<'u> = Lifecycle<'u>;

    fn fetch<'u>(
        universe: &'u Universe,
        _: u64,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u> {
        Lifecycle { universe }
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// Entities whose component was removed during this tick, including despawns.
/// Every system can read the same records. Values are dropped on removal, and
/// records expire at the next tick. An entity may have re-added the component.
pub struct RemovedComponents<'u, C: Component> {
    universe: &'u Universe,
    _marker: PhantomData<C>,
}

impl<C: Component> RemovedComponents<'_, C> {
    /// Iterates over removed entity IDs, in command order.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.universe.lifecycle().filter_map(|event| match *event {
            LifecycleEvent::ComponentRemoved { entity, type_id }
                if type_id == TypeId::of::<C>() =>
            {
                Some(entity)
            }
            _ => None,
        })
    }
}

impl<C: Component> SystemParam for RemovedComponents<'_, C> {
    type Item<'u> = RemovedComponents<'u, C>;

    fn fetch<'u>(
        universe: &'u Universe,
        _: u64,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u> {
        RemovedComponents {
            universe,
            _marker: PhantomData,
        }
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// Time since the previous tick, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaTime(pub f64);

impl SystemParam for DeltaTime {
    type Item<'u> = Self;

    fn fetch<'u>(
        _: &'u Universe,
        _: u64,
        delta_time: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Self::Item<'u> {
        Self(delta_time)
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// A stateful system declaring its inputs in the trait's type parameter.
///
/// Systems run once per tick, even when their queries are empty. Queries are
/// independent collections; their contents are iterated explicitly. Mutable
/// edits are visible to later systems in the same tick; structural commands
/// apply on the next tick.
pub trait System<Params: SystemParam>: 'static {
    /// Returns the type name for diagnostics.
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Runs once with the fetched parameters.
    fn run(&mut self, params: Params::Item<'_>);
}

/// Converts a system or function into a registered system.
///
/// The marker distinguishes function signatures from [`System`] implementations
/// and is inferred by `UniverseBuilder::with_system`.
pub trait ToSystem<Marker>: 'static {
    /// Converts this value into the internal system representation.
    fn to_system(self) -> Box<dyn ErasedSystem>;
}

/// Internal execution interface used by the universe's system list.
#[doc(hidden)]
pub trait ErasedSystem: 'static {
    /// Runs a system once and advances its change-detection counter.
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>);
}

/// Marker distinguishing stateful systems from function signatures.
#[doc(hidden)]
pub struct SystemMarker<Params>(PhantomData<fn() -> Params>);

struct SystemRunner<S, Params> {
    system: S,
    last_run: u64,
    _marker: PhantomData<fn() -> Params>,
}

impl<S: System<Params>, Params: SystemParam + 'static> ErasedSystem for SystemRunner<S, Params> {
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>) {
        let this_run = universe.advance_change_tick();
        self.system
            .run(Params::fetch(universe, self.last_run, delta_time, commands));
        self.last_run = this_run;
    }
}

impl<S: System<Params>, Params: SystemParam + 'static> ToSystem<SystemMarker<Params>> for S {
    fn to_system(self) -> Box<dyn ErasedSystem> {
        Params::register_access(&mut QueryAccess::default());
        Box::new(SystemRunner::<_, Params> {
            system: self,
            last_run: 0,
            _marker: PhantomData,
        })
    }
}

struct FunctionSystem<Func>(Func);

macro_rules! impl_function_system {
    ($($param:ident: $arg:ident),*) => {
        impl<Func, $($param: SystemParam + 'static),*> ToSystem<fn($($param),*)> for Func
        where
            Func: FnMut($($param),*) + for<'u> FnMut($($param::Item<'u>),*) + 'static,
        {
            fn to_system(self) -> Box<dyn ErasedSystem> {
                <FunctionSystem<_> as ToSystem<SystemMarker<($($param,)*)>>>::to_system(FunctionSystem(self))
            }
        }

        impl<Func, $($param: SystemParam),*> System<($($param,)*)> for FunctionSystem<Func>
        where
            Func: FnMut($($param),*) + for<'u> FnMut($($param::Item<'u>),*) + 'static,
        {
            fn run(&mut self, ($($arg,)*): <($($param,)*) as SystemParam>::Item<'_>) {
                (self.0)($($arg),*);
            }
        }
    };
}

impl_function_system!();
impl_function_system!(A: a);
impl_function_system!(A: a, B: b);
impl_function_system!(A: a, B: b, C: c);
impl_function_system!(A: a, B: b, C: c, D: d);
