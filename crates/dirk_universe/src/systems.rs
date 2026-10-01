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
    macros::sealed::Sealed,
    query::{Query, QueryAccess, QueryData, filter::QueryFilter},
};

/// Everything a system parameter can be fetched from during one run.
#[doc(hidden)]
pub struct SystemContext<'u> {
    universe: &'u Universe,
    last_run: u64,
    delta_time: f64,
    commands: &'u RefCell<CommandBuffer>,
}

/// Data supplied to one system invocation: queries, [`Commands`],
/// [`DeltaTime`], lifecycle records, or tuples of these.
pub trait SystemParam: Sealed {
    /// The value supplied while the universe is borrowed for this invocation.
    type Item<'u>;

    /// Fetches this parameter for one run.
    #[doc(hidden)]
    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u>;

    /// Registers this parameter's accesses before the system runs.
    #[doc(hidden)]
    fn register_access(access: &mut QueryAccess);
}

impl SystemParam for () {
    type Item<'u> = ();

    fn fetch<'u>(_: &SystemContext<'u>) -> Self::Item<'u> {}

    fn register_access(_: &mut QueryAccess) {}
}

macro_rules! impl_system_param_for_tuple {
    ($($ty:ident $binding:ident),+) => {
        impl<$($ty: SystemParam),+> SystemParam for ($($ty,)+) {
            type Item<'u> = ($($ty::Item<'u>,)+);

            fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
                ($($ty::fetch(context),)+)
            }

            fn register_access(access: &mut QueryAccess) {
                $($ty::register_access(access);)+
            }
        }
    };
}
all_tuples!(impl_system_param_for_tuple);

impl<D: QueryData, F: QueryFilter> Sealed for Query<'_, D, F> {}
impl<D: QueryData, F: QueryFilter> SystemParam for Query<'_, D, F> {
    type Item<'u> = Query<'u, D, F>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Query::new(context.universe, context.last_run)
    }

    fn register_access(access: &mut QueryAccess) {
        D::register_access(access);
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

impl Sealed for Commands<'_> {}
impl SystemParam for Commands<'_> {
    type Item<'u> = Commands<'u>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Commands {
            buffer: context.commands.borrow_mut(),
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

impl Sealed for Lifecycle<'_> {}
impl SystemParam for Lifecycle<'_> {
    type Item<'u> = Lifecycle<'u>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Lifecycle {
            universe: context.universe,
        }
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

impl<C: Component> Sealed for RemovedComponents<'_, C> {}
impl<C: Component> SystemParam for RemovedComponents<'_, C> {
    type Item<'u> = RemovedComponents<'u, C>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        RemovedComponents {
            universe: context.universe,
            _marker: PhantomData,
        }
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// Time since the previous tick, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaTime(pub f64);

impl Sealed for DeltaTime {}
impl SystemParam for DeltaTime {
    type Item<'u> = Self;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Self(context.delta_time)
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// A stateful system. `Params` names its inputs once, with `'u` standing for
/// the universe borrow of one run:
///
/// ```
/// # use dirk_universe::prelude::*;
/// # #[derive(Debug, Component)] struct Position(f64);
/// struct Drift { speed: f64 }
///
/// impl System for Drift {
///     type Params<'u> = (Query<'u, &'u mut Position>, DeltaTime);
///
///     fn run(&mut self, (mut positions, DeltaTime(dt)): Self::Params<'_>) {
///         for mut position in &mut positions {
///             position.0 += self.speed * dt;
///         }
///     }
/// }
/// ```
///
/// Systems run once per tick, even when their queries are empty. Mutable
/// edits are visible to later systems in the same tick; structural commands
/// apply on the next tick.
pub trait System: 'static {
    /// The parameters fetched for each run.
    type Params<'u>: SystemParam<Item<'u> = Self::Params<'u>>;

    /// Returns the type name for diagnostics.
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Runs once with the fetched parameters.
    fn run(&mut self, params: Self::Params<'_>);
}

/// Converts a [`System`] or a function into a registered system.
///
/// Functions and `FnMut` closures qualify when every argument is a
/// [`SystemParam`]. The marker distinguishes these cases and is inferred by
/// `UniverseBuilder::with_system`.
pub trait IntoSystem<Marker>: 'static {
    /// Converts this value into the internal system representation.
    #[doc(hidden)]
    fn into_system(self) -> Box<dyn ErasedSystem>;
}

/// Internal execution interface used by the universe's system list.
#[doc(hidden)]
pub trait ErasedSystem: 'static {
    /// Runs a system once and advances its change-detection counter.
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>);
}

/// Fetches parameters from a context and invokes a system's body.
trait Run: 'static {
    fn run(&mut self, context: &SystemContext<'_>);

    fn register_access(access: &mut QueryAccess);
}

/// Tracks the change-detection counter shared by every kind of system.
struct Runner<R> {
    inner: R,
    last_run: u64,
}

impl<R: Run> Runner<R> {
    fn boxed(inner: R) -> Box<dyn ErasedSystem> {
        R::register_access(&mut QueryAccess::default());
        Box::new(Self { inner, last_run: 0 })
    }
}

impl<R: Run> ErasedSystem for Runner<R> {
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>) {
        let this_run = universe.advance_change_tick();
        self.inner.run(&SystemContext {
            universe,
            last_run: self.last_run,
            delta_time,
            commands,
        });
        self.last_run = this_run;
    }
}

/// Adapts a [`System`]. Its `'static` parameters describe access for every borrow.
struct StructSystem<S>(S);

impl<S: System> Run for StructSystem<S> {
    fn run(&mut self, context: &SystemContext<'_>) {
        self.0.run(S::Params::fetch(context));
    }

    fn register_access(access: &mut QueryAccess) {
        S::Params::<'static>::register_access(access);
    }
}

/// Marker distinguishing [`System`] implementations from functions.
#[doc(hidden)]
pub struct SystemMarker;

impl<S: System> IntoSystem<SystemMarker> for S {
    fn into_system(self) -> Box<dyn ErasedSystem> {
        Runner::boxed(StructSystem(self))
    }
}

/// Adapts a function whose arguments are the system parameters `Params`.
struct FunctionSystem<Func, Params> {
    func: Func,
    _marker: PhantomData<fn() -> Params>,
}

macro_rules! impl_function_system {
    ($($ty:ident $binding:ident),*) => {
        impl<Func, $($ty: SystemParam + 'static),*> Run for FunctionSystem<Func, ($($ty,)*)>
        where
            Func: FnMut($($ty),*) + for<'u> FnMut($($ty::Item<'u>),*) + 'static,
        {
            fn run(&mut self, context: &SystemContext<'_>) {
                let ($($binding,)*) = <($($ty,)*) as SystemParam>::fetch(context);
                (self.func)($($binding),*);
            }

            fn register_access(access: &mut QueryAccess) {
                <($($ty,)*) as SystemParam>::register_access(access);
            }
        }

        impl<Func, $($ty: SystemParam + 'static),*> IntoSystem<fn($($ty),*)> for Func
        where
            Func: FnMut($($ty),*) + for<'u> FnMut($($ty::Item<'u>),*) + 'static,
        {
            fn into_system(self) -> Box<dyn ErasedSystem> {
                Runner::boxed(FunctionSystem::<_, ($($ty,)*)> {
                    func: self,
                    _marker: PhantomData,
                })
            }
        }
    };
}
impl_function_system!();
all_tuples!(impl_function_system);
