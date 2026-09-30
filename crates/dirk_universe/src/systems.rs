//! Systems run once per tick, ordered by the data they declare; see
//! [`schedule`](crate::schedule).
//!
//! Functions and stateful [`System`] implementations declare the data they
//! receive. The universe fetches those parameters before each run.

use std::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    CommandBuffer, Universe, WorldId,
    macros::sealed::Sealed,
    query::{Join, Query, QueryData, ReadOnlyQueryData, Scope, filter::QueryFilter},
    schedule::{Access, SystemConfig, SystemId},
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
/// [`DeltaTime`], or tuples of these.
pub trait SystemParam: Sealed {
    /// The value supplied while the universe is borrowed for this invocation.
    type Item<'u>;

    /// Fetches this parameter for one run.
    #[doc(hidden)]
    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u>;

    /// Registers this parameter's accesses before the system runs.
    #[doc(hidden)]
    fn register_access(access: &mut Access);
}

impl SystemParam for () {
    type Item<'u> = ();

    fn fetch<'u>(_: &SystemContext<'u>) -> Self::Item<'u> {}

    fn register_access(_: &mut Access) {}
}

macro_rules! impl_system_param_for_tuple {
    ($($ty:ident $binding:ident),+) => {
        impl<$($ty: SystemParam),+> SystemParam for ($($ty,)+) {
            type Item<'u> = ($($ty::Item<'u>,)+);

            fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
                ($($ty::fetch(context),)+)
            }

            fn register_access(access: &mut Access) {
                $($ty::register_access(access);)+
            }
        }
    };
}
all_tuples!(impl_system_param_for_tuple);

impl<D: QueryData, F: QueryFilter> Sealed for Query<'_, D, F> {}
impl<D: QueryData, F: QueryFilter> SystemParam for Query<'_, D, F>
where
    D::Liveness: Join<F::Liveness>,
{
    type Item<'u> = Query<'u, D, F>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Query::new(context.universe, context.last_run, Scope::Shared)
    }

    fn register_access(access: &mut Access) {
        D::register_access(access);
        F::register_access(access);
    }
}

/// Parameters that can be restricted to one world: queries and tuples of them.
pub trait WorldScoped: SystemParam {
    /// Fetches this parameter, seeing only the entities of `world`.
    #[doc(hidden)]
    fn fetch_in(universe: &Universe, last_run: u64, world: WorldId) -> Self::Item<'_>;
}

impl<D: QueryData, F: QueryFilter> WorldScoped for Query<'_, D, F>
where
    D::Liveness: Join<F::Liveness>,
{
    fn fetch_in(universe: &Universe, last_run: u64, world: WorldId) -> Self::Item<'_> {
        Query::new(universe, last_run, Scope::World(world))
    }
}

macro_rules! impl_world_scoped_for_tuple {
    ($($ty:ident $binding:ident),+) => {
        impl<$($ty: WorldScoped),+> WorldScoped for ($($ty,)+) {
            fn fetch_in(universe: &Universe, last_run: u64, world: WorldId) -> Self::Item<'_> {
                ($($ty::fetch_in(universe, last_run, world),)+)
            }
        }
    };
}
all_tuples!(impl_world_scoped_for_tuple);

/// World-scoped parameters that only read, which may be fetched several
/// times at once.
pub trait ReadOnlyWorldScoped: WorldScoped {}

impl<D: ReadOnlyQueryData, F: QueryFilter> ReadOnlyWorldScoped for Query<'_, D, F> where
    D::Liveness: Join<F::Liveness>
{
}

macro_rules! impl_read_only_world_scoped_for_tuple {
    ($($ty:ident $binding:ident),+) => {
        impl<$($ty: ReadOnlyWorldScoped),+> ReadOnlyWorldScoped for ($($ty,)+) {}
    };
}
all_tuples!(impl_read_only_world_scoped_for_tuple);

/// Runs `P`, a query or a tuple of queries, once per world. Each world's
/// queries see only that world's entities, including in isolated worlds,
/// which queries spanning worlds skip. Worlds destroyed since the system last
/// ran are visited too, so `Delta` reports their removals.
///
/// An entity moving between worlds appears as `Set` in its destination; its
/// previous world does not see it leave. World entities belong to no world,
/// so read world-level data with a query spanning worlds and
/// [`WorldId::entity`].
///
/// Mutable queries are iterated through `&mut`, so each world's data is
/// handed out once; iterating them through `&` does not compile:
///
/// ```compile_fail
/// # use dirk_universe::prelude::*;
/// # #[derive(Debug, Clone, Component)] struct Position(f64);
/// fn twice(worlds: PerWorld<Query<&mut Position>>) {
///     let first = worlds.iter().next();
///     let second = worlds.iter().next();
/// }
/// ```
///
/// ```
/// # use dirk_universe::prelude::*;
/// # #[derive(Debug, Clone, Component)] struct Position(f64);
/// fn separate(mut worlds: PerWorld<Query<(Entity, &mut Position)>>) {
///     for (world, mut positions) in &mut worlds {
///         for (entity, mut position) in &mut positions {
///             // Only entities of `world` interact here.
///         }
///     }
/// }
/// ```
pub struct PerWorld<'u, P: WorldScoped> {
    universe: &'u Universe,
    last_run: u64,
    worlds: Vec<WorldId>,
    _marker: PhantomData<fn() -> P>,
}

impl<P: WorldScoped> PerWorld<'_, P> {
    /// Iterates over worlds, in creation order, with `P` scoped to each.
    /// Items borrow this `PerWorld`, so each world's data is handed out once.
    #[must_use]
    pub fn iter_mut(&mut self) -> PerWorldIter<'_, P> {
        PerWorldIter::new(self.universe, self.last_run, self.worlds.clone())
    }
}

impl<P: ReadOnlyWorldScoped> PerWorld<'_, P> {
    /// Iterates over worlds, in creation order, with read-only `P` scoped to
    /// each.
    #[must_use]
    pub fn iter(&self) -> PerWorldIter<'_, P> {
        PerWorldIter::new(self.universe, self.last_run, self.worlds.clone())
    }
}

impl<'u, P: WorldScoped> IntoIterator for PerWorld<'u, P> {
    type Item = (WorldId, P::Item<'u>);
    type IntoIter = PerWorldIter<'u, P>;

    fn into_iter(self) -> Self::IntoIter {
        PerWorldIter::new(self.universe, self.last_run, self.worlds)
    }
}

impl<'a, P: WorldScoped> IntoIterator for &'a mut PerWorld<'_, P> {
    type Item = (WorldId, P::Item<'a>);
    type IntoIter = PerWorldIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, P: ReadOnlyWorldScoped> IntoIterator for &'a PerWorld<'_, P> {
    type Item = (WorldId, P::Item<'a>);
    type IntoIter = PerWorldIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over the worlds of a [`PerWorld`].
pub struct PerWorldIter<'a, P: WorldScoped> {
    universe: &'a Universe,
    last_run: u64,
    worlds: std::vec::IntoIter<WorldId>,
    _marker: PhantomData<fn() -> P>,
}

impl<'a, P: WorldScoped> PerWorldIter<'a, P> {
    fn new(universe: &'a Universe, last_run: u64, worlds: Vec<WorldId>) -> Self {
        Self {
            universe,
            last_run,
            worlds: worlds.into_iter(),
            _marker: PhantomData,
        }
    }
}

impl<'a, P: WorldScoped> Iterator for PerWorldIter<'a, P> {
    type Item = (WorldId, P::Item<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let world = self.worlds.next()?;
        Some((world, P::fetch_in(self.universe, self.last_run, world)))
    }
}

impl<P: WorldScoped> Sealed for PerWorld<'_, P> {}
impl<P: WorldScoped> SystemParam for PerWorld<'_, P> {
    type Item<'u> = PerWorld<'u, P>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        let universe = context.universe;
        let mut worlds: Vec<_> = universe
            .worlds()
            .map(|world| world.id())
            .chain(universe.worlds_removed_since(context.last_run))
            .collect();
        worlds.sort_by_key(|world| world.raw());
        worlds.dedup();
        PerWorld {
            universe,
            last_run: context.last_run,
            worlds,
            _marker: PhantomData,
        }
    }

    fn register_access(access: &mut Access) {
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

impl Sealed for Commands<'_> {}
impl SystemParam for Commands<'_> {
    type Item<'u> = Commands<'u>;

    fn fetch<'u>(context: &SystemContext<'u>) -> Self::Item<'u> {
        Commands {
            buffer: context.commands.borrow_mut(),
        }
    }

    fn register_access(access: &mut Access) {
        access.commands();
    }
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

    fn register_access(_: &mut Access) {}
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

    /// Runs once with the fetched parameters.
    fn run(&mut self, params: Self::Params<'_>);

    /// Names this system type for [`IntoSystem::before`] and [`IntoSystem::after`].
    #[must_use]
    fn label() -> Label<Self>
    where
        Self: Sized,
    {
        Label(PhantomData)
    }
}

/// Names a [`System`] type in [`IntoSystem::before`] and [`IntoSystem::after`].
/// Functions and closures name themselves instead.
pub struct Label<S>(PhantomData<fn() -> S>);

/// A registered system that other systems can be ordered against: a function,
/// a closure, or a [`Label`] naming a [`System`] type.
pub trait SystemLabel<Marker> {
    /// Returns the system's identity.
    #[doc(hidden)]
    fn id(&self) -> SystemId;
}

impl<S: System> SystemLabel<SystemMarker> for Label<S> {
    fn id(&self) -> SystemId {
        SystemId::of::<S>()
    }
}

/// Converts a [`System`] or a function into a registered system.
///
/// Functions and `FnMut` closures qualify when every argument is a
/// [`SystemParam`]. The marker distinguishes these cases and is inferred by
/// `UniverseBuilder::with_system`.
pub trait IntoSystem<Marker>: Sized + 'static {
    /// Converts this value into a system ready to be scheduled.
    #[doc(hidden)]
    fn into_config(self) -> SystemConfig;

    /// Converts this value into the internal system representation.
    #[doc(hidden)]
    fn into_system(self) -> Box<dyn ErasedSystem> {
        self.into_config().system
    }

    /// Runs this system before `other`, overriding any order derived from
    /// the data they access.
    fn before<M>(self, other: impl SystemLabel<M>) -> SystemConfig {
        let mut config = self.into_config();
        config.before.push(other.id());
        config
    }

    /// Runs this system after `other`, overriding any order derived from
    /// the data they access.
    fn after<M>(self, other: impl SystemLabel<M>) -> SystemConfig {
        let mut config = self.into_config();
        config.after.push(other.id());
        config
    }
}

impl IntoSystem<()> for SystemConfig {
    fn into_config(self) -> SystemConfig {
        self
    }
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

    fn register_access(access: &mut Access);
}

/// Tracks the change-detection counter shared by every kind of system.
struct Runner<R> {
    inner: R,
    last_run: u64,
}

impl<R: Run> Runner<R> {
    fn config(inner: R, id: SystemId) -> SystemConfig {
        let mut access = Access::default();
        R::register_access(&mut access);
        SystemConfig {
            system: Box::new(Self { inner, last_run: 0 }),
            id,
            access,
            before: Vec::new(),
            after: Vec::new(),
        }
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

    fn register_access(access: &mut Access) {
        S::Params::<'static>::register_access(access);
    }
}

/// Marker distinguishing [`System`] implementations from functions.
#[doc(hidden)]
pub struct SystemMarker;

impl<S: System> IntoSystem<SystemMarker> for S {
    fn into_config(self) -> SystemConfig {
        Runner::config(StructSystem(self), SystemId::of::<S>())
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

            fn register_access(access: &mut Access) {
                <($($ty,)*) as SystemParam>::register_access(access);
            }
        }

        impl<Func, $($ty: SystemParam + 'static),*> IntoSystem<fn($($ty),*)> for Func
        where
            Func: FnMut($($ty),*) + for<'u> FnMut($($ty::Item<'u>),*) + 'static,
        {
            fn into_config(self) -> SystemConfig {
                Runner::config(
                    FunctionSystem::<_, ($($ty,)*)> {
                        func: self,
                        _marker: PhantomData,
                    },
                    SystemId::of::<Func>(),
                )
            }
        }

        impl<Func, $($ty: SystemParam + 'static),*> SystemLabel<fn($($ty),*)> for Func
        where
            Func: FnMut($($ty),*) + for<'u> FnMut($($ty::Item<'u>),*) + 'static,
        {
            fn id(&self) -> SystemId {
                SystemId::of::<Func>()
            }
        }
    };
}
impl_function_system!();
all_tuples!(impl_function_system);
