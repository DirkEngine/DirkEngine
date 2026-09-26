//! Systems run in registration order, once per entity matching their queries.
//!
//! Functions and stateful [`System`] implementations declare the data they
//! receive. The universe fetches those parameters before each run.

use std::{
    any::type_name,
    cell::{RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    CommandBuffer, Entity, Universe,
    changes::{Change, ComponentChange},
    components::Component,
    query::{
        Query, QueryAccess, QueryParameter, QueryView, ReadOnlyQueryParameter, filter::Filter,
    },
};

/// Data supplied to one system invocation.
pub trait SystemParam {
    /// Whether this parameter requires one invocation per matching entity.
    const PER_ENTITY: bool = false;

    /// The value supplied while the universe is borrowed for this invocation.
    type Item<'u>;

    /// Fetches a parameter, or skips the invocation if the entity does not match.
    fn fetch<'u>(
        universe: &'u Universe,
        entity: Option<Entity>,
        delta_time: f64,
        commands: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>>;

    /// Registers this parameter's accesses before the system runs.
    fn register_access(access: &mut QueryAccess);

    /// Calls a system for each match, or once when it has no entity query.
    fn for_each(
        universe: &Universe,
        delta_time: f64,
        commands: &RefCell<CommandBuffer>,
        mut run: impl FnMut(Self::Item<'_>),
    ) {
        let mut invoke = |entity| {
            if let Some(params) = Self::fetch(universe, entity, delta_time, commands) {
                run(params);
            }
        };
        if Self::PER_ENTITY {
            for entity in universe.entities.keys().copied() {
                invoke(Some(entity));
            }
        } else {
            invoke(None);
        }
    }
}

impl SystemParam for () {
    type Item<'u> = ();

    fn fetch<'u>(
        _: &'u Universe,
        _: Option<Entity>,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Some(())
    }

    fn register_access(_: &mut QueryAccess) {}
}

macro_rules! impl_system_param_tuple {
    ($($param:ident),+) => {
        impl<$($param: SystemParam),+> SystemParam for ($($param,)+) {
            const PER_ENTITY: bool = false $(|| $param::PER_ENTITY)+;
            type Item<'u> = ($($param::Item<'u>,)+);

            fn fetch<'u>(
                universe: &'u Universe,
                entity: Option<Entity>,
                delta_time: f64,
                commands: &'u RefCell<CommandBuffer>,
            ) -> Option<Self::Item<'u>> {
                Some(($($param::fetch(universe, entity, delta_time, commands)?,)+))
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
    const PER_ENTITY: bool = true;
    type Item<'u> = Query<'u, P, F>;

    fn fetch<'u>(
        universe: &'u Universe,
        entity: Option<Entity>,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Query::matches(entity?, universe)
    }

    fn register_access(access: &mut QueryAccess) {
        P::register_access(access);
    }
}

impl<P: ReadOnlyQueryParameter, F: Filter> SystemParam for QueryView<'_, P, F> {
    type Item<'u> = QueryView<'u, P, F>;

    fn fetch<'u>(
        universe: &'u Universe,
        _: Option<Entity>,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Some(QueryView::new(universe))
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
        _: Option<Entity>,
        _: f64,
        commands: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Some(Commands {
            buffer: commands.borrow_mut(),
        })
    }

    fn register_access(access: &mut QueryAccess) {
        access.commands();
    }
}

/// Ordered changes applied at the start of this tick.
pub struct Changes<'u> {
    universe: &'u Universe,
}

impl<'u> Changes<'u> {
    /// Iterates over every change in command order.
    pub fn iter(&self) -> impl Iterator<Item = &'u Change> {
        self.universe.changes()
    }

    /// Iterates over changes for one component type.
    pub fn components<C: Component>(&self) -> impl Iterator<Item = ComponentChange<'u, C>> {
        self.universe.component_changes::<C>()
    }
}

impl SystemParam for Changes<'_> {
    type Item<'u> = Changes<'u>;

    fn fetch<'u>(
        universe: &'u Universe,
        _: Option<Entity>,
        _: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Some(Changes { universe })
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
        _: Option<Entity>,
        delta_time: f64,
        _: &'u RefCell<CommandBuffer>,
    ) -> Option<Self::Item<'u>> {
        Some(Self(delta_time))
    }

    fn register_access(_: &mut QueryAccess) {}
}

/// A stateful system declaring its inputs in the trait's type parameter.
///
/// Each query supplies one matching entity. Multiple queries must all match the
/// same entity. Systems without `Query` parameters run once per tick, even in an
/// empty universe. Mutable edits are visible to later systems in the same tick;
/// structural commands apply on the next tick.
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
    /// Runs a system for every match of its declared parameters.
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>);
}

/// Marker distinguishing stateful systems from function signatures.
#[doc(hidden)]
pub struct SystemMarker<Params>(PhantomData<fn() -> Params>);

struct SystemRunner<S, Params> {
    system: S,
    _marker: PhantomData<fn() -> Params>,
}

impl<S: System<Params>, Params: SystemParam + 'static> ErasedSystem for SystemRunner<S, Params> {
    fn run(&mut self, universe: &Universe, delta_time: f64, commands: &RefCell<CommandBuffer>) {
        Params::for_each(universe, delta_time, commands, |params| {
            self.system.run(params);
        });
    }
}

impl<S: System<Params>, Params: SystemParam + 'static> ToSystem<SystemMarker<Params>> for S {
    fn to_system(self) -> Box<dyn ErasedSystem> {
        Params::register_access(&mut QueryAccess::default());
        Box::new(SystemRunner::<_, Params> {
            system: self,
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
