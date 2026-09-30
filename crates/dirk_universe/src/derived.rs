//! Components computed from other components.
//!
//! A derived component is a function of an entity's other components:
//!
//! ```
//! # use dirk_universe::prelude::*;
//! #[derive(Debug, Component)]
//! struct Velocity(f64);
//! #[derive(Debug, Component)]
//! struct Mass(f64);
//! #[derive(Debug, PartialEq, Component)]
//! #[component(read_only)]
//! struct Momentum(f64);
//!
//! fn momentum(velocity: &Velocity, mass: Option<&Mass>) -> Momentum {
//!     Momentum(velocity.0 * mass.map_or(1.0, |mass| mass.0))
//! }
//!
//! let universe = Universe::builder().with_derived(momentum).build();
//! # assert!(universe.is_ok());
//! ```
//!
//! Every entity whose inputs are present gets the output; `Option<&C>` inputs
//! are not required. The function reruns only for entities whose inputs
//! changed since it last ran, and the output is written only when its value
//! changes, so `Changed<Momentum>` stays precise. When an input disappears,
//! so does the output. The output is read-only, so nothing else can write it,
//! and each component has one derivation. Derivations are ordered by the data
//! they access alone: after the writers of their inputs, before the readers
//! of their output. A mutable output does not compile:
//!
//! ```compile_fail
//! # use dirk_universe::prelude::*;
//! # #[derive(Debug, Component)] struct Velocity(f64);
//! #[derive(Debug, PartialEq, Component)]
//! struct Speed(f64);
//!
//! let universe = Universe::builder().with_derived(|velocity: &Velocity| Speed(velocity.0));
//! ```

use std::{any::TypeId, cell::RefCell, marker::PhantomData};

use crate::{
    CommandBuffer, Entity, Universe,
    components::{Component, ReadOnly},
    macros::sealed::Sealed,
    query::{QueryData, ReadOnlyQueryData},
    schedule::{Access, SystemConfig, SystemId},
    systems::ErasedSystem,
};

/// An argument of a derived component's function: `&C`, or `Option<&C>` for
/// an input that is not required.
pub trait DerivedArg: Sealed {
    /// The query data fetching this argument.
    #[doc(hidden)]
    type Data: ReadOnlyQueryData;

    /// The argument borrowed from the fetched data.
    #[doc(hidden)]
    type Arg<'a>;

    /// Borrows the argument from fetched data.
    #[doc(hidden)]
    fn arg<'a>(item: &'a <Self::Data as QueryData>::Item<'_>) -> Self::Arg<'a>;

    /// Returns whether this input changed for `entity` after `last_run`.
    #[doc(hidden)]
    fn changed(entity: Entity, universe: &Universe, last_run: u64) -> bool;
}

impl<C: Component> DerivedArg for &C {
    type Data = &'static C;
    type Arg<'a> = &'a C;

    fn arg<'a>(item: &'a <Self::Data as QueryData>::Item<'_>) -> Self::Arg<'a> {
        item
    }

    fn changed(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.changed::<C>(entity, last_run)
    }
}

impl<C: Component> DerivedArg for Option<&C> {
    type Data = Option<&'static C>;
    type Arg<'a> = Option<&'a C>;

    fn arg<'a>(item: &'a <Self::Data as QueryData>::Item<'_>) -> Self::Arg<'a> {
        item.as_deref()
    }

    fn changed(entity: Entity, universe: &Universe, last_run: u64) -> bool {
        universe.components.changed::<C>(entity, last_run)
            || universe
                .removed_since::<C>(last_run)
                .any(|removed| removed == entity)
    }
}

/// A function computing a derived component, registered with
/// `UniverseBuilder::with_derived`. The marker is inferred.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot derive a component",
    note = "a derivation takes `&C` or `Option<&C>` arguments and returns a \
            `#[component(read_only)]` component implementing `PartialEq`"
)]
pub trait IntoDerived<Marker>: 'static {
    /// Converts this function into a scheduled derivation.
    #[doc(hidden)]
    fn into_config(self) -> SystemConfig;
}

/// Computes `Out` from the arguments `Args` for every matching entity.
struct Derived<Func, Args, Out> {
    func: Func,
    last_run: u64,
    _marker: PhantomData<fn(Args) -> Out>,
}

macro_rules! impl_derived {
    ($($ty:ident $binding:ident),+) => {
        impl<Func, Out, $($ty: DerivedArg + 'static),+> IntoDerived<fn($($ty),+) -> Out> for Func
        where
            Func: Fn($($ty),+) -> Out + for<'a> Fn($($ty::Arg<'a>),+) -> Out + 'static,
            Out: Component<Mutability = ReadOnly> + PartialEq,
        {
            fn into_config(self) -> SystemConfig {
                let mut access = Access::default();
                $(<$ty::Data as QueryData>::register_access(&mut access);)+
                access.write::<Out>();
                SystemConfig {
                    system: Box::new(Derived::<Func, ($($ty,)+), Out> {
                        func: self,
                        last_run: 0,
                        _marker: PhantomData,
                    }),
                    id: SystemId::of::<Func>(),
                    access,
                    derives: Some(SystemId::of::<Out>()),
                    before: Vec::new(),
                    after: Vec::new(),
                }
            }
        }

        impl<Func, Out, $($ty: DerivedArg + 'static),+> ErasedSystem
            for Derived<Func, ($($ty,)+), Out>
        where
            Func: Fn($($ty),+) -> Out + for<'a> Fn($($ty::Arg<'a>),+) -> Out + 'static,
            Out: Component<Mutability = ReadOnly> + PartialEq,
        {
            fn run(&mut self, universe: &mut Universe, _: f64, _: &RefCell<CommandBuffer>) {
                let this_run = universe.advance_change_tick();
                let last_run = self.last_run;
                let mut writes = Vec::new();
                let mut removals = Vec::new();
                for &entity in &universe.alive {
                    let has_output = universe.components.contains(entity, TypeId::of::<Out>());
                    let inputs = ($(<$ty::Data as QueryData>::fetch(entity, universe, last_run),)+);
                    let ($(Some($binding),)+) = inputs else {
                        if has_output {
                            removals.push(entity);
                        }
                        continue;
                    };
                    if has_output $(&& !$ty::changed(entity, universe, last_run))+ {
                        continue;
                    }
                    let value = (self.func)($($ty::arg(&$binding)),+);
                    let current = universe.component::<Out>(entity);
                    if current.is_none_or(|current| *current != value) {
                        writes.push((entity, value));
                    }
                }
                for (entity, value) in writes {
                    universe.set_component(entity, Box::new(value));
                }
                for entity in removals {
                    let location = universe.live_location(entity);
                    universe.remove_component(entity, TypeId::of::<Out>(), location);
                }
                self.last_run = this_run;
            }
        }
    };
}
all_tuples!(impl_derived);
