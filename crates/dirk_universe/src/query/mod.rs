//! Typed iteration and lookups over live entities across all worlds.
//! Tuples fetch components together; filter tuples combine conditions with AND.

pub mod filter;

use self::filter::Filter;
use crate::{
    Entity, Universe, WorldId,
    components::{Component, ComponentMut},
};
use std::{
    any::{TypeId, type_name},
    cell::Ref,
    collections::{HashMap, hash_map::Keys},
    marker::PhantomData,
};

/// A system's view of matching entities. Systems run once per tick, even when
/// their queries are empty. Include `Entity` in `P` to fetch entity IDs.
///
/// `&C` yields a shared borrow; `&mut C` yields a mutable borrow that marks the
/// component changed on mutable dereference. Neither requires `C: Clone`.
pub struct Query<'u, P: QueryParameter, F: Filter = ()> {
    universe: &'u Universe,
    last_run: u64,
    _marker: PhantomData<fn() -> (P, F)>,
}

impl<'u, P: QueryParameter, F: Filter> Query<'u, P, F> {
    pub(crate) fn for_system(universe: &'u Universe, last_run: u64) -> Self {
        Self {
            universe,
            last_run,
            _marker: PhantomData,
        }
    }

    fn iterator(&self, world: Option<WorldId>) -> QueryIter<'_, P, F> {
        QueryIter {
            entities: self.universe.entities.keys(),
            universe: self.universe,
            last_run: self.last_run,
            world,
            _marker: PhantomData,
        }
    }

    fn fetch(&self, entity: Entity) -> Option<P::Item<'_>> {
        if !self.universe.is_alive(entity) || !F::matches(entity, self.universe, self.last_run) {
            return None;
        }
        P::from_entity(entity, self.universe)
    }

    /// Iterates over matching entities, allowing component mutation.
    pub fn iter_mut(&mut self) -> QueryIter<'_, P, F> {
        self.iterator(None)
    }

    /// Iterates over matching entities in one world, allowing mutation.
    pub fn iter_in_world_mut(&mut self, world: WorldId) -> QueryIter<'_, P, F> {
        self.iterator(Some(world))
    }

    /// Borrows one matching entity's data for mutation.
    pub fn get_mut(&mut self, entity: Entity) -> Option<P::Item<'_>> {
        self.fetch(entity)
    }
}

impl<'u, P: ReadOnlyQueryParameter, F: Filter> Query<'u, P, F> {
    /// Creates a read-only query outside a system. `Added` and `Changed` match
    /// all present components here, since there is no previous system run.
    #[must_use]
    pub fn new(universe: &'u Universe) -> Self {
        Self::for_system(universe, 0)
    }

    /// Iterates over matching entities.
    #[must_use]
    pub fn iter(&self) -> QueryIter<'_, P, F> {
        self.iterator(None)
    }

    /// Iterates over matching entities in one world.
    #[must_use]
    pub fn iter_in_world(&self, world: WorldId) -> QueryIter<'_, P, F> {
        self.iterator(Some(world))
    }

    /// Borrows one entity if it is alive and matches this query.
    #[must_use]
    pub fn get(&self, entity: Entity) -> Option<P::Item<'_>> {
        self.fetch(entity)
    }
}

/// An iterator over a query's matching data. Entity order is unspecified.
pub struct QueryIter<'u, P: QueryParameter, F: Filter> {
    entities: Keys<'u, Entity, WorldId>,
    universe: &'u Universe,
    last_run: u64,
    world: Option<WorldId>,
    _marker: PhantomData<fn() -> (P, F)>,
}

impl<'u, P: QueryParameter, F: Filter> Iterator for QueryIter<'u, P, F> {
    type Item = P::Item<'u>;

    fn next(&mut self) -> Option<Self::Item> {
        for &entity in self.entities.by_ref() {
            if self
                .world
                .is_some_and(|world| !self.universe.is_in_world(world, entity))
                || !F::matches(entity, self.universe, self.last_run)
            {
                continue;
            }
            if let Some(item) = P::from_entity(entity, self.universe) {
                return Some(item);
            }
        }
        None
    }
}

impl<'q, P: ReadOnlyQueryParameter, F: Filter> IntoIterator for &'q Query<'_, P, F> {
    type Item = P::Item<'q>;
    type IntoIter = QueryIter<'q, P, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'q, P: QueryParameter, F: Filter> IntoIterator for &'q mut Query<'_, P, F> {
    type Item = P::Item<'q>;
    type IntoIter = QueryIter<'q, P, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Describes the data fetched for each entity matched by a query.
pub trait QueryParameter: Sized {
    /// The concrete value borrowed or produced for one matched entity.
    type Item<'u>;

    /// Builds this parameter value for `entity`, returning `None` if it does not match.
    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>>;

    /// Registers component access for conflict validation.
    fn register_access(access: &mut QueryAccess);
}

/// Tracks the component access declared by one system.
#[doc(hidden)]
#[derive(Default)]
pub struct QueryAccess {
    components: HashMap<TypeId, AccessKind>,
    commands: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AccessKind {
    Read,
    Write,
}

impl QueryAccess {
    /// Registers a read of `C`.
    ///
    /// # Panics
    /// Panics when this system also declares mutable access to `C`.
    pub fn read<C: Component>(&mut self) {
        assert!(
            self.components.get(&TypeId::of::<C>()) != Some(&AccessKind::Write),
            "system reads and writes {} through overlapping queries",
            type_name::<C>()
        );
        self.components
            .entry(TypeId::of::<C>())
            .or_insert(AccessKind::Read);
    }

    /// Registers a write of `C`.
    ///
    /// # Panics
    /// Panics when this system already declares any access to `C`.
    pub fn write<C: Component>(&mut self) {
        assert!(
            self.components
                .insert(TypeId::of::<C>(), AccessKind::Write)
                .is_none(),
            "system accesses {} through overlapping mutable queries",
            type_name::<C>()
        );
    }

    /// Registers structural command access.
    ///
    /// # Panics
    /// Panics when this system declares `Commands` more than once.
    pub fn commands(&mut self) {
        assert!(
            !std::mem::replace(&mut self.commands, true),
            "system requests Commands more than once"
        );
    }
}

/// Query parameters that do not require mutable access.
pub trait ReadOnlyQueryParameter: QueryParameter {}

impl QueryParameter for () {
    type Item<'u> = ();

    fn from_entity(_: Entity, _: &Universe) -> Option<Self::Item<'_>> {
        Some(())
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryParameter for () {}

macro_rules! impl_query_parameter_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> QueryParameter for ($($name,)+)
        where
            $($name: QueryParameter),+
        {
            type Item<'u> = ($($name::Item<'u>,)+);

            fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
                Some(($($name::from_entity(entity, universe)?,)+))
            }

            fn register_access(access: &mut QueryAccess) {
                $($name::register_access(access);)+
            }
        }

        impl<$($name: ReadOnlyQueryParameter),+> ReadOnlyQueryParameter for ($($name,)+) {}
    };
}

impl_query_parameter_for_tuple!(A, B);
impl_query_parameter_for_tuple!(A, B, C);
impl_query_parameter_for_tuple!(A, B, C, D);
impl_query_parameter_for_tuple!(A, B, C, D, E);
impl_query_parameter_for_tuple!(A, B, C, D, E, F);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G);
impl_query_parameter_for_tuple!(A, B, C, D, E, F, G, H);

impl<C: Component> QueryParameter for &C {
    type Item<'u> = Ref<'u, C>;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }

    fn register_access(access: &mut QueryAccess) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryParameter for &C {}

impl<C: Component> QueryParameter for &mut C {
    type Item<'u> = ComponentMut<'u, C>;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe
            .components
            .get_mut::<C>(entity, universe.change_tick.get())
    }

    fn register_access(access: &mut QueryAccess) {
        access.write::<C>();
    }
}

impl QueryParameter for Entity {
    type Item<'u> = Entity;

    fn from_entity(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe.is_alive(entity).then_some(entity)
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryParameter for Entity {}
