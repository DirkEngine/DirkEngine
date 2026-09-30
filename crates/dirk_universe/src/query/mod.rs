//! Typed iteration and lookups over live entities across all worlds.
//! Tuples fetch components together; filter tuples combine conditions with AND.

pub mod filter;

use self::filter::QueryFilter;
use crate::{
    Entity, Universe, WorldId,
    components::{Component, ComponentMut},
    macros::sealed::Sealed,
};
use std::{
    any::{TypeId, type_name},
    cell::Ref,
    collections::{HashMap, hash_map::Keys},
    marker::PhantomData,
};

/// A system's view of matching entities. Systems run once per tick, even when
/// their queries are empty. Include `Entity` in `D` to fetch entity IDs.
///
/// `&C` yields a shared borrow; `&mut C` yields a mutable borrow that marks the
/// component changed on mutable dereference. `Option<D>` fetches `D` when
/// present without requiring it. Neither requires `C: Clone`.
pub struct Query<'u, D: QueryData, F: QueryFilter = ()> {
    universe: &'u Universe,
    last_run: u64,
    _marker: PhantomData<fn() -> (D, F)>,
}

impl<'u, D: QueryData, F: QueryFilter> Query<'u, D, F> {
    pub(crate) fn new(universe: &'u Universe, last_run: u64) -> Self {
        Self {
            universe,
            last_run,
            _marker: PhantomData,
        }
    }

    fn iterator(&self, world: Option<WorldId>) -> QueryIter<'_, D, F> {
        QueryIter {
            entities: self.universe.entities.keys(),
            universe: self.universe,
            last_run: self.last_run,
            world,
            _marker: PhantomData,
        }
    }

    fn fetch(&self, entity: Entity) -> Option<D::Item<'_>> {
        if !self.universe.is_alive(entity) || !F::matches(entity, self.universe, self.last_run) {
            return None;
        }
        D::fetch(entity, self.universe)
    }

    /// Iterates over matching entities, allowing component mutation.
    pub fn iter_mut(&mut self) -> QueryIter<'_, D, F> {
        self.iterator(None)
    }

    /// Iterates over matching entities in one world, allowing mutation.
    pub fn iter_in_world_mut(&mut self, world: WorldId) -> QueryIter<'_, D, F> {
        self.iterator(Some(world))
    }

    /// Borrows one matching entity's data for mutation.
    pub fn get_mut(&mut self, entity: Entity) -> Option<D::Item<'_>> {
        self.fetch(entity)
    }
}

impl<D: ReadOnlyQueryData, F: QueryFilter> Query<'_, D, F> {
    /// Iterates over matching entities.
    #[must_use]
    pub fn iter(&self) -> QueryIter<'_, D, F> {
        self.iterator(None)
    }

    /// Iterates over matching entities in one world.
    #[must_use]
    pub fn iter_in_world(&self, world: WorldId) -> QueryIter<'_, D, F> {
        self.iterator(Some(world))
    }

    /// Borrows one entity if it is alive and matches this query.
    #[must_use]
    pub fn get(&self, entity: Entity) -> Option<D::Item<'_>> {
        self.fetch(entity)
    }
}

/// An iterator over a query's matching data. Entity order is unspecified.
pub struct QueryIter<'u, D: QueryData, F: QueryFilter> {
    entities: Keys<'u, Entity, WorldId>,
    universe: &'u Universe,
    last_run: u64,
    world: Option<WorldId>,
    _marker: PhantomData<fn() -> (D, F)>,
}

impl<'u, D: QueryData, F: QueryFilter> Iterator for QueryIter<'u, D, F> {
    type Item = D::Item<'u>;

    fn next(&mut self) -> Option<Self::Item> {
        for &entity in self.entities.by_ref() {
            if self
                .world
                .is_some_and(|world| !self.universe.is_in_world(world, entity))
                || !F::matches(entity, self.universe, self.last_run)
            {
                continue;
            }
            if let Some(item) = D::fetch(entity, self.universe) {
                return Some(item);
            }
        }
        None
    }
}

impl<'q, D: ReadOnlyQueryData, F: QueryFilter> IntoIterator for &'q Query<'_, D, F> {
    type Item = D::Item<'q>;
    type IntoIter = QueryIter<'q, D, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'q, D: QueryData, F: QueryFilter> IntoIterator for &'q mut Query<'_, D, F> {
    type Item = D::Item<'q>;
    type IntoIter = QueryIter<'q, D, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Describes the data fetched for each entity matched by a query: `Entity`,
/// `&C`, `&mut C`, `Option<D>`, or tuples of these.
pub trait QueryData: Sealed + Sized {
    /// The concrete value borrowed or produced for one matched entity.
    type Item<'u>;

    /// Builds this data for `entity`, returning `None` if it does not match.
    #[doc(hidden)]
    fn fetch(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>>;

    /// Registers component access for conflict validation.
    #[doc(hidden)]
    fn register_access(access: &mut QueryAccess);
}

/// Query data that does not require mutable access.
pub trait ReadOnlyQueryData: QueryData {}

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

impl QueryData for () {
    type Item<'u> = ();

    fn fetch(_: Entity, _: &Universe) -> Option<Self::Item<'_>> {
        Some(())
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryData for () {}

macro_rules! impl_query_data_for_tuple {
    ($($ty:ident $binding:ident),+) => {
        impl<$($ty: QueryData),+> QueryData for ($($ty,)+) {
            type Item<'u> = ($($ty::Item<'u>,)+);

            fn fetch(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
                Some(($($ty::fetch(entity, universe)?,)+))
            }

            fn register_access(access: &mut QueryAccess) {
                $($ty::register_access(access);)+
            }
        }

        impl<$($ty: ReadOnlyQueryData),+> ReadOnlyQueryData for ($($ty,)+) {}
    };
}
all_tuples!(impl_query_data_for_tuple);

impl<C: Component> Sealed for &C {}
impl<C: Component> QueryData for &C {
    type Item<'u> = Ref<'u, C>;

    fn fetch(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }

    fn register_access(access: &mut QueryAccess) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryData for &C {}

impl<C: Component> Sealed for &mut C {}
impl<C: Component> QueryData for &mut C {
    type Item<'u> = ComponentMut<'u, C>;

    fn fetch(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        universe
            .components
            .get_mut::<C>(entity, universe.change_tick.get())
    }

    fn register_access(access: &mut QueryAccess) {
        access.write::<C>();
    }
}

impl<D: QueryData> Sealed for Option<D> {}
impl<D: QueryData> QueryData for Option<D> {
    type Item<'u> = Option<D::Item<'u>>;

    fn fetch(entity: Entity, universe: &Universe) -> Option<Self::Item<'_>> {
        Some(D::fetch(entity, universe))
    }

    fn register_access(access: &mut QueryAccess) {
        D::register_access(access);
    }
}
impl<D: ReadOnlyQueryData> ReadOnlyQueryData for Option<D> {}

impl Sealed for Entity {}
impl QueryData for Entity {
    type Item<'u> = Entity;

    fn fetch(entity: Entity, _: &Universe) -> Option<Self::Item<'_>> {
        Some(entity)
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryData for Entity {}
