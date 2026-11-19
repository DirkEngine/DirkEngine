//! Typed iteration and lookups over live entities across all worlds.
//! Tuples fetch components together; filter tuples combine conditions with AND.

pub mod filter;

use self::filter::QueryFilter;
use crate::{
    Entity, Universe, WorldId,
    components::{Component, ComponentMut, MutableComponent},
    macros::sealed::Sealed,
    schedule::Access,
};
use std::{any::TypeId, cell::Ref, collections::hash_set, marker::PhantomData};

/// A system's view of matching entities. Systems run once per tick, even when
/// their queries are empty. Include `Entity` in `D` to fetch entity IDs.
///
/// `&C` yields a shared borrow; `&mut C` yields a mutable borrow that marks the
/// component changed on mutable dereference. `Option<D>` fetches `D` when
/// present without requiring it. To observe what happened to `C` since the
/// system last ran, including removals, use [`Changes<C>`] instead.
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
        let candidates = self.universe.alive.iter();
        QueryIter {
            candidates,
            universe: self.universe,
            last_run: self.last_run,
            world,
            _marker: PhantomData,
        }
    }

    fn fetch(&self, entity: Entity) -> Option<D::Item<'_>> {
        if !self.universe.is_alive(entity) {
            return None;
        }
        if !F::matches(entity, self.universe, self.last_run) {
            return None;
        }
        D::fetch(entity, self.universe, self.last_run)
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

    /// Borrows one entity if it matches this query.
    #[must_use]
    pub fn get(&self, entity: Entity) -> Option<D::Item<'_>> {
        self.fetch(entity)
    }
}

/// An iterator over a query's matching data. Entity order is unspecified.
pub struct QueryIter<'u, D: QueryData, F: QueryFilter> {
    candidates: hash_set::Iter<'u, Entity>,
    universe: &'u Universe,
    last_run: u64,
    world: Option<WorldId>,
    _marker: PhantomData<fn() -> (D, F)>,
}

impl<'u, D: QueryData, F: QueryFilter> Iterator for QueryIter<'u, D, F> {
    type Item = D::Item<'u>;

    fn next(&mut self) -> Option<Self::Item> {
        for &entity in self.candidates.by_ref() {
            if self
                .world
                .is_some_and(|world| !self.universe.is_in_world(world, entity))
                || !F::matches(entity, self.universe, self.last_run)
            {
                continue;
            }
            if let Some(item) = D::fetch(entity, self.universe, self.last_run) {
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

/// Every entity whose component `C` changed since the observing system last
/// ran, with what happened to it. A system parameter:
///
/// ```
/// # use dirk_universe::prelude::*;
/// # #[derive(Debug, Component)] struct Health(u32);
/// fn synchronize(health: Changes<Health>) {
///     for (entity, delta) in &health {
///         match delta {
///             Delta::Set(value) => println!("{entity:?} now has {} health", value.0),
///             Delta::Removed => println!("forget {entity:?}"),
///         }
///     }
/// }
/// ```
///
/// Changes are net: a component removed and re-added is `Set`, and one added
/// and removed again between two runs is not reported. On a system's first
/// run, every present component is `Set`. Each system observes changes
/// independently, and removals are kept until every system has run once
/// after them.
pub struct Changes<'u, C: Component> {
    universe: &'u Universe,
    last_run: u64,
    _marker: PhantomData<fn() -> C>,
}

impl<'u, C: Component> Changes<'u, C> {
    pub(crate) fn new(universe: &'u Universe, last_run: u64) -> Self {
        Self {
            universe,
            last_run,
            _marker: PhantomData,
        }
    }

    /// Iterates over the changed entities. Entity order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (Entity, Delta<'u, C>)> + 'u {
        let (universe, last_run) = (self.universe, self.last_run);
        let set = universe
            .components
            .changed_since::<C>(last_run)
            .filter_map(move |entity| Some((entity, Delta::Set(universe.component::<C>(entity)?))));
        let removed = universe
            .removed_since::<C>(last_run)
            .filter(move |&entity| !universe.components.contains(entity, TypeId::of::<C>()))
            .map(|entity| (entity, Delta::Removed));
        set.chain(removed)
    }
}

impl<'u, C: Component> IntoIterator for &Changes<'u, C> {
    type Item = (Entity, Delta<'u, C>);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'u>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

/// What happened to one entity's component `C`, as reported by [`Changes`].
#[derive(Debug)]
pub enum Delta<'u, C: Component> {
    /// `C` was added, replaced or mutated; this is its current value.
    Set(Ref<'u, C>),
    /// `C` was removed, possibly by despawning its entity.
    Removed,
}

impl<C: Component> Delta<'_, C> {
    /// Returns the current value, or `None` when `C` was removed.
    #[must_use]
    pub fn value(&self) -> Option<&C> {
        match self {
            Self::Set(value) => Some(value),
            Self::Removed => None,
        }
    }
}

/// Describes the data fetched for each entity matched by a query: `Entity`,
/// `&C`, `&mut C`, `Option<D>`, or tuples of these.
pub trait QueryData: Sealed + Sized {
    /// The concrete value borrowed or produced for one matched entity.
    type Item<'u>;

    /// Builds this data for `entity`, returning `None` if it does not match.
    #[doc(hidden)]
    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>>;

    /// Registers component access for conflict validation.
    #[doc(hidden)]
    fn register_access(access: &mut Access);
}

/// Query data that does not require mutable access.
pub trait ReadOnlyQueryData: QueryData {}

impl QueryData for () {
    type Item<'u> = ();

    fn fetch(_: Entity, _: &Universe, _: u64) -> Option<Self::Item<'_>> {
        Some(())
    }

    fn register_access(_: &mut Access) {}
}
impl ReadOnlyQueryData for () {}

macro_rules! impl_query_data_for_tuple {
    ($first:ident $first_binding:ident $(, $ty:ident $binding:ident)*) => {
        impl<$first: QueryData $(, $ty: QueryData)*> QueryData for ($first, $($ty,)*)
        where
            ($($ty,)*): QueryData,
        {
            type Item<'u> = ($first::Item<'u>, $($ty::Item<'u>,)*);

            fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
                Some((
                    $first::fetch(entity, universe, last_run)?,
                    $($ty::fetch(entity, universe, last_run)?,)*
                ))
            }

            fn register_access(access: &mut Access) {
                $first::register_access(access);
                $($ty::register_access(access);)*
            }
        }

        impl<$first: ReadOnlyQueryData $(, $ty: ReadOnlyQueryData)*> ReadOnlyQueryData
            for ($first, $($ty,)*)
        where
            Self: QueryData,
        {
        }
    };
}
all_tuples!(impl_query_data_for_tuple);

impl<C: Component> Sealed for &C {}
impl<C: Component> QueryData for &C {
    type Item<'u> = Ref<'u, C>;

    fn fetch(entity: Entity, universe: &Universe, _: u64) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }

    fn register_access(access: &mut Access) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryData for &C {}

impl<C: MutableComponent> Sealed for &mut C {}
impl<C: MutableComponent> QueryData for &mut C {
    type Item<'u> = ComponentMut<'u, C>;

    fn fetch(entity: Entity, universe: &Universe, _: u64) -> Option<Self::Item<'_>> {
        universe
            .components
            .get_mut::<C>(entity, universe.change_tick.get())
    }

    fn register_access(access: &mut Access) {
        access.write::<C>();
    }
}

impl<D: QueryData> Sealed for Option<D> {}
impl<D: QueryData> QueryData for Option<D> {
    type Item<'u> = Option<D::Item<'u>>;

    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
        Some(D::fetch(entity, universe, last_run))
    }

    fn register_access(access: &mut Access) {
        D::register_access(access);
    }
}
impl<D: ReadOnlyQueryData> ReadOnlyQueryData for Option<D> {}

/// Reads `D` before this tick's systems write it: systems writing `D`'s
/// components run after this one, which breaks dependency cycles. Commands
/// applied at the start of the tick are already visible, and an explicit
/// `.before()` placing a writer first takes precedence.
pub struct Lagged<D>(PhantomData<D>);

impl<D: ReadOnlyQueryData> Sealed for Lagged<D> {}
impl<D: ReadOnlyQueryData> QueryData for Lagged<D> {
    type Item<'u> = D::Item<'u>;

    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
        D::fetch(entity, universe, last_run)
    }

    fn register_access(access: &mut Access) {
        access.lagged(D::register_access);
    }
}
impl<D: ReadOnlyQueryData> ReadOnlyQueryData for Lagged<D> {}

impl Sealed for Entity {}
impl QueryData for Entity {
    type Item<'u> = Entity;

    fn fetch(entity: Entity, _: &Universe, _: u64) -> Option<Self::Item<'_>> {
        Some(entity)
    }

    fn register_access(_: &mut Access) {}
}
impl ReadOnlyQueryData for Entity {}
