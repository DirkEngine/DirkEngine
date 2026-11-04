//! Typed iteration and lookups over live entities across all worlds.
//! Tuples fetch components together; filter tuples combine conditions with AND.

pub mod filter;

use self::filter::QueryFilter;
use crate::{
    Entity, Universe, WorldId,
    components::{Component, ComponentMut, MutableComponent},
    macros::sealed::Sealed,
};
use std::{
    any::{TypeId, type_name},
    cell::Ref,
    collections::{HashMap, HashSet, hash_set},
    marker::PhantomData,
};

/// A system's view of matching entities. Systems run once per tick, even when
/// their queries are empty. Include `Entity` in `D` to fetch entity IDs.
///
/// `&C` yields a shared borrow; `&mut C` yields a mutable borrow that marks the
/// component changed on mutable dereference. `Option<D>` fetches `D` when
/// present without requiring it. [`Delta<C>`] yields what happened to `C`
/// since the system last ran.
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
        let candidates = if D::Liveness::TRACKED {
            let mut entities = HashSet::new();
            D::candidates(self.universe, self.last_run, &mut entities);
            Candidates::Listed(entities.into_iter())
        } else {
            Candidates::Alive(self.universe.alive.iter())
        };
        QueryIter {
            candidates,
            universe: self.universe,
            last_run: self.last_run,
            world,
            _marker: PhantomData,
        }
    }

    fn fetch(&self, entity: Entity) -> Option<D::Item<'_>> {
        if !D::Liveness::TRACKED && !self.universe.is_alive(entity) {
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

/// The entities a query examines: every live entity, or the entities with a
/// change for [`Delta`] queries.
enum Candidates<'u> {
    Alive(hash_set::Iter<'u, Entity>),
    Listed(hash_set::IntoIter<Entity>),
}

impl Iterator for Candidates<'_> {
    type Item = Entity;

    fn next(&mut self) -> Option<Entity> {
        match self {
            Self::Alive(entities) => entities.next().copied(),
            Self::Listed(entities) => entities.next(),
        }
    }
}

/// An iterator over a query's matching data. Entity order is unspecified.
pub struct QueryIter<'u, D: QueryData, F: QueryFilter> {
    candidates: Candidates<'u>,
    universe: &'u Universe,
    last_run: u64,
    world: Option<WorldId>,
    _marker: PhantomData<fn() -> (D, F)>,
}

impl<'u, D: QueryData, F: QueryFilter> Iterator for QueryIter<'u, D, F> {
    type Item = D::Item<'u>;

    fn next(&mut self) -> Option<Self::Item> {
        for entity in self.candidates.by_ref() {
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

/// What happened to component `C` since the observing system last ran.
///
/// A `Delta` reports the net change: a component removed and re-added is
/// `Set`, and one added and removed again in between runs is not reported.
/// On a system's first run, every present component is `Set`. Despawned
/// entities report `Removed`, so combine `Delta` only with `Entity`:
///
/// ```compile_fail
/// # use dirk_universe::prelude::*;
/// # #[derive(Debug, Component)] struct Health(u32);
/// # #[derive(Debug, Component)] struct Armor(u32);
/// fn sync(query: Query<(Delta<Health>, &Armor)>) {}
/// ```
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

    /// Which entities this data can describe.
    #[doc(hidden)]
    type Liveness: Liveness;

    /// Builds this data for `entity`, returning `None` if it does not match.
    #[doc(hidden)]
    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>>;

    /// Adds the entities a [`Delta`] query must examine.
    #[doc(hidden)]
    fn candidates(_universe: &Universe, _last_run: u64, _entities: &mut HashSet<Entity>) {}

    /// Registers component access for conflict validation.
    #[doc(hidden)]
    fn register_access(access: &mut QueryAccess);
}

/// Query data that does not require mutable access.
pub trait ReadOnlyQueryData: QueryData {}

/// Which entities a query term can describe. Terms combine with [`Join`], so
/// the type system rejects queries mixing [`Delta`] with other data.
#[doc(hidden)]
pub trait Liveness: Sealed + Join<AnyEntity, Output = Self> {
    /// Whether queries iterate changed and removed entities instead of live ones.
    const TRACKED: bool;
}

/// Describes any entity, live or despawned: `Entity`, `()` and `Without<C>`.
#[doc(hidden)]
pub enum AnyEntity {}
/// Describes live entities only: component data and most filters.
#[doc(hidden)]
pub enum LiveEntity {}
/// Describes changed and despawned entities: [`Delta`].
#[doc(hidden)]
pub enum TrackedEntity {}

impl Sealed for AnyEntity {}
impl Sealed for LiveEntity {}
impl Sealed for TrackedEntity {}
impl Liveness for AnyEntity {
    const TRACKED: bool = false;
}
impl Liveness for LiveEntity {
    const TRACKED: bool = false;
}
impl Liveness for TrackedEntity {
    const TRACKED: bool = true;
}

/// Combines the [`Liveness`] of two query terms.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "`Delta` can only be combined with `Entity`",
    label = "this query combines `Delta` with component data, filters or another `Delta`",
    note = "`Delta` also yields despawned entities, which have no other data"
)]
pub trait Join<R> {
    /// The combined liveness.
    type Output: Liveness;
}
impl<R: Liveness> Join<R> for AnyEntity {
    type Output = R;
}
impl Join<AnyEntity> for LiveEntity {
    type Output = LiveEntity;
}
impl Join<LiveEntity> for LiveEntity {
    type Output = LiveEntity;
}
impl Join<AnyEntity> for TrackedEntity {
    type Output = TrackedEntity;
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

impl QueryData for () {
    type Item<'u> = ();
    type Liveness = AnyEntity;

    fn fetch(_: Entity, _: &Universe, _: u64) -> Option<Self::Item<'_>> {
        Some(())
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryData for () {}

macro_rules! impl_query_data_for_tuple {
    ($first:ident $first_binding:ident $(, $ty:ident $binding:ident)*) => {
        impl<$first: QueryData $(, $ty: QueryData)*> QueryData for ($first, $($ty,)*)
        where
            ($($ty,)*): QueryData,
            $first::Liveness: Join<<($($ty,)*) as QueryData>::Liveness>,
        {
            type Item<'u> = ($first::Item<'u>, $($ty::Item<'u>,)*);
            type Liveness =
                <$first::Liveness as Join<<($($ty,)*) as QueryData>::Liveness>>::Output;

            fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
                Some((
                    $first::fetch(entity, universe, last_run)?,
                    $($ty::fetch(entity, universe, last_run)?,)*
                ))
            }

            fn candidates(universe: &Universe, last_run: u64, entities: &mut HashSet<Entity>) {
                $first::candidates(universe, last_run, entities);
                $($ty::candidates(universe, last_run, entities);)*
            }

            fn register_access(access: &mut QueryAccess) {
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
    type Liveness = LiveEntity;

    fn fetch(entity: Entity, universe: &Universe, _: u64) -> Option<Self::Item<'_>> {
        universe.component::<C>(entity)
    }

    fn register_access(access: &mut QueryAccess) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryData for &C {}

impl<C: MutableComponent> Sealed for &mut C {}
impl<C: MutableComponent> QueryData for &mut C {
    type Item<'u> = ComponentMut<'u, C>;
    type Liveness = LiveEntity;

    fn fetch(entity: Entity, universe: &Universe, _: u64) -> Option<Self::Item<'_>> {
        universe
            .components
            .get_mut::<C>(entity, universe.change_tick.get())
    }

    fn register_access(access: &mut QueryAccess) {
        access.write::<C>();
    }
}

impl<D: QueryData> Sealed for Option<D> {}
impl<D: QueryData> QueryData for Option<D>
where
    D::Liveness: Join<LiveEntity>,
{
    type Item<'u> = Option<D::Item<'u>>;
    type Liveness = LiveEntity;

    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
        Some(D::fetch(entity, universe, last_run))
    }

    fn register_access(access: &mut QueryAccess) {
        D::register_access(access);
    }
}
impl<D: ReadOnlyQueryData> ReadOnlyQueryData for Option<D> where D::Liveness: Join<LiveEntity> {}

impl<C: Component> Sealed for Delta<'_, C> {}
impl<C: Component> QueryData for Delta<'_, C> {
    type Item<'u> = Delta<'u, C>;
    type Liveness = TrackedEntity;

    fn fetch(entity: Entity, universe: &Universe, last_run: u64) -> Option<Self::Item<'_>> {
        if universe.components.changed::<C>(entity, last_run) {
            return universe.component::<C>(entity).map(Delta::Set);
        }
        let removed = !universe.components.contains(entity, TypeId::of::<C>())
            && universe
                .removed_since::<C>(last_run)
                .any(|removed| removed == entity);
        removed.then_some(Delta::Removed)
    }

    fn candidates(universe: &Universe, last_run: u64, entities: &mut HashSet<Entity>) {
        entities.extend(universe.components.changed_since::<C>(last_run));
        entities.extend(universe.removed_since::<C>(last_run));
    }

    fn register_access(access: &mut QueryAccess) {
        access.read::<C>();
    }
}
impl<C: Component> ReadOnlyQueryData for Delta<'_, C> {}

impl Sealed for Entity {}
impl QueryData for Entity {
    type Item<'u> = Entity;
    type Liveness = AnyEntity;

    fn fetch(entity: Entity, _: &Universe, _: u64) -> Option<Self::Item<'_>> {
        Some(entity)
    }

    fn register_access(_: &mut QueryAccess) {}
}
impl ReadOnlyQueryData for Entity {}
