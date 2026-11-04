//! Component definitions and storage.

use crate::{Entity, macros::sealed::Sealed};
use std::{
    any::{Any, TypeId},
    cell::{Cell, Ref, RefCell, RefMut},
    collections::HashMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

/// Data attached to an entity. Derive it with `#[derive(Component)]`.
pub trait Component: 'static + Sized + Debug + Send {
    /// [`Mutable`] for ordinary components, or [`ReadOnly`] for components
    /// that only the engine writes, such as [`InWorld`](crate::InWorld).
    type Mutability: Mutability;
}
#[doc(hidden)]
pub use dirk_proc::Component;

/// Whether systems and commands may write a component.
pub trait Mutability: Sealed {}

/// Components that systems edit and commands insert or remove.
pub enum Mutable {}
impl Sealed for Mutable {}
impl Mutability for Mutable {}

/// Components that queries can read but only the engine writes.
pub enum ReadOnly {}
impl Sealed for ReadOnly {}
impl Mutability for ReadOnly {}

/// A component that `&mut C` queries, commands and entity builders may write.
pub trait MutableComponent: Component<Mutability = Mutable> {}
impl<C: Component<Mutability = Mutable>> MutableComponent for C {}

pub(crate) trait AnyComponent: Send + Any + Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn component_type_id(&self) -> TypeId;
    fn component_type_name(&self) -> &'static str;
}

impl<C: Component> AnyComponent for C {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn component_type_id(&self) -> TypeId {
        TypeId::of::<C>()
    }
    fn component_type_name(&self) -> &'static str {
        std::any::type_name::<C>()
    }
}

/// A mutable component borrow. Mutable dereferencing marks it changed, even
/// when the assigned value is equal. Merely reading through it does not.
pub struct ComponentMut<'a, C: Component> {
    value: RefMut<'a, C>,
    changed: &'a Cell<u64>,
    tick: u64,
}

impl<C: Component> Deref for ComponentMut<'_, C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<C: Component> DerefMut for ComponentMut<'_, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.changed.set(self.tick);
        &mut self.value
    }
}

#[derive(Debug)]
struct ComponentSlot {
    value: RefCell<Box<dyn AnyComponent>>,
    added: u64,
    changed: Cell<u64>,
}

/// Per-component borrows allow disjoint queries without unsafe storage access.
#[derive(Default, Debug)]
pub(crate) struct Components {
    storages: HashMap<TypeId, HashMap<Entity, ComponentSlot>>,
}

impl Components {
    pub fn insert(&mut self, entity: Entity, value: Box<dyn AnyComponent>, tick: u64) {
        let storage = self.storages.entry(value.component_type_id()).or_default();
        if let Some(slot) = storage.get_mut(&entity) {
            *slot.value.get_mut() = value;
            slot.changed.set(tick);
        } else {
            storage.insert(
                entity,
                ComponentSlot {
                    value: RefCell::new(value),
                    added: tick,
                    changed: Cell::new(tick),
                },
            );
        }
    }

    pub fn get_all(
        &self,
        entity: Entity,
    ) -> impl Iterator<Item = (TypeId, Ref<'_, dyn AnyComponent>)> {
        self.storages.iter().filter_map(move |(id, storage)| {
            storage
                .get(&entity)
                .map(|slot| (*id, Ref::map(slot.value.borrow(), |value| value.as_ref())))
        })
    }

    pub fn get<C: Component>(&self, entity: Entity) -> Option<Ref<'_, C>> {
        let slot = self.storages.get(&TypeId::of::<C>())?.get(&entity)?;
        Some(Ref::map(slot.value.borrow(), |value| {
            value
                .as_any()
                .downcast_ref::<C>()
                .expect("component storage type must match its TypeId")
        }))
    }

    pub fn get_mut<C: Component>(&self, entity: Entity, tick: u64) -> Option<ComponentMut<'_, C>> {
        let slot = self.storages.get(&TypeId::of::<C>())?.get(&entity)?;
        Some(ComponentMut {
            value: RefMut::map(slot.value.borrow_mut(), |value| {
                value
                    .as_any_mut()
                    .downcast_mut::<C>()
                    .expect("component storage type must match its TypeId")
            }),
            changed: &slot.changed,
            tick,
        })
    }

    pub fn added<C: Component>(&self, entity: Entity, last_run: u64) -> bool {
        self.storages
            .get(&TypeId::of::<C>())
            .and_then(|storage| storage.get(&entity))
            .is_some_and(|slot| slot.added > last_run)
    }

    pub fn changed<C: Component>(&self, entity: Entity, last_run: u64) -> bool {
        self.storages
            .get(&TypeId::of::<C>())
            .and_then(|storage| storage.get(&entity))
            .is_some_and(|slot| slot.changed.get() > last_run)
    }

    /// Entities whose `C` changed after `last_run`.
    pub fn changed_since<C: Component>(&self, last_run: u64) -> impl Iterator<Item = Entity> + '_ {
        self.storages
            .get(&TypeId::of::<C>())
            .into_iter()
            .flat_map(move |storage| {
                storage
                    .iter()
                    .filter(move |(_, slot)| slot.changed.get() > last_run)
                    .map(|(entity, _)| *entity)
            })
    }

    /// Iterates over every entity holding `C`, with its value.
    pub fn iter<C: Component>(&self) -> impl Iterator<Item = (Entity, Ref<'_, C>)> {
        self.storages
            .get(&TypeId::of::<C>())
            .into_iter()
            .flat_map(|storage| storage.keys())
            .filter_map(|entity| Some((*entity, self.get::<C>(*entity)?)))
    }

    /// Removes a component, returning the tick at which it was added.
    pub fn remove(&mut self, entity: Entity, type_id: TypeId) -> Option<u64> {
        self.storages
            .get_mut(&type_id)?
            .remove(&entity)
            .map(|slot| slot.added)
    }

    pub fn contains(&self, entity: Entity, type_id: TypeId) -> bool {
        self.storages
            .get(&type_id)
            .is_some_and(|storage| storage.contains_key(&entity))
    }
}
