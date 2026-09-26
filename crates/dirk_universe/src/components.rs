//! Component definitions and storage.

use crate::Entity;
use std::{
    any::{Any, TypeId},
    cell::{Cell, RefCell, RefMut},
    collections::HashMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
    rc::Rc,
};

/// Base marker trait for component types.
pub trait Component: 'static + Sized + Debug + Send {}
#[doc(hidden)]
pub use dirk_proc::Component;

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

/// A retained component value in the ordered change log.
///
/// Values are shared with storage, so retaining old or removed components does
/// not require `Component: Clone`. Mutable queries preserve old values by
/// cloning their components. Mutation through interior-mutability types is not
/// tracked.
#[derive(Clone, Debug)]
pub struct ComponentValue(pub(crate) Rc<dyn AnyComponent>);

impl ComponentValue {
    /// Borrows the value when it has component type `C`.
    #[must_use]
    pub fn get<C: Component>(&self) -> Option<&C> {
        self.0.as_any().downcast_ref()
    }
}

/// A mutable component view. Mutably dereferencing it records a change.
pub struct ComponentMut<'a, C: Component> {
    value: RefMut<'a, C>,
    dirty: &'a Cell<bool>,
    edit_order: &'a RefCell<Vec<(Entity, TypeId)>>,
    entity: Entity,
}

impl<C: Component> Deref for ComponentMut<'_, C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<C: Component> DerefMut for ComponentMut<'_, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if !self.dirty.replace(true) {
            self.edit_order
                .borrow_mut()
                .push((self.entity, TypeId::of::<C>()));
        }
        &mut self.value
    }
}

#[derive(Debug)]
struct ComponentSlot {
    value: ComponentValue,
    edit: RefCell<Option<Box<dyn AnyComponent>>>,
    dirty: Cell<bool>,
}

impl ComponentSlot {
    fn new(value: ComponentValue) -> Self {
        Self {
            value,
            edit: RefCell::new(None),
            dirty: Cell::new(false),
        }
    }
}

/// Components share their stored values with the command change log. Mutable
/// queries edit a separate copy so those historical values remain unchanged.
#[derive(Default, Debug)]
pub(crate) struct Components {
    storages: HashMap<TypeId, HashMap<Entity, ComponentSlot>>,
}

impl Components {
    pub fn insert(&mut self, entity: Entity, value: ComponentValue) -> Option<ComponentValue> {
        self.storages
            .entry(value.0.component_type_id())
            .or_default()
            .insert(entity, ComponentSlot::new(value))
            .map(|slot| slot.value)
    }

    pub fn get_all(&self, entity: Entity) -> impl Iterator<Item = (TypeId, &dyn AnyComponent)> {
        self.storages.iter().filter_map(move |(id, storage)| {
            storage
                .get(&entity)
                .map(|slot| (*id, slot.value.0.as_ref()))
        })
    }

    pub fn get<C: Component>(&self, entity: Entity) -> Option<&C> {
        self.storages
            .get(&TypeId::of::<C>())?
            .get(&entity)?
            .value
            .get()
    }

    pub fn get_mut<'a, C: Component + Clone>(
        &'a self,
        entity: Entity,
        edit_order: &'a RefCell<Vec<(Entity, TypeId)>>,
        prepared_order: &RefCell<Vec<(Entity, TypeId)>>,
    ) -> Option<ComponentMut<'a, C>> {
        let slot = self.storages.get(&TypeId::of::<C>())?.get(&entity)?;
        let mut edit = slot.edit.borrow_mut();
        if edit.is_none() {
            *edit = Some(Box::new(slot.value.get::<C>()?.clone()));
            prepared_order
                .borrow_mut()
                .push((entity, TypeId::of::<C>()));
        }
        Some(ComponentMut {
            value: RefMut::map(edit, |value| {
                value
                    .as_mut()
                    .and_then(|value| value.as_any_mut().downcast_mut::<C>())
                    .expect("component storage type must match its TypeId")
            }),
            dirty: &slot.dirty,
            edit_order,
            entity,
        })
    }

    /// Commits mutable query edits in first-write order for the next change log.
    pub fn finish_edits(
        &mut self,
        edit_order: &mut Vec<(Entity, TypeId)>,
        prepared_order: &mut Vec<(Entity, TypeId)>,
    ) -> Vec<(Entity, ComponentValue, ComponentValue)> {
        let mut updates = Vec::with_capacity(edit_order.len());
        for (entity, type_id) in edit_order.drain(..) {
            let slot = self
                .storages
                .get_mut(&type_id)
                .and_then(|storage| storage.get_mut(&entity))
                .expect("edited component must still exist until the next tick");
            let edited = slot
                .edit
                .get_mut()
                .take()
                .expect("dirty component must have an edit");
            let new = ComponentValue(edited.into());
            let old = std::mem::replace(&mut slot.value, new.clone());
            slot.dirty.set(false);
            updates.push((entity, old, new));
        }
        // Read-only mutable views may have prepared a copy without writing it.
        for (entity, type_id) in prepared_order.drain(..) {
            self.storages
                .get_mut(&type_id)
                .and_then(|storage| storage.get_mut(&entity))
                .expect("prepared component must still exist until the next tick")
                .edit
                .get_mut()
                .take();
        }
        updates
    }

    pub fn remove(&mut self, entity: Entity, type_id: TypeId) -> Option<ComponentValue> {
        self.storages
            .get_mut(&type_id)?
            .remove(&entity)
            .map(|slot| slot.value)
    }

    pub fn contains(&self, entity: Entity, type_id: TypeId) -> bool {
        self.storages
            .get(&type_id)
            .is_some_and(|storage| storage.contains_key(&entity))
    }
}
