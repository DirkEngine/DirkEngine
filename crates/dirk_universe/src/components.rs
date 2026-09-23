//! Component definitions and storage.

use crate::Entity;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
    rc::Rc,
};

/// Base marker trait for component types.
pub trait Component: 'static + Sized + Debug + Send {}
#[doc(hidden)]
pub use dirk_proc::Component;

pub(crate) trait AnyComponent: Send + Any + Debug {
    fn as_any(&self) -> &dyn Any;
    fn component_type_id(&self) -> TypeId;
    fn component_type_name(&self) -> &'static str;
}

impl<C: Component> AnyComponent for C {
    fn as_any(&self) -> &dyn Any {
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
/// not require `Component: Clone`. Interior mutation is not tracked; changes
/// describe commands, not mutations made through interior-mutability types.
#[derive(Clone, Debug)]
pub struct ComponentValue(pub(crate) Rc<dyn AnyComponent>);

impl ComponentValue {
    /// Borrows the value when it has component type `C`.
    #[must_use]
    pub fn get<C: Component>(&self) -> Option<&C> {
        self.0.as_any().downcast_ref()
    }
}

/// Components share their values with this tick's log. Queries still return
/// ordinary references; replacing a component leaves its old value in the log.
#[derive(Default, Debug)]
pub(crate) struct Components {
    storages: HashMap<TypeId, HashMap<Entity, ComponentValue>>,
}

impl Components {
    pub fn insert(&mut self, entity: Entity, value: ComponentValue) -> Option<ComponentValue> {
        self.storages
            .entry(value.0.component_type_id())
            .or_default()
            .insert(entity, value)
    }

    pub fn get_all(&self, entity: Entity) -> impl Iterator<Item = (TypeId, &dyn AnyComponent)> {
        self.storages.iter().filter_map(move |(id, storage)| {
            storage.get(&entity).map(|value| (*id, value.0.as_ref()))
        })
    }

    pub fn get<C: Component>(&self, entity: Entity) -> Option<&C> {
        self.storages.get(&TypeId::of::<C>())?.get(&entity)?.get()
    }

    pub fn remove(&mut self, entity: Entity, type_id: TypeId) -> Option<ComponentValue> {
        self.storages.get_mut(&type_id)?.remove(&entity)
    }

    pub fn contains(&self, entity: Entity, type_id: TypeId) -> bool {
        self.storages
            .get(&type_id)
            .is_some_and(|storage| storage.contains_key(&entity))
    }
}
