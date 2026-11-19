//! This module has all the traits & structs required for component storage
//! and creation.
//!
//! Implement the [`Component`] traits to get
//! started (you can do this via a derive macro).
use crate::Entity;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
    ops::{Index, IndexMut},
};

/// Base marker trait for component types.
pub trait Component: 'static + Sized + Debug + Send {}
#[doc(hidden)]
pub use dirk_proc::Component;

/// A dyn-compatible wrapper around Component, used wherever
/// type-erased component values must be passed around at runtime.
#[doc(hidden)]
pub(crate) trait AnyComponent: Send + Any + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    /// Consume this boxed value and return it as a plain `Box<dyn Any>`,
    /// enabling `Box::downcast::<C>()` at call sites.
    fn into_any(self: Box<Self>) -> Box<dyn Any>;

    /// Takes `self` so it can be called on a [`AnyComponent`] directly
    /// without needing to know the type.
    fn new_storage(&self) -> Box<dyn AnyStorage>;

    /// Returns the [`TypeId`] of the concrete component behind the any type.
    fn component_type_id(&self) -> TypeId;
    /// Returns the Rust type name of the concrete component.
    fn component_type_name(&self) -> &'static str;
}

// Blanket impl: every concrete Component automatically becomes an AnyComponent.
impl<T: Component> AnyComponent for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn new_storage(&self) -> Box<dyn AnyStorage> {
        Box::new(ComponentStorage::<T>::default())
    }
    fn component_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
    fn component_type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

/// Type-erased storage for a single component type.
///
/// The `as_any` / `as_any_mut` pattern lets us downcast back to the concrete
/// `EntityComponentStorage<C>` without exposing `C` through the trait object.
pub(crate) trait AnyStorage {
    /// Remove the component for `entity` and return it as a type-erased box,
    /// or `None` if the entity had no component in this storage.
    fn remove(&mut self, entity: Entity) -> Option<Box<dyn AnyComponent>>;
    /// Insert a type-erased component for `entity`.
    ///
    /// The concrete type inside `component` must match the type this storage
    /// was created for; mismatches are silently ignored.
    fn insert(
        &mut self,
        entity: Entity,
        component: Box<dyn AnyComponent>,
    ) -> Option<Box<dyn AnyComponent>>;
    /// Will return the component associated with the entity
    fn get(&self, entity: Entity) -> Option<&dyn AnyComponent>;
    /// Returns `true` if this storage holds a component for `entity`.
    fn contains(&self, entity: Entity) -> bool;
    fn as_any(&self) -> &dyn Any;
}

struct ComponentStorage<C: Component> {
    map: HashMap<Entity, C>,
}

impl<C: Component> Default for ComponentStorage<C> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl<C: Component> ComponentStorage<C> {
    fn get(&self, entity: Entity) -> Option<&C> {
        self.map.get(&entity)
    }
}

impl<C: Component> AnyStorage for ComponentStorage<C> {
    fn remove(&mut self, entity: Entity) -> Option<Box<dyn AnyComponent>> {
        self.map
            .remove(&entity)
            .map(|c| Box::new(c) as Box<dyn AnyComponent>)
    }
    fn insert(
        &mut self,
        entity: Entity,
        component: Box<dyn AnyComponent>,
    ) -> Option<Box<dyn AnyComponent>> {
        // Downcast through Box<dyn Any> so we can move the value out of the box.
        if let Ok(c) = component.into_any().downcast::<C>() {
            self.map
                .insert(entity, *c)
                .map(|component| Box::new(component) as Box<dyn AnyComponent>)
        } else {
            // Type mismatch — should never happen in practice because
            // insert_box always routes to the storage that matches C.
            debug_assert!(false, "insert_any called with wrong component type");
            None
        }
    }

    fn contains(&self, entity: Entity) -> bool {
        self.map.contains_key(&entity)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get(&self, entity: Entity) -> Option<&dyn AnyComponent> {
        self.map.get(&entity).map(|c| c as &dyn AnyComponent)
    }
}

impl<C: Component> IndexMut<Entity> for ComponentStorage<C> {
    #[expect(
        clippy::expect_used,
        reason = "Indexing panics on a missing key, like std's maps."
    )]
    fn index_mut(&mut self, index: Entity) -> &mut Self::Output {
        self.map
            .get_mut(&index)
            .expect("entity should have component if indexing")
    }
}

impl<C: Component> Index<Entity> for ComponentStorage<C> {
    type Output = C;
    fn index(&self, index: Entity) -> &Self::Output {
        &self.map[&index]
    }
}

/// Dynamic, open-ended component storage.
///
/// Each component type gets its own `HashMap<Entity, C>`, looked up by
/// [`TypeId`].  No central registration is required — storage for a type is
/// created on the first `insert` and lives until the `World` is dropped.
#[derive(Default)]
pub(crate) struct Components {
    storages: HashMap<TypeId, Box<dyn AnyStorage>>,
}

impl std::fmt::Debug for Components {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EntityComponents {{ {} type(s) }}", self.storages.len())
    }
}

impl Components {
    /// Insert a type-erased component.
    ///
    /// Storage for the component's concrete type is created on demand using
    /// [`AnyComponent::new_storage`].  The `TypeId` used as the key comes from
    /// `Any::type_id`, which dispatches to the concrete type at runtime.
    pub fn insert_any(
        &mut self,
        entity: Entity,
        component: Box<dyn AnyComponent>,
    ) -> Option<Box<dyn AnyComponent>> {
        self.storages
            .entry(component.component_type_id())
            .or_insert_with(|| component.new_storage())
            .insert(entity, component)
    }

    pub fn get_any(&self, entity: Entity, type_id: TypeId) -> Option<&dyn AnyComponent> {
        self.storages.get(&type_id)?.get(entity)
    }

    pub fn get_all(&self, entity: Entity) -> impl Iterator<Item = (TypeId, &dyn AnyComponent)> {
        self.storages.iter().filter_map(move |(type_id, storage)| {
            storage.get(entity).map(|component| (*type_id, component))
        })
    }

    pub fn get<C: Component>(&self, entity: Entity) -> Option<&C> {
        self.storages
            .get(&TypeId::of::<C>())
            .and_then(|b| b.as_any().downcast_ref::<ComponentStorage<C>>())?
            .get(entity)
    }

    /// Will remove the [`Component`] of the specified [`TypeId`] from the [`Entity`].
    pub fn remove_any(&mut self, entity: Entity, type_id: TypeId) -> Option<Box<dyn AnyComponent>> {
        self.storages
            .get_mut(&type_id)
            .and_then(|b| b.remove(entity))
    }

    /// Check whether `entity` has a component of the given `TypeId`.
    ///
    /// Used by [`Query`] so it can work with a plain `Vec<TypeId>` rather
    /// than monomorphised generics.
    ///
    /// [`Query`]: crate::query::Query
    pub fn contains(&self, entity: Entity, type_id: TypeId) -> bool {
        self.storages
            .get(&type_id)
            .is_some_and(|s| s.contains(entity))
    }
}
