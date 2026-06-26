//! This crate has all the traits for the ECS [`System`]s.

pub mod experimental;

use std::{any::TypeId, collections::HashMap};

use crate::{
    CommandBuffer, Entity, Universe, World, WorldId,
    components::{AnyComponent, Component},
    query::Query,
};
use dirk_proc::system_trait;

/// All systems must implement this trait.
pub trait System: 'static {
    /// Get a name for the system. For debug purposes only.
    fn name() -> &'static str;
}
#[doc(hidden)]
pub use dirk_proc::System;

/// A system that is run by the [`Universe`].
#[system_trait]
pub trait UniverseSystem: System {
    /// Called right after the world is created.
    fn world_created(&self, cmd: &mut CommandBuffer, universe: &Universe, world: &World);
    /// Called as the world is being destroyed.
    /// In this state, the world is still valid
    /// and no entities have been removed.
    fn world_destroyed(&self, cmd: &mut CommandBuffer, universe: &Universe, world: &World);

    /// Called when an entity is spawned. At this point, components have
    /// already been added. They can thus be queried for.
    fn entity_spawned(&self, cmd: &mut CommandBuffer, universe: &Universe, entity: Entity);
    /// Called when the entity is moved to another [`World`].
    fn entity_sent(
        &self,
        cmd: &mut CommandBuffer,
        universe: &Universe,
        entity: Entity,
        old: WorldId,
        new: WorldId,
    );
    /// Called when an entity is despawned. At this point, components have
    /// not yet been removed. They can thus be queried for.
    /// However, the entity has been removed from the [`World`], so
    /// querying for it will not work.
    fn entity_despawned(&self, cmd: &mut CommandBuffer, universe: &Universe, entity: Entity);

    /// This function will be called by the [`Universe`] on every tick.
    fn tick(&self, cmd: &mut CommandBuffer, universe: &Universe, delta_time: f64);
}

/// A [`System`] that is run on every entity that matches the query.
#[system_trait]
pub trait EntitySystem: System {
    /// Called when an entity is spawned. At this point, components have
    /// already been added. They can thus be queried for.
    fn spawned(&self, cmd: &mut CommandBuffer, universe: &Universe, entity: Entity);
    /// Called when an entity is despawned. At this point, components have
    /// not yet been removed. They can thus be queried for.
    /// However, the entity has been removed from the [`World`], so
    /// querying for it will not work.
    fn despawned(&self, cmd: &mut CommandBuffer, universe: &Universe, entity: Entity);

    /// Called when the entity is moved to another [`World`].
    fn sent(
        &self,
        cmd: &mut CommandBuffer,
        universe: &Universe,
        entity: Entity,
        from: WorldId,
        to: WorldId,
    );

    /// This query will decide if `entity_spawned` & `entity_despawned` should
    /// be run for given entities. If there is not query, the system will run
    /// on every entity.
    fn query(&self) -> Query;
}

/// Run for [`Entity`]s that match the query
#[system_trait]
pub trait TickingSystem: System {
    /// `entities`: the list of entities that were returned by the query returned
    /// by [`TickingSystem::query`].
    fn tick(
        &self,
        cmd: &mut CommandBuffer,
        universe: &Universe,
        delta_time: f64,
        entities: &mut dyn Iterator<Item = Entity>,
    );
    /// Returns the query used to construct the `entities` of the tick function.
    fn query(&self) -> Query;
}

/// System run for every component of the specified type.
/// Can be registered on both the [`Entity`] & [`Universe`]
pub trait ComponentSystem: System {
    /// The concrete component type this system handles.
    type Component: Component;

    /// When a component is added.
    /// `entity`: the entity with this component.
    fn added(&self, cmd: &mut CommandBuffer, entity: Entity, component: &Self::Component);

    /// When a component is updated.
    /// `entity`: the entity with this component.
    fn updated(
        &self,
        cmd: &mut CommandBuffer,
        entity: Entity,
        old: &Self::Component,
        new: &Self::Component,
    );

    /// When a component is removed.
    /// `entity`: the entity with this component.
    fn removed(&self, cmd: &mut CommandBuffer, entity: Entity, component: &Self::Component);
}

/// Private type-erasure trait for storage in [`Entity`] & [`Universe`]
pub(crate) trait AnyComponentSystem {
    /// Returns the [`TypeId`] of the [`Component`] that this system is running for.
    fn component_type_id(&self) -> TypeId;
    fn added(&self, cmd: &mut CommandBuffer, entity: Entity, component: &dyn AnyComponent);
    fn updated(
        &self,
        cmd: &mut CommandBuffer,
        entity: Entity,
        old: &dyn AnyComponent,
        new: &dyn AnyComponent,
    );
    fn removed(&self, cmd: &mut CommandBuffer, entity: Entity, component: &dyn AnyComponent);
}

impl<T: ComponentSystem> AnyComponentSystem for T {
    fn component_type_id(&self) -> TypeId {
        TypeId::of::<T::Component>()
    }

    fn added(&self, cmd: &mut CommandBuffer, entity: Entity, component: &dyn AnyComponent) {
        if let Some(component) = component.as_any().downcast_ref::<T::Component>() {
            T::added(self, cmd, entity, component);
        }
    }

    fn updated(
        &self,
        cmd: &mut CommandBuffer,
        entity: Entity,
        old: &dyn AnyComponent,
        new: &dyn AnyComponent,
    ) {
        let Some(old) = old.as_any().downcast_ref::<T::Component>() else {
            return;
        };
        let Some(new) = new.as_any().downcast_ref::<T::Component>() else {
            return;
        };

        T::updated(self, cmd, entity, old, new);
    }

    fn removed(&self, cmd: &mut CommandBuffer, entity: Entity, component: &dyn AnyComponent) {
        if let Some(component) = component.as_any().downcast_ref::<T::Component>() {
            T::removed(self, cmd, entity, component);
        }
    }
}

#[derive(Default)]
pub(crate) struct ComponentSystemStorage {
    systems: HashMap<TypeId, Vec<Box<dyn AnyComponentSystem>>>,
}

impl ComponentSystemStorage {
    pub fn push<S: ComponentSystem>(&mut self, system: S) {
        self.systems
            .entry(system.component_type_id())
            .or_default()
            .push(Box::new(system));
    }

    pub fn push_any(&mut self, type_id: TypeId, system: Box<dyn AnyComponentSystem>) {
        self.systems.entry(type_id).or_default().push(system);
    }

    pub fn iter(&self, type_id: TypeId) -> std::slice::Iter<'_, Box<dyn AnyComponentSystem>> {
        // fixing the lint breaks because of some weird type stuff, so we allow.
        #[allow(clippy::map_unwrap_or)]
        self.systems
            .get(&type_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
    }
}

impl IntoIterator for ComponentSystemStorage {
    type Item = (TypeId, Vec<Box<dyn AnyComponentSystem>>);
    type IntoIter = std::collections::hash_map::IntoIter<TypeId, Vec<Box<dyn AnyComponentSystem>>>;
    fn into_iter(self) -> Self::IntoIter {
        self.systems.into_iter()
    }
}
