#![doc = include_str!("../README.md")]

use std::{
    any::TypeId,
    cell::RefCell,
    collections::HashMap,
    fmt::Debug,
    sync::mpsc::{self, Receiver, Sender},
};
use tracing::warn;

pub mod components;
use components::{AnyComponent, Component, ComponentValue, Components};

pub mod changes;
use changes::{Change, ComponentChange};

pub mod query;

pub mod systems;
use systems::{ErasedSystem, ToSystem};

mod command_buffer;
use command_buffer::Command;
pub use command_buffer::CommandBuffer;

mod entity;
pub use entity::{Entity, EntityBuilder};

mod world;
pub use world::{World, WorldBuilder, WorldId};

mod allocator;
use allocator::Allocator;

/// Read-only information about one component attached to an entity.
pub struct ComponentInfo<'a> {
    /// Component [`TypeId`].
    pub type_id: TypeId,
    /// Fully-qualified Rust component type name.
    pub type_name: &'static str,
    /// Debug view of the component value.
    pub debug: &'a dyn Debug,
}

/// A cheap clonable handle for access to the [`Universe`].
/// This give write-only access to the [`Universe`]. To write to it, use
/// a [`System`].
///
/// [`System`]: systems::System
#[derive(Clone)]
pub struct UniverseHandle {
    allocator: Allocator,
    buffer_sender: Sender<CommandBuffer>,
}

impl UniverseHandle {
    /// Creates a command buffer
    #[must_use]
    pub fn command_buffer(&self) -> CommandBuffer {
        CommandBuffer::new(self.clone())
    }
}

/// This struct is the manager for all the worlds.
pub struct Universe {
    worlds: HashMap<WorldId, World>,
    entities: HashMap<Entity, WorldId>,

    handle: UniverseHandle,
    buffer_receiver: Receiver<CommandBuffer>,

    // Systems mutate their own state while borrowing the universe's component
    // data. Only tick borrows this private list, so callbacks cannot reborrow it.
    systems: RefCell<Vec<Box<dyn ErasedSystem>>>,

    components: Components,
    changes: Vec<Change>,
    pending_changes: Vec<Change>,
    edit_order: RefCell<Vec<(Entity, TypeId)>>,
    prepared_order: RefCell<Vec<(Entity, TypeId)>>,
}

impl Universe {
    /// Returns a [`UniverseBuilder`] to easily construct a [`Universe`].
    #[must_use]
    pub fn builder() -> UniverseBuilder {
        UniverseBuilder::new()
    }

    #[must_use]
    fn build(builder: UniverseBuilder) -> Self {
        let universe = Self {
            worlds: HashMap::new(),
            entities: HashMap::new(),
            handle: builder.handle,
            buffer_receiver: builder.buffer_receiver,
            systems: RefCell::new(builder.systems),
            components: Components::default(),
            changes: Vec::new(),
            pending_changes: Vec::new(),
            edit_order: RefCell::new(Vec::new()),
            prepared_order: RefCell::new(Vec::new()),
        };

        let mut cmd = universe.handle.command_buffer();
        for builder in builder.worlds {
            cmd.create_world(builder);
        }
        cmd.submit();
        universe
    }

    /// Returns a cheap handle to the [`Universe`].
    #[must_use]
    pub fn handle(&self) -> UniverseHandle {
        self.handle.clone()
    }

    /// Applies queued commands, then runs systems against the resulting universe.
    ///
    /// Systems run once each, in registration order, even with no entities.
    /// Commands produced by any system become visible on the next tick.
    /// Mutable query edits are visible to later systems in the same tick.
    /// `delta_time` is measured in seconds.
    ///
    /// # Panics
    ///
    /// Will panic in certain internal conditions like if a [`World`] that
    /// was just created is not found in the [`Universe`].
    /// Panics from system callbacks propagate to the caller.
    pub fn tick(&mut self, delta_time: f64) {
        let cmd = RefCell::new(self.handle.command_buffer());

        let mut commands: Vec<Command> = Vec::new();
        for sub in self.buffer_receiver.try_iter() {
            commands.append(&mut sub.commands());
        }

        self.changes.clear();
        self.changes.append(&mut self.pending_changes);
        self.run_commands(commands);

        for system in self.systems.borrow_mut().iter_mut() {
            system.run(self, delta_time, &cmd);
            self.pending_changes.extend(
                self.components
                    .finish_edits(
                        &mut self.edit_order.borrow_mut(),
                        &mut self.prepared_order.borrow_mut(),
                    )
                    .into_iter()
                    .map(|(entity, old, new)| Change::ComponentUpdated { entity, old, new }),
            );
        }

        cmd.into_inner().submit();
    }

    fn run_commands(&mut self, commands: Vec<Command>) {
        for command in commands {
            self.apply_command(command);
        }
    }

    fn apply_command(&mut self, command: Command) {
        match command {
            Command::CreateWorld(id, name) => {
                if self.worlds.contains_key(&id) {
                    warn!("cannot create world {id} as it already exists");
                    return;
                }
                self.worlds.insert(id, World::new(id, name));
                self.changes.push(Change::WorldCreated { world: id });
            }
            Command::DestroyWorld(id) => {
                let Some(world) = self.worlds.get(&id) else {
                    return;
                };
                let entities: Vec<_> = world.alive.iter().copied().collect();
                for entity in entities {
                    self.despawn(entity);
                }
                self.worlds.remove(&id);
                self.changes.push(Change::WorldDestroyed { world: id });
            }
            Command::Spawn(entity, builder, world) => {
                if self.is_alive(entity) {
                    warn!("cannot spawn {entity:?} as it already exists");
                    return;
                }
                let Some(world_ref) = self.worlds.get_mut(&world) else {
                    warn!("cannot spawn {entity:?} in missing world {world:?}");
                    return;
                };
                world_ref.alive.insert(entity);
                self.entities.insert(entity, world);
                self.changes.push(Change::EntitySpawned { entity, world });
                for component in builder.components.into_values() {
                    self.set_component(entity, component);
                }
            }
            Command::Despawn(entity) => self.despawn(entity),
            Command::Send(entity, to) => {
                let Some(from) = self.get_world(entity) else {
                    warn!("cannot send {entity:?} as it does not exist");
                    return;
                };
                if from == to {
                    return;
                }
                if !self.worlds.contains_key(&to) {
                    warn!("cannot send {entity:?} to missing world {to}");
                    return;
                }
                self.worlds
                    .get_mut(&from)
                    .expect("entity's world exists")
                    .alive
                    .remove(&entity);
                self.worlds
                    .get_mut(&to)
                    .expect("destination exists")
                    .alive
                    .insert(entity);
                self.entities.insert(entity, to);
                self.changes.push(Change::EntityMoved { entity, from, to });
            }
            Command::SetComponent(entity, component) => {
                if self.is_alive(entity) {
                    self.set_component(entity, component);
                }
            }
            Command::RemoveComponent(entity, type_id) => self.remove_component(entity, type_id),
        }
    }

    fn set_component(&mut self, entity: Entity, component: Box<dyn AnyComponent>) {
        let new = ComponentValue(component.into());
        let change = match self.components.insert(entity, new.clone()) {
            Some(old) => Change::ComponentUpdated { entity, old, new },
            None => Change::ComponentAdded {
                entity,
                component: new,
            },
        };
        self.changes.push(change);
    }

    fn remove_component(&mut self, entity: Entity, type_id: TypeId) {
        if let Some(component) = self.components.remove(entity, type_id) {
            self.changes
                .push(Change::ComponentRemoved { entity, component });
        }
    }

    fn despawn(&mut self, entity: Entity) {
        let Some(world) = self.get_world(entity) else {
            return;
        };
        let types: Vec<_> = self.components.get_all(entity).map(|(id, _)| id).collect();
        for type_id in types {
            self.remove_component(entity, type_id);
        }
        self.worlds
            .get_mut(&world)
            .expect("entity's world exists")
            .alive
            .remove(&entity);
        self.entities.remove(&entity);
        self.changes.push(Change::EntityDespawned { entity, world });
    }

    /// Returns this tick's changes in command application order.
    ///
    /// All systems see the same log. Values survive removal and intermediate
    /// replacements; the log is replaced at the start of the next tick.
    pub fn changes(&self) -> impl Iterator<Item = &Change> {
        self.changes.iter()
    }

    /// Returns typed changes for `C`, preserving their application order.
    pub fn component_changes<C: Component>(&self) -> impl Iterator<Item = ComponentChange<'_, C>> {
        self.changes.iter().filter_map(Change::component::<C>)
    }

    // UTILITIES & GETTERS

    /// Returns an optional reference to the requested [`World`].
    #[must_use]
    pub fn world(&self, world: WorldId) -> Option<&World> {
        self.worlds.get(&world)
    }

    /// Returns all live worlds.
    pub fn worlds(&self) -> impl Iterator<Item = &World> {
        self.worlds.values()
    }

    /// Returns all live entities with their current world.
    pub fn entities(&self) -> impl Iterator<Item = (Entity, WorldId)> + '_ {
        self.entities
            .iter()
            .map(|(entity, world)| (*entity, *world))
    }

    /// Returns all live entities currently in `world`.
    pub fn entities_in_world(&self, world: WorldId) -> impl Iterator<Item = Entity> + '_ {
        self.entities
            .iter()
            .filter_map(move |(entity, entity_world)| {
                if *entity_world == world {
                    Some(*entity)
                } else {
                    None
                }
            })
    }

    /// Returns read-only component information for `entity`.
    pub fn component_infos(&self, entity: Entity) -> impl Iterator<Item = ComponentInfo<'_>> {
        self.components
            .get_all(entity)
            .map(|(type_id, component)| ComponentInfo {
                type_id,
                type_name: component.component_type_name(),
                debug: component,
            })
    }

    /// Returns the [`WorldId`] of the [`Entity`]'s [`World`].
    #[must_use]
    pub fn get_world(&self, entity: Entity) -> Option<WorldId> {
        self.entities.get(&entity).copied()
    }

    /// Returns if the given [`Entity`] is in the given [`World`].
    #[must_use]
    pub fn is_in_world(&self, world: WorldId, entity: Entity) -> bool {
        self.entities.get(&entity) == Some(&world)
    }

    /// Returns the total number of alive entities.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns if the specified entity is alive
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.contains_key(&entity)
    }

    /// Returns a shared reference to a component, or `None` if the entity
    /// does not have one.
    #[must_use]
    pub fn component<C: Component>(&self, entity: Entity) -> Option<&C> {
        self.components.get(entity)
    }
}

/// Builder struct used to construct a [`Universe`].
pub struct UniverseBuilder {
    handle: UniverseHandle,
    buffer_receiver: Receiver<CommandBuffer>,
    worlds: Vec<WorldBuilder>,
    systems: Vec<Box<dyn ErasedSystem>>,
}

impl UniverseBuilder {
    #[must_use]
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            handle: UniverseHandle {
                allocator: Allocator::new(),
                buffer_sender: sender,
            },
            buffer_receiver: receiver,
            worlds: Vec::new(),
            systems: Vec::new(),
        }
    }

    /// Returns the handle that will access the built [`Universe`].
    #[must_use]
    pub fn handle(&self) -> UniverseHandle {
        self.handle.clone()
    }

    /// Will build a new [`Universe`] from the current builder.
    #[must_use]
    pub fn build(self) -> Universe {
        Universe::build(self)
    }

    /// Adds a [`World`] that will be created at the same time as the [`Universe`].
    #[must_use]
    pub fn with_world(mut self, builder: WorldBuilder) -> Self {
        self.worlds.push(builder);
        self
    }

    /// Adds a system to run on each tick, in registration order.
    ///
    /// Functions declare the queries, changes, and delta time they need in
    /// their arguments. Stateful [`systems::System`] implementations use
    /// the same parameters. Writes are applied on the following tick.
    #[must_use]
    pub fn with_system<Marker>(mut self, system: impl ToSystem<Marker>) -> Self {
        self.systems.push(system.to_system());
        self
    }

    /// Appends the worlds and systems configured on `other`.
    ///
    /// Only configuration can be merged. Handles from independent builders
    /// allocate overlapping IDs and submit to different queues, so `other`
    /// must not have issued a handle or command buffer that is still alive.
    ///
    /// # Panics
    ///
    /// Panics if `other` has a live handle or a queued command buffer. Build
    /// it separately instead, or submit commands through this builder's handle.
    #[must_use]
    pub fn with_other(mut self, other: Self) -> Self {
        assert!(
            other.handle.allocator.is_unique(),
            "cannot merge a UniverseBuilder with issued handles or queued commands"
        );
        for world in other.worlds {
            self.worlds.push(world);
        }

        self.systems.extend(other.systems);

        self
    }
}

impl Default for UniverseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
