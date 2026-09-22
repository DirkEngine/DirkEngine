#![doc = include_str!("../README.md")]

use std::{
    any::TypeId,
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::mpsc::{self, Receiver, Sender},
};
use tracing::warn;

pub mod components;
use components::{AnyComponent, Component, Components};

pub mod query;

pub mod systems;
use systems::{
    ComponentSystem, ComponentSystemStorage, EntitySystem, EntitySystemStorage, TickingSystem,
    TickingSystemStorage, UniverseSystem, UniverseSystemStorage, experimental::StandaloneSystem,
};

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

    // this field starts with `universe`
    #[allow(clippy::struct_field_names)]
    universe_systems: UniverseSystemStorage,
    ticking_systems: TickingSystemStorage,
    entity_systems: EntitySystemStorage,
    component_systems: ComponentSystemStorage,

    // Systems mutate their own state while borrowing the universe's component
    // data. Only tick borrows this private list, so callbacks cannot reborrow it.
    systems: RefCell<Vec<Box<dyn StandaloneSystem>>>,

    components: Components,
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
            universe_systems: builder.universe_systems,
            ticking_systems: builder.ticking_systems,
            entity_systems: builder.entity_systems,
            component_systems: builder.component_systems,
            systems: RefCell::new(builder.systems),
            components: Components::default(),
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
    /// Experimental systems run after the legacy universe and ticking systems,
    /// in registration order. Commands produced by any system become visible
    /// on the next tick. `delta_time` is measured in seconds.
    ///
    /// # Panics
    ///
    /// Will panic in certain internal conditions like if a [`World`] that
    /// was just created is not found in the [`Universe`].
    /// Panics from system callbacks propagate to the caller.
    pub fn tick(&mut self, delta_time: f64) {
        let mut cmd = self.handle.command_buffer();

        let mut commands: Vec<Command> = Vec::new();
        for sub in self.buffer_receiver.try_iter() {
            commands.append(&mut sub.commands());
        }

        self.run_commands(&mut cmd, commands);

        self.universe_systems
            .iter()
            .for_each(|system| system.tick(&mut cmd, self, delta_time));

        self.ticking_systems.iter().for_each(|system| {
            system.tick(&mut cmd, self, delta_time, &mut system.query().query(self));
        });

        for system in self.systems.borrow_mut().iter_mut() {
            system.run(&mut cmd, self, delta_time);
        }

        cmd.submit();
    }

    // HELPERS FOR THE TICK FUNCTION

    #[allow(clippy::too_many_lines)]
    fn run_commands(&mut self, cmd: &mut CommandBuffer, commands: Vec<Command>) {
        let mut created_worlds: HashSet<WorldId> = HashSet::new();
        let mut destroyed_worlds: HashSet<WorldId> = HashSet::new();
        let mut spawned_entities: HashSet<Entity> = HashSet::new();
        let mut despawned_entities: HashSet<Entity> = HashSet::new();
        let mut sent_entities: HashSet<(Entity, WorldId, WorldId)> = HashSet::new(); // entity, from, to
        let mut added_components: HashSet<(Entity, TypeId)> = HashSet::new();
        let mut updated_components: Vec<(Entity, TypeId, Box<dyn AnyComponent>)> = Vec::new(); // we use a vec as updated_components is not `Hash`
        let mut removed_components: HashSet<(Entity, TypeId)> = HashSet::new();

        for command in commands {
            match command {
                Command::CreateWorld(id, name) => {
                    if self.worlds.contains_key(&id) {
                        warn!("cannot create world {id} as it already exists");
                        continue;
                    }
                    let world = World::new(id, name);
                    self.worlds.insert(id, world);

                    created_worlds.insert(id);
                }
                Command::DestroyWorld(world) => {
                    let Some(world) = self.worlds.get(&world) else {
                        continue;
                    };

                    for &entity in &world.alive {
                        despawned_entities.insert(entity);
                        for (type_id, _) in self.components.get_all(entity) {
                            removed_components.insert((entity, type_id));
                        }
                    }
                    destroyed_worlds.insert(world.id());
                }
                Command::Spawn(entity, builder, world) => {
                    if self.is_alive(entity) {
                        warn!("cannot add entity {entity:?} as it already exists");
                        continue;
                    }
                    let Some(world_ref) = self.worlds.get_mut(&world) else {
                        warn!(
                            "cannot add entity {entity:?} to world {world:?} as the world does not exist"
                        );
                        continue;
                    };

                    self.entities.insert(entity, world);

                    spawned_entities.insert(entity);

                    builder.components.into_values().for_each(|component| {
                        let type_id = component.component_type_id();
                        self.components.insert_any(entity, component);
                        added_components.insert((entity, type_id));
                    });

                    world_ref.alive.insert(entity);
                }
                Command::Despawn(entity) => {
                    if !self.is_alive(entity) {
                        continue;
                    }
                    despawned_entities.insert(entity);
                    for (type_id, _) in self.components.get_all(entity) {
                        removed_components.insert((entity, type_id));
                    }
                }
                Command::Send(entity, to) => {
                    let Some(from) = self.entities.get(&entity).copied() else {
                        warn!("cannot send {entity:?} as it does not exist");
                        continue;
                    };
                    if from == to {
                        continue;
                    }
                    if !self.worlds.contains_key(&to) {
                        warn!("cannot send {entity:?} to world {to} as it does not exist");
                        continue;
                    }

                    let old = self
                        .worlds
                        .get_mut(&from)
                        .expect("entity is registered as in this world");
                    old.alive.remove(&entity);

                    let new = self.worlds.get_mut(&to).expect("just checked it exists");
                    new.alive.insert(entity);

                    self.entities.insert(entity, to);
                    sent_entities.insert((entity, from, to));
                }
                Command::SetComponent(entity, component) => {
                    if !self.is_alive(entity) {
                        continue;
                    }

                    let type_id = component.component_type_id();
                    if let Some(old) = self.components.insert_any(entity, component) {
                        updated_components.push((entity, type_id, old));
                    } else {
                        added_components.insert((entity, type_id));
                    }
                }
                Command::RemoveComponent(entity, type_id) => {
                    if self.components.get_any(entity, type_id).is_some() {
                        removed_components.insert((entity, type_id));
                    }
                }
            }
        }

        // World: created
        for world in created_worlds {
            let world = self.worlds.get(&world).expect("we just added this world");
            self.universe_systems
                .iter()
                .for_each(|system| system.world_created(cmd, self, world));
        }

        // Entity: spawned
        for entity in spawned_entities {
            self.universe_systems
                .iter()
                .for_each(|system| system.entity_spawned(cmd, self, entity));
            self.entity_systems.iter().for_each(|system| {
                if !system.query().matches(self, entity) {
                    return;
                }
                system.spawned(cmd, self, entity);
            });
        }

        // Component: added
        for (entity, type_id) in added_components {
            let component = self
                .components
                .get_any(entity, type_id)
                .expect("just added component");
            self.component_systems
                .iter(type_id)
                .for_each(|system| system.added(cmd, entity, component));
        }

        // Component: updated
        for (entity, type_id, old) in updated_components {
            let component = self
                .components
                .get_any(entity, type_id)
                .expect("just updated component");
            self.component_systems
                .iter(type_id)
                .for_each(|system| system.updated(cmd, entity, old.as_ref(), component));
        }

        // Entity: sent
        for (entity, from, to) in sent_entities {
            self.universe_systems
                .iter()
                .for_each(|system| system.entity_sent(cmd, self, entity, from, to));
            self.entity_systems.iter().for_each(|system| {
                if !system.query().matches(self, entity) {
                    return;
                }

                system.sent(cmd, self, entity, from, to);
            });
        }

        // Component: removed
        for &(entity, type_id) in &removed_components {
            let component = self
                .components
                .get_any(entity, type_id)
                .expect("haven't removed component yet");
            self.component_systems
                .iter(type_id)
                .for_each(|system| system.removed(cmd, entity, component));
        }

        // Entity: despawned
        for &entity in &despawned_entities {
            self.universe_systems
                .iter()
                .for_each(|system| system.entity_despawned(cmd, self, entity));
            self.entity_systems.iter().for_each(|system| {
                if !system.query().matches(self, entity) {
                    return;
                }

                system.despawned(cmd, self, entity);
            });
        }

        // World: destroyed
        for world in &destroyed_worlds {
            let world = self.worlds.get(world).expect("world not added yet");
            self.universe_systems
                .iter()
                .for_each(|system| system.world_destroyed(cmd, self, world));
        }

        // destroy components
        for (entity, type_id) in removed_components {
            self.components.remove_any(entity, type_id);
        }

        // destroy entities
        for entity in despawned_entities {
            if let Some(world) = self.get_world(entity)
                && let Some(world) = self.worlds.get_mut(&world)
            {
                world.alive.remove(&entity);
            }
            self.entities.remove(&entity);
        }

        // destroy worlds
        for world in destroyed_worlds {
            self.worlds.remove(&world);
        }
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
    universe_systems: UniverseSystemStorage,
    ticking_systems: TickingSystemStorage,
    entity_systems: EntitySystemStorage,
    component_systems: ComponentSystemStorage,
    systems: Vec<Box<dyn StandaloneSystem>>,
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
            universe_systems: UniverseSystemStorage::default(),
            ticking_systems: TickingSystemStorage::default(),
            entity_systems: EntitySystemStorage::default(),
            component_systems: ComponentSystemStorage::default(),
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

    /// Adds a [`UniverseSystem`] that will be added to the [`Universe`].
    #[must_use]
    pub fn with_universe_system(mut self, system: impl UniverseSystem) -> Self {
        self.universe_systems.push(system);
        self
    }

    /// Adds a [`EntitySystem`] that will be added to the [`Universe`].
    #[must_use]
    pub fn with_entity_system(mut self, system: impl EntitySystem) -> Self {
        self.entity_systems.push(system);
        self
    }

    /// Adds a [`TickingSystem`] that will be added to the [`Universe`].
    #[must_use]
    pub fn with_ticking_system(mut self, system: impl TickingSystem) -> Self {
        self.ticking_systems.push(system);
        self
    }

    /// Adds an experimental system to run on each tick, in registration order.
    ///
    /// Use [`systems::experimental::FuncSystem::new`] to wrap a function or
    /// stateful closure. Systems run after legacy ticking systems and share
    /// their command buffer; writes are applied on the following tick.
    #[must_use]
    pub fn with_system(mut self, system: impl StandaloneSystem) -> Self {
        self.systems.push(Box::new(system));
        self
    }

    /// Adds a [`ComponentSystem`] that will be added to the [`Universe`].
    #[must_use]
    pub fn with_component_system(mut self, system: impl ComponentSystem) -> Self {
        self.component_systems.push(system);
        self
    }

    /// Will combine the `other` [`UniverseBuilder`] with this one.
    #[must_use]
    pub fn with_other(mut self, other: Self) -> Self {
        for world in other.worlds {
            self.worlds.push(world);
        }

        for system in other.universe_systems {
            self.universe_systems.push_any(system);
        }

        for system in other.entity_systems {
            self.entity_systems.push_any(system);
        }

        for system in other.ticking_systems {
            self.ticking_systems.push_any(system);
        }

        for (type_id, systems) in other.component_systems {
            for system in systems {
                self.component_systems.push_any(type_id, system);
            }
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
