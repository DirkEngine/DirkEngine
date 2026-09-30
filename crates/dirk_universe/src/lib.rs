#![doc = include_str!("../README.md")]

// Lets `#[derive(Component)]` name `::dirk_universe` inside this crate too.
extern crate self as dirk_universe;

use std::{
    any::TypeId,
    cell::{Cell, Ref, RefCell},
    collections::HashSet,
    fmt::Debug,
    sync::mpsc::{self, Receiver, Sender},
};
use tracing::warn;

// Declared first so its tuple macros are in scope for the modules below.
#[macro_use]
mod macros;

pub mod components;
use components::{AnyComponent, Component, Components};

pub mod query;
use query::{Join, Query, ReadOnlyQueryData, filter::QueryFilter};

pub mod systems;
use systems::{ErasedSystem, IntoSystem};

pub mod schedule;
use schedule::{Schedule, ScheduleError, SystemConfig};

mod command_buffer;
use command_buffer::Command;
pub use command_buffer::CommandBuffer;

mod entity;
pub use entity::{Entity, EntityBuilder};

mod world;
pub use world::{InWorld, World, WorldBuilder, WorldId};

mod allocator;
use allocator::Allocator;

/// The types needed to define components and systems.
pub mod prelude {
    pub use crate::{
        CommandBuffer, Entity, InWorld, Universe, UniverseHandle, World, WorldId,
        components::Component,
        query::{
            Delta, Lagged, Query,
            filter::{Added, Changed, With, Without},
        },
        systems::{Commands, DeltaTime, IntoSystem, System},
    };
}

/// Read-only information about one component attached to an entity.
pub struct ComponentInfo<'a> {
    /// Component [`TypeId`].
    pub type_id: TypeId,
    /// Fully-qualified Rust component type name.
    pub type_name: &'static str,
    /// Debug view of the component value.
    pub debug: Ref<'a, dyn Debug>,
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

/// A component removal, kept until every system has observed it.
struct Removal {
    type_id: TypeId,
    entity: Entity,
    /// When the removed component was added, so additions and removals that
    /// cancel out between two runs are not reported.
    added: u64,
    tick: u64,
}

/// This struct is the manager for all the worlds.
///
/// Worlds are entities carrying a [`World`] component. Every other entity
/// lives in a world, recorded by its [`InWorld`] component.
pub struct Universe {
    alive: HashSet<Entity>,

    handle: UniverseHandle,
    buffer_receiver: Receiver<CommandBuffer>,

    // Systems mutate their own state while borrowing the universe's component
    // data. Only tick borrows this private list, so callbacks cannot reborrow it.
    systems: RefCell<Vec<Box<dyn ErasedSystem>>>,
    schedule: Schedule,

    components: Components,
    removals: Vec<Removal>,
    change_tick: Cell<u64>,
    /// The change tick at which the previous tick started.
    previous_tick: u64,
}

impl Universe {
    /// Returns a [`UniverseBuilder`] to easily construct a [`Universe`].
    #[must_use]
    pub fn builder() -> UniverseBuilder {
        UniverseBuilder::new()
    }

    fn build(builder: UniverseBuilder) -> Result<Self, ScheduleError> {
        let (systems, schedule) = schedule::build(builder.systems)?;
        let universe = Self {
            alive: HashSet::new(),
            handle: builder.handle,
            buffer_receiver: builder.buffer_receiver,
            systems: RefCell::new(systems),
            schedule,
            components: Components::default(),
            removals: Vec::new(),
            change_tick: Cell::new(0),
            previous_tick: 0,
        };

        let mut cmd = universe.handle.command_buffer();
        for builder in builder.worlds {
            cmd.create_world(builder);
        }
        cmd.submit();
        Ok(universe)
    }

    /// Returns a cheap handle to the [`Universe`].
    #[must_use]
    pub fn handle(&self) -> UniverseHandle {
        self.handle.clone()
    }

    /// Returns the order in which systems run, with the reasons for it.
    #[must_use]
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Applies queued commands, then runs systems against the resulting universe.
    ///
    /// Systems run once each, in [`schedule`](Self::schedule) order, even
    /// with no entities.
    /// Commands produced by any system become visible on the next tick.
    /// Mutable query edits are visible to later systems in the same tick.
    /// `delta_time` is measured in seconds.
    ///
    /// # Panics
    ///
    /// Panics from system callbacks propagate to the caller.
    pub fn tick(&mut self, delta_time: f64) {
        let cmd = RefCell::new(self.handle.command_buffer());

        let mut commands: Vec<Command> = Vec::new();
        for sub in self.buffer_receiver.try_iter() {
            commands.append(&mut sub.commands());
        }

        // Every system ran after the previous tick started, so removals
        // recorded before then have been observed by all of them.
        let tick = self.advance_change_tick();
        let previous_tick = std::mem::replace(&mut self.previous_tick, tick);
        self.removals.retain(|removal| removal.tick > previous_tick);
        for command in commands {
            self.apply_command(command);
        }

        for system in self.systems.borrow_mut().iter_mut() {
            system.run(self, delta_time, &cmd);
        }

        cmd.into_inner().submit();
    }

    fn apply_command(&mut self, command: Command) {
        match command {
            Command::CreateWorld(id, name) => {
                if self.is_alive(id.entity()) {
                    warn!("cannot create world {id} as it already exists");
                    return;
                }
                self.alive.insert(id.entity());
                self.set_component(id.entity(), Box::new(World::new(id, name)));
            }
            Command::DestroyWorld(id) => self.destroy_world(id),
            Command::Spawn(entity, builder, world) => {
                if self.is_alive(entity) {
                    warn!("cannot spawn {entity:?} as it already exists");
                    return;
                }
                if self.world(world).is_none() {
                    warn!("cannot spawn {entity:?} in missing world {world}");
                    return;
                }
                self.alive.insert(entity);
                self.set_component(entity, Box::new(InWorld(world)));
                for component in builder.components.into_values() {
                    self.set_component(entity, component);
                }
            }
            Command::Despawn(entity) => {
                if self.component::<World>(entity).is_some() {
                    self.destroy_world(WorldId::new(entity));
                } else {
                    self.despawn(entity);
                }
            }
            Command::Send(entity, to) => {
                let Some(from) = self.get_world(entity) else {
                    warn!("cannot send {entity:?} as it does not exist");
                    return;
                };
                if from == to {
                    return;
                }
                if self.world(to).is_none() {
                    warn!("cannot send {entity:?} to missing world {to}");
                    return;
                }
                self.set_component(entity, Box::new(InWorld(to)));
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
        self.components
            .insert(entity, component, self.change_tick.get());
    }

    fn remove_component(&mut self, entity: Entity, type_id: TypeId) {
        if let Some(added) = self.components.remove(entity, type_id) {
            self.removals.push(Removal {
                type_id,
                entity,
                added,
                tick: self.change_tick.get(),
            });
        }
    }

    fn despawn(&mut self, entity: Entity) {
        if !self.alive.remove(&entity) {
            return;
        }
        let types: Vec<_> = self.components.get_all(entity).map(|(id, _)| id).collect();
        for type_id in types {
            self.remove_component(entity, type_id);
        }
    }

    fn destroy_world(&mut self, world: WorldId) {
        if self.world(world).is_none() {
            return;
        }
        let entities: Vec<_> = self.entities_in_world(world).collect();
        for entity in entities {
            self.despawn(entity);
        }
        self.despawn(world.entity());
    }

    /// Entities whose `C` was removed after `last_run`, having existed then.
    pub(crate) fn removed_since<C: Component>(
        &self,
        last_run: u64,
    ) -> impl Iterator<Item = Entity> + '_ {
        self.removals
            .iter()
            .filter(move |removal| {
                removal.type_id == TypeId::of::<C>()
                    && removal.tick > last_run
                    && removal.added <= last_run
            })
            .map(|removal| removal.entity)
    }

    // A separate counter for each invocation makes same-frame ordering visible.
    fn advance_change_tick(&self) -> u64 {
        let tick = self
            .change_tick
            .get()
            .checked_add(1)
            .expect("change counter exhausted");
        self.change_tick.set(tick);
        tick
    }

    // UTILITIES & GETTERS

    /// Returns the [`World`] component of a live world.
    #[must_use]
    pub fn world(&self, world: WorldId) -> Option<Ref<'_, World>> {
        self.component(world.entity())
    }

    /// Returns all live worlds.
    pub fn worlds(&self) -> impl Iterator<Item = Ref<'_, World>> {
        self.components.iter::<World>().map(|(_, world)| world)
    }

    /// Returns every entity living in a world, with that world. World
    /// entities themselves are not included.
    pub fn entities(&self) -> impl Iterator<Item = (Entity, WorldId)> + '_ {
        self.components
            .iter::<InWorld>()
            .map(|(entity, in_world)| (entity, in_world.0))
    }

    /// Returns all live entities currently in `world`.
    pub fn entities_in_world(&self, world: WorldId) -> impl Iterator<Item = Entity> + '_ {
        self.entities()
            .filter_map(move |(entity, entity_world)| (entity_world == world).then_some(entity))
    }

    /// Returns read-only component information for `entity`.
    pub fn component_infos(&self, entity: Entity) -> impl Iterator<Item = ComponentInfo<'_>> {
        self.components
            .get_all(entity)
            .map(|(type_id, component)| ComponentInfo {
                type_id,
                type_name: component.component_type_name(),
                debug: Ref::map(component, |value| value as &dyn Debug),
            })
    }

    /// Returns the [`WorldId`] of the [`Entity`]'s [`World`].
    #[must_use]
    pub fn get_world(&self, entity: Entity) -> Option<WorldId> {
        self.component::<InWorld>(entity).map(|in_world| in_world.0)
    }

    /// Returns if the given [`Entity`] is in the given [`World`].
    #[must_use]
    pub fn is_in_world(&self, world: WorldId, entity: Entity) -> bool {
        self.get_world(entity) == Some(world)
    }

    /// Returns if the specified entity, which may be a world, is alive.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.alive.contains(&entity)
    }

    /// Returns a read-only query over the current universe, for use outside
    /// systems. `Added`, `Changed` and `Delta` treat every present component
    /// as new, since there is no previous run to compare against.
    #[must_use]
    pub fn query<D: ReadOnlyQueryData>(&self) -> Query<'_, D> {
        Query::new(self, 0)
    }

    /// Like [`Universe::query`], restricted by the filter `F`.
    #[must_use]
    pub fn query_filtered<D: ReadOnlyQueryData, F: QueryFilter>(&self) -> Query<'_, D, F>
    where
        D::Liveness: Join<F::Liveness>,
    {
        Query::new(self, 0)
    }

    /// Returns a shared borrow of a component, or `None` if the entity
    /// does not have one.
    #[must_use]
    pub fn component<C: Component>(&self, entity: Entity) -> Option<Ref<'_, C>> {
        self.components.get(entity)
    }
}

/// Builder struct used to construct a [`Universe`].
pub struct UniverseBuilder {
    handle: UniverseHandle,
    buffer_receiver: Receiver<CommandBuffer>,
    worlds: Vec<WorldBuilder>,
    systems: Vec<SystemConfig>,
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

    /// Will build a new [`Universe`] from the current builder, ordering its
    /// systems from the data they access.
    ///
    /// # Errors
    ///
    /// Returns a [`ScheduleError`] when systems conflict: one system borrows
    /// a component incompatibly, two systems write a component in no defined
    /// order, systems depend on each other in a cycle, or a system is ordered
    /// against one that was never registered.
    pub fn build(self) -> Result<Universe, ScheduleError> {
        Universe::build(self)
    }

    /// Adds a [`World`] that will be created at the same time as the [`Universe`].
    #[must_use]
    pub fn with_world(mut self, builder: WorldBuilder) -> Self {
        self.worlds.push(builder);
        self
    }

    /// Adds a system to run on each tick. Systems reading a component run
    /// after the systems writing it; see [`schedule`] for the full rules.
    ///
    /// Functions declare queries, commands, and delta time through their
    /// arguments. Stateful [`systems::System`] implementations use
    /// the same parameters. Component edits are immediate; structural commands
    /// are applied on the following tick.
    #[must_use]
    pub fn with_system<Marker>(mut self, system: impl IntoSystem<Marker>) -> Self {
        self.systems.push(system.into_config());
        self
    }

    /// Appends the worlds and systems configured on `other`. Systems that
    /// depend on each other are ordered the same whatever the merge order;
    /// unrelated systems keep their registration order.
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
