use std::any::TypeId;

use tracing::warn;

use crate::{
    Entity, EntityBuilder, UniverseHandle, WorldBuilder, WorldId,
    components::{AnyComponent, Component},
};

/// A buffer to record edits to the [`Universe`](crate::Universe).
/// Commands are applied in submission order on the next tick.
pub struct CommandBuffer {
    commands: Vec<Command>,
    handle: UniverseHandle,
}

pub(crate) enum Command {
    CreateWorld(WorldId, String),
    DestroyWorld(WorldId),
    Spawn(Entity, EntityBuilder, WorldId),
    Despawn(Entity),
    Send(Entity, WorldId),
    SetComponent(Entity, Box<dyn AnyComponent>),
    RemoveComponent(Entity, TypeId),
}

impl CommandBuffer {
    /// Creates a new empty command buffer
    #[must_use]
    pub(crate) fn new(handle: UniverseHandle) -> Self {
        Self {
            commands: Vec::new(),
            handle,
        }
    }

    pub(crate) fn commands(self) -> Vec<Command> {
        self.commands
    }

    /// Will submit the [`CommandBuffer`] to the [`Universe`]'s queue.
    pub fn submit(self) {
        let sender = self.handle.buffer_sender.clone();
        if sender.send(self).is_err() {
            warn!("failed to submit CommandBuffer: receiver is closed");
        }
    }

    /// Returns if the command buffer has had no submitted commands. This is useful to skip
    /// all submission logic when no commands should be run.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    // WORLD MANAGEMENT

    /// Will create a new empty world & return its ID.
    pub fn create_world(&mut self, builder: WorldBuilder) -> WorldId {
        let id = self.handle.allocator.allocate_world();
        self.commands.push(Command::CreateWorld(id, builder.name));

        for entity in builder.entities {
            self.spawn(id, entity);
        }
        id
    }

    /// Destroys the world and its entities, recording their removal changes.
    pub fn destroy_world(&mut self, world: WorldId) {
        self.commands.push(Command::DestroyWorld(world));
    }

    // ENTITY MANAGEMENT

    /// Will spawn a new [`Entity`] using the provided [`EntityBuilder`].
    /// Returns the handle of the new [`Entity`].
    ///
    /// If the [`World`](crate::World) does not exist when this command is applied, the
    /// returned handle will not become alive.
    pub fn spawn(&mut self, world: WorldId, builder: EntityBuilder) -> Entity {
        let entity = self.handle.allocator.allocate_entity();
        self.commands.push(Command::Spawn(entity, builder, world));
        entity
    }

    /// Will despawn the provided [`Entity`].
    ///
    /// Records component removals before the entity despawn change.
    pub fn despawn(&mut self, entity: Entity) {
        self.commands.push(Command::Despawn(entity));
    }
    /// Will send the [`Entity`] to the specified [`WorldId`].
    pub fn send(&mut self, entity: Entity, to: WorldId) {
        self.commands.push(Command::Send(entity, to));
    }

    // COMPONENT MANAGEMENT

    /// Attaches a [`Component`] to [`Entity`], replacing any existing component of
    /// the same type.
    ///
    /// Records an addition or an update containing both the old and new values.
    ///
    /// [`Entity`]: crate::Entity
    pub fn set_component<C: Component>(&mut self, entity: Entity, component: C) {
        self.commands
            .push(Command::SetComponent(entity, Box::new(component)));
    }
    /// Removes a single component and records its old value if present.
    ///
    /// The entity itself is **not** despawned. If the component is not
    /// present this is a no-op.
    pub fn remove_component<C: Component>(&mut self, entity: Entity) {
        self.commands
            .push(Command::RemoveComponent(entity, TypeId::of::<C>()));
    }
}
