#![doc = include_str!("../README.md")]

use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Add, AddAssign},
    sync::Arc,
};

use dirk_engine::{EngineBuilder, EngineHandle, EnginePlugin, Subsystem};
use dirk_events::{Dispatcher, EventManager};
#[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
use dirk_platform::WindowId;
#[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
use dirk_platform::WindowInputEvent;
use dirk_universe::{Universe, components::Component};
use input::InputContext;
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard};

mod events;
pub mod input;
mod movement;
use events::PlayerInput;
pub use events::{PlayerDespawned, PlayerSpawned};
pub use movement::{DEFAULT_PLAYER_LOOK_SENSITIVITY, DEFAULT_PLAYER_MOVE_SPEED};

use crate::movement::PlayerMovementSystem;

/// Registers player management as an engine subsystem.
pub struct PlayerPlugin;

impl EnginePlugin for PlayerPlugin {
    fn name(&self) -> &'static str {
        "player"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.add_subsystem(|ctx| {
            let players = PlayerManager::new(ctx.events());
            ctx.add_resource(players.registry())?;
            ctx.add_resource(players.input_sender())?;
            #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
            ctx.add_resource(players.presentation_assignments())?;
            ctx.extend_universe(
                dirk_universe::Universe::builder()
                    .with_ticking_system(PlayerMovementSystem::new(players.registry.input_state())),
            );
            Ok(players)
        });
        Ok(())
    }
}

// PlayerId

/// A lightweight, copyable identifier for a player.
///
/// Implements [`Component`] so that ECS entities can declare ownership by
/// a player. This is the canonical link between [`PlayerRegistry`] and the
/// ECS: the manager never stores entity references; instead, game code attaches
/// a `PlayerId` component when spawning the player's entity.
///
/// # ECS Relationship
///
/// ```text
/// PlayerRegistry               Universe
///  └─ PlayerHandle(id=1) ◄──── Entity { PlayerId(1), Transform, Camera, ... }
/// ```
///
/// To find a player's entity, query for [`PlayerId`] in the universe and match
/// the value.  To find which player owns an entity, read its `PlayerId` component.
#[derive(Component, Clone, Copy, Debug, Default, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct PlayerId(u32);

impl Display for PlayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add<u32> for PlayerId {
    type Output = Self;
    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl AddAssign<u32> for PlayerId {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

// PlayerHandle

/// The engine's representation of a connected player.
///
/// A `PlayerHandle` is intentionally a pure data container: it has no entity,
/// no world reference, and no knowledge of ECS state. The link between a player
/// and their in-world entity is established by the game in response to
/// [`PlayerSpawned`].
///
/// Access handles via [`PlayerRegistry::get_player`] or
/// [`PlayerRegistry::get_player_mut`].
pub struct PlayerHandle {
    id: PlayerId,
    input: InputContext,
}

impl PlayerHandle {
    /// Returns this player's [`PlayerId`].
    #[must_use]
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// Returns the current hard-coded movement input for this player.
    ///
    /// The vector is in local camera movement space: `x` is right/left, `y` is
    /// up/down, and `z` is forward/back.
    #[must_use]
    pub fn movement_input(&self) -> glam::Vec3 {
        self.input.movement_input()
    }
}

// PlayerManager

/// Manages the full lifecycle of all players.
///
/// `PlayerManager` is the single authority on which players exist.  It does not
/// interact with the ECS directly — all coupling goes through events.
///
/// # Lifecycle
///
/// ```text
/// new_player()      ──► PlayerSpawned      — game code spawns ECS entity
/// remove_player(id) ──► PlayerDespawned    — game code despawns ECS entity
/// ```
///
/// The public [`PlayerRegistry`] exposes player creation and lookup, while this
/// internal subsystem consumes input events and updates per-frame input state
/// for player movement systems.
struct PlayerManager {
    registry: PlayerRegistry,
    player_input_consumer: dirk_events::Consumer<PlayerInput>,
    input_sender: PlayerInputSender,
    #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
    presentation_assignments: PlayerPresentationAssignments,
    #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
    window_input_consumer: dirk_events::Consumer<WindowInputEvent>,
}

/// Shared player registry owned by the player subsystem.
#[derive(Clone)]
pub struct PlayerRegistry {
    state: Arc<RwLock<PlayerState>>,
    input_state: PlayerInputState,
    spawned_dispatcher: Dispatcher<PlayerSpawned>,
    despawned_dispatcher: Dispatcher<PlayerDespawned>,
}

#[derive(Default)]
struct PlayerState {
    // TODO: setup generation based player allocation
    next_id: PlayerId,
    players: HashMap<PlayerId, PlayerHandle>,
}

impl PlayerManager {
    /// Creates a new [`PlayerManager`], registering its event channels with
    /// `events`.
    #[must_use]
    fn new(events: &EventManager) -> Self {
        let registry = PlayerRegistry {
            state: Arc::default(),
            input_state: PlayerInputState::default(),
            spawned_dispatcher: events.register(),
            despawned_dispatcher: events.register(),
        };

        Self {
            registry,
            player_input_consumer: events.subscribe(),
            input_sender: PlayerInputSender {
                dispatcher: events.register(),
            },
            #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
            presentation_assignments: PlayerPresentationAssignments::default(),
            #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
            window_input_consumer: events.subscribe(),
        }
    }

    /// Returns a shared registry for creating and removing players.
    #[must_use]
    fn registry(&self) -> PlayerRegistry {
        self.registry.clone()
    }

    /// Returns a shared player input sender.
    #[must_use]
    fn input_sender(&self) -> PlayerInputSender {
        self.input_sender.clone()
    }

    /// Returns shared presentation assignments.
    #[must_use]
    #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
    fn presentation_assignments(&self) -> PlayerPresentationAssignments {
        self.presentation_assignments.clone()
    }
}

impl Subsystem for PlayerManager {
    fn name(&self) -> &'static str {
        "player"
    }

    /// Ticks internal player state.
    fn tick(
        &mut self,
        _delta_time: f64,
        _handle: &EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        #[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
        for event in self.window_input_consumer.consume_all() {
            if let Some(player) = self
                .presentation_assignments
                .player_for_window(event.window)
            {
                self.input_sender.send(player, event.event);
            }
        }

        let player_events = self.player_input_consumer.consume_all().collect::<Vec<_>>();
        let mut state = self.registry.state.write();
        for event in player_events {
            if let Some(player) = state.players.get_mut(&event.id) {
                player.input.handle_event(&event.event);
            }
        }
        for player in state.players.values() {
            self.registry.input_state.set(
                player.id,
                PlayerInputFrame {
                    movement: player.movement_input(),
                    look: player.input.look_input(),
                },
            );
        }
        for player in state.players.values_mut() {
            player.input.clear_frame_state();
        }
        Ok(())
    }
}

impl PlayerRegistry {
    /// Returns a shared read handle for systems that consume player input.
    #[must_use]
    pub fn input_state(&self) -> PlayerInputState {
        self.input_state.clone()
    }

    /// Creates a new player and fires [`PlayerSpawned`].
    ///
    /// # Returns
    ///
    /// The [`PlayerId`] of the new player.
    #[must_use]
    pub fn new_player(&self) -> PlayerId {
        let id = {
            let mut state = self.state.write();
            let id = Self::allocate_id(&mut state);
            state.players.insert(
                id,
                PlayerHandle {
                    id,
                    input: InputContext::new(),
                },
            );
            id
        };

        self.input_state.set(
            id,
            PlayerInputFrame {
                movement: glam::Vec3::ZERO,
                look: glam::DVec2::ZERO,
            },
        );
        self.spawned_dispatcher.dispatch(PlayerSpawned { id });
        id
    }

    /// Removes the player with `id` and fires [`PlayerDespawned`].
    pub fn remove_player(&self, id: PlayerId) {
        if self.state.write().players.remove(&id).is_some() {
            self.input_state.remove_player(id);
            self.despawned_dispatcher.dispatch(PlayerDespawned { id });
        }
    }

    /// Returns a guard for the player with `id`, or `None`.
    ///
    /// # Panics
    ///
    /// Panics only if the player disappears while the same read lock is held,
    /// which would indicate internal registry corruption.
    #[must_use]
    pub fn get_player(&self, id: PlayerId) -> Option<MappedRwLockReadGuard<'_, PlayerHandle>> {
        let state = self.state.read();
        if !state.players.contains_key(&id) {
            return None;
        }

        Some(RwLockReadGuard::map(state, |state| {
            state
                .players
                .get(&id)
                .expect("player was checked before mapping guard")
        }))
    }

    /// Returns a mutable guard for the player with `id`, or `None`.
    ///
    /// # Panics
    ///
    /// Panics only if the player disappears while the same write lock is held,
    /// which would indicate internal registry corruption.
    #[must_use]
    pub fn get_player_mut(&self, id: PlayerId) -> Option<MappedRwLockWriteGuard<'_, PlayerHandle>> {
        let state = self.state.write();
        if !state.players.contains_key(&id) {
            return None;
        }

        Some(parking_lot::RwLockWriteGuard::map(state, |state| {
            state
                .players
                .get_mut(&id)
                .expect("player was checked before mapping guard")
        }))
    }

    /// Returns the IDs of all live players.
    #[must_use]
    pub fn players(&self) -> Vec<PlayerId> {
        self.state.read().players.keys().copied().collect()
    }

    fn allocate_id(state: &mut PlayerState) -> PlayerId {
        let id = state.next_id;
        state.next_id += 1;
        id
    }
}

/// Shared per-player input values consumed by universe systems.
#[derive(Clone, Default)]
pub struct PlayerInputState {
    frames: Arc<RwLock<HashMap<PlayerId, PlayerInputFrame>>>,
}

/// Cloneable sender for targeted player input.
#[derive(Clone)]
pub struct PlayerInputSender {
    dispatcher: Dispatcher<PlayerInput>,
}

impl PlayerInputSender {
    /// Sends input to one player.
    pub fn send(&self, id: PlayerId, event: dirk_input::InputEvent) {
        self.dispatcher.dispatch(PlayerInput { id, event });
    }
}

/// Window-to-player presentation assignments used by non-editor input routing.
#[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
#[derive(Clone, Default)]
pub struct PlayerPresentationAssignments {
    assignments: Arc<RwLock<HashMap<WindowId, PlayerId>>>,
}

#[cfg(any(not(feature = "editor"), target_vendor = "apple"))]
impl PlayerPresentationAssignments {
    /// Replaces all assignments.
    pub fn set(&self, assignments: Vec<(WindowId, PlayerId)>) {
        *self.assignments.write() = assignments.into_iter().collect();
    }

    /// Returns the player assigned to a window.
    #[must_use]
    pub fn player_for_window(&self, window: WindowId) -> Option<PlayerId> {
        self.assignments.read().get(&window).copied()
    }
}

/// Per-frame input values for one player.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerInputFrame {
    /// Movement intent in local player space.
    pub movement: glam::Vec3,
    /// Pointer-look delta in normalized viewport units.
    pub look: glam::DVec2,
}

impl PlayerInputState {
    pub(crate) fn get(&self, player: PlayerId) -> PlayerInputFrame {
        self.frames.read().get(&player).copied().unwrap_or_default()
    }

    fn set(&self, player: PlayerId, frame: PlayerInputFrame) {
        self.frames.write().insert(player, frame);
    }

    fn remove_player(&self, player: PlayerId) {
        self.frames.write().remove(&player);
    }
}
