//! Records deliberate edits so the editor can undo and redo them.
//!
//! Enable the `journal` feature and call `UniverseBuilder::with_journal`.
//! Each command buffer marked [`undoable`](crate::CommandBuffer::undoable)
//! becomes one [`Entry`], holding each component it changed with its value
//! before and after. Structure is data, so entries also cover spawns, moves,
//! despawns and world changes. Changes made by systems and by other buffers
//! are not recorded, so a running simulation neither floods the history nor
//! discards what can be redone. Submit `CommandBuffer::undo` or
//! `CommandBuffer::redo` to step through the history; restored values count as
//! ordinary changes, so systems observe them through `Delta` and `Changed`.
//!
//! The journal keeps a copy of every component, which is why it is meant for
//! the editor rather than shipped games.

use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
};

use crate::{Entity, Universe, components::AnyComponent};

/// The recorded history of a universe.
pub struct Journal {
    /// Component values as of the last record.
    mirror: HashMap<(Entity, TypeId), Box<dyn AnyComponent>>,
    /// The change tick up to which changes are recorded.
    recorded: u64,
    history: VecDeque<Entry>,
    undone: Vec<Entry>,
    capacity: usize,
}

/// The component changes of one tick, or of one undo or redo.
pub struct Entry {
    changes: Vec<Change>,
}

/// One component's change within an [`Entry`].
pub struct Change {
    entity: Entity,
    type_id: TypeId,
    component: &'static str,
    before: Option<Box<dyn AnyComponent>>,
    after: Option<Box<dyn AnyComponent>>,
}

/// How a component changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeKind {
    /// The component was added.
    Added,
    /// The component's value changed.
    Changed,
    /// The component was removed.
    Removed,
}

impl Journal {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            mirror: HashMap::new(),
            recorded: 0,
            history: VecDeque::new(),
            undone: Vec::new(),
            capacity,
        }
    }

    /// Returns the recorded entries, oldest first. Undoing reverts the last.
    #[must_use]
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = &Entry> + ExactSizeIterator {
        self.history.iter()
    }

    /// Returns whether an entry can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    /// Returns whether an undone entry can be redone. Recording a new entry
    /// discards undone ones.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.undone.is_empty()
    }

    /// Records the changes made since the last update as one entry.
    pub(crate) fn record(&mut self, universe: &Universe) {
        let changes = self.update(universe);
        if changes.is_empty() {
            return;
        }
        self.undone.clear();
        self.history.push_back(Entry { changes });
        while self.history.len() > self.capacity {
            self.history.pop_front();
        }
    }

    /// Takes in the changes made since the last update without recording
    /// them, so later entries start from current values.
    pub(crate) fn sync(&mut self, universe: &Universe) {
        self.update(universe);
    }

    /// Brings the copy of every component up to date, returning the changes.
    fn update(&mut self, universe: &Universe) -> Vec<Change> {
        let since = self.recorded;
        let mut changes = Vec::new();
        for (entity, type_id, value) in universe.components.changed_since_any(since) {
            let before = self.mirror.insert((entity, type_id), value.clone_box());
            changes.push(Change {
                entity,
                type_id,
                component: value.component_type_name(),
                before,
                after: Some(value.clone_box()),
            });
        }
        for (entity, type_id) in universe.removed_since_any(since) {
            if universe.components.contains(entity, type_id) {
                continue; // Re-added, so recorded as changed above.
            }
            if let Some(before) = self.mirror.remove(&(entity, type_id)) {
                changes.push(Change {
                    entity,
                    type_id,
                    component: before.component_type_name(),
                    before: Some(before),
                    after: None,
                });
            }
        }
        self.recorded = universe.change_tick.get();
        changes
    }

    /// Undoes the last entry, or redoes the last undone one, returning the
    /// values to restore. `None` removes a component.
    pub(crate) fn step(&mut self, undo: bool, tick: u64) -> Vec<Restore> {
        let entry = if undo {
            self.history.pop_back()
        } else {
            self.undone.pop()
        };
        let Some(entry) = entry else {
            return Vec::new();
        };
        let restores = entry
            .changes
            .iter()
            .map(|change| {
                let value = if undo { &change.before } else { &change.after };
                let value = value.as_ref().map(|value| value.clone_box());
                match &value {
                    Some(value) => {
                        self.mirror
                            .insert((change.entity, change.type_id), value.clone_box());
                    }
                    None => {
                        self.mirror.remove(&(change.entity, change.type_id));
                    }
                }
                Restore {
                    entity: change.entity,
                    type_id: change.type_id,
                    value,
                }
            })
            .collect();
        if undo {
            self.undone.push(entry);
        } else {
            self.history.push_back(entry);
        }
        self.recorded = tick;
        restores
    }
}

/// A component value to put back: `None` removes the component.
pub(crate) struct Restore {
    pub(crate) entity: Entity,
    pub(crate) type_id: TypeId,
    pub(crate) value: Option<Box<dyn AnyComponent>>,
}

impl Entry {
    /// Returns the changes in this entry.
    pub fn changes(&self) -> impl Iterator<Item = &Change> {
        self.changes.iter()
    }

    /// Returns the number of changed components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Returns whether the entry changed nothing. Recorded entries never are.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

impl Change {
    /// Returns the changed entity.
    #[must_use]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Returns the component's type name.
    #[must_use]
    pub fn component(&self) -> &'static str {
        self.component
    }

    /// Returns how the component changed.
    #[must_use]
    pub fn kind(&self) -> ChangeKind {
        match (&self.before, &self.after) {
            (None, _) => ChangeKind::Added,
            (Some(_), Some(_)) => ChangeKind::Changed,
            (Some(_), None) => ChangeKind::Removed,
        }
    }
}
