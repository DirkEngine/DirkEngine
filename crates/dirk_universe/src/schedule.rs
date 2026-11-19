//! Orders systems from the data they declare.
//!
//! A system reading `C` (through `&C`, `Delta<C>` or a filter) runs after
//! every system writing `C`, so it sees this tick's values. `Lagged<D>` flips
//! that: the reader runs before the writers, seeing values from before this
//! tick's systems ran.
//! Systems that write the same component must be ordered explicitly with
//! [`IntoSystem::before`](crate::systems::IntoSystem::before) or
//! [`IntoSystem::after`](crate::systems::IntoSystem::after). Unrelated systems
//! keep their registration order.

use std::{
    any::{TypeId, type_name},
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    fmt::{self, Display},
};

use crate::{components::Component, systems::ErasedSystem};

/// A problem with the systems registered on a universe, found by
/// `UniverseBuilder::build`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ScheduleError {
    /// One system borrows a component in incompatible ways, such as through
    /// `&C` and `&mut C`, or requests `Commands` twice.
    #[error("system `{system}` accesses `{component}` through conflicting parameters")]
    ConflictingAccess {
        /// The system.
        system: &'static str,
        /// The conflicting component, or `Commands`.
        component: &'static str,
    },
    /// Two systems write the same component without an order between them.
    #[error(
        "systems `{first}` and `{second}` both write `{component}` in no defined order; \
         order them with `.before()` or `.after()`"
    )]
    AmbiguousWriters {
        /// The component written by both.
        component: &'static str,
        /// The system registered first.
        first: &'static str,
        /// The system registered second.
        second: &'static str,
    },
    /// Systems depend on each other in a loop.
    #[error("systems depend on each other in a cycle: {}", Cycle(.0))]
    Cycle(Vec<Dependency>),
    /// A system is ordered against a system that was never registered.
    #[error("system `{system}` is ordered against `{label}`, which is not registered")]
    UnknownLabel {
        /// The system declaring the order.
        system: &'static str,
        /// The missing system.
        label: &'static str,
    },
}

/// Why one system runs after another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The earlier system writes a component the later one reads.
    Writes(&'static str),
    /// The earlier system reads last tick's value of a component the later
    /// one writes, through `Lagged`.
    ReadsLagged(&'static str),
    /// The order was declared with `.before()` or `.after()`.
    Explicit,
}

impl Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Writes(component) => write!(f, "which writes `{component}`"),
            Self::ReadsLagged(component) => write!(f, "which reads last tick's `{component}`"),
            Self::Explicit => write!(f, "as ordered explicitly"),
        }
    }
}

/// A system that must run earlier, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dependency {
    /// The earlier system.
    pub system: &'static str,
    /// Why it runs earlier.
    pub reason: Reason,
}

impl Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`, {}", self.system, self.reason)
    }
}

/// Displays a cycle as the chain of dependencies that closes it.
struct Cycle<'a>(&'a [Dependency]);

impl Display for Cycle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, dependency) in self.0.iter().enumerate() {
            let next = self.0[(index + 1) % self.0.len()].system;
            if index > 0 {
                write!(f, "; ")?;
            }
            write!(f, "`{next}` runs after {dependency}")?;
        }
        Ok(())
    }
}

/// The order in which a universe runs its systems, with the reasons for it.
/// Its `Display` output lists one system per line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    systems: Vec<ScheduledSystem>,
}

/// One system in a [`Schedule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledSystem {
    /// The system's name.
    pub name: &'static str,
    /// The systems it runs after, and why.
    pub after: Vec<Dependency>,
}

impl Schedule {
    /// Returns the systems in execution order.
    #[must_use]
    pub fn systems(&self) -> &[ScheduledSystem] {
        &self.systems
    }
}

impl Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, system) in self.systems.iter().enumerate() {
            write!(f, "{}. {}", index + 1, system.name)?;
            for dependency in &system.after {
                write!(f, "\n   after {dependency}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Identifies a registered system by its type: a function item, a closure or
/// a `System` implementation.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct SystemId {
    id: TypeId,
    name: &'static str,
}

impl SystemId {
    pub(crate) fn of<T: 'static>() -> Self {
        Self {
            id: TypeId::of::<T>(),
            name: type_name::<T>(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Borrow {
    Shared,
    Exclusive,
}

/// The data one system declares. Borrows are checked for conflicts within
/// the system; reads and writes order it against other systems.
#[doc(hidden)]
#[derive(Default)]
pub struct Access {
    borrows: HashMap<TypeId, Borrow>,
    reads: HashSet<TypeId>,
    lagged: HashSet<TypeId>,
    writes: HashSet<TypeId>,
    names: HashMap<TypeId, &'static str>,
    commands: bool,
    lagging: bool,
    conflict: Option<&'static str>,
}

impl Access {
    /// Registers a shared borrow of `C`.
    pub fn read<C: Component>(&mut self) {
        if self.borrows.get(&TypeId::of::<C>()) == Some(&Borrow::Exclusive) {
            self.conflict::<C>();
        }
        self.borrows
            .entry(TypeId::of::<C>())
            .or_insert(Borrow::Shared);
        self.observe::<C>();
    }

    /// Registers an exclusive borrow of `C`.
    pub fn write<C: Component>(&mut self) {
        if self
            .borrows
            .insert(TypeId::of::<C>(), Borrow::Exclusive)
            .is_some()
        {
            self.conflict::<C>();
        }
        self.name::<C>();
        self.writes.insert(TypeId::of::<C>());
    }

    /// Registers a dependency on `C` without borrowing it, as filters do.
    pub fn observe<C: Component>(&mut self) {
        self.name::<C>();
        if self.lagging {
            self.lagged.insert(TypeId::of::<C>());
        } else {
            self.reads.insert(TypeId::of::<C>());
        }
    }

    /// Registers structural command access.
    pub fn commands(&mut self) {
        if std::mem::replace(&mut self.commands, true) {
            self.conflict = self.conflict.or(Some("Commands"));
        }
    }

    /// Registers the reads made by `register` as reads of last tick's values.
    pub fn lagged(&mut self, register: impl FnOnce(&mut Self)) {
        let lagging = std::mem::replace(&mut self.lagging, true);
        register(self);
        self.lagging = lagging;
    }

    fn name<C: Component>(&mut self) {
        self.names.insert(TypeId::of::<C>(), type_name::<C>());
    }

    fn conflict<C: Component>(&mut self) {
        self.conflict = self.conflict.or(Some(type_name::<C>()));
    }
}

/// A system ready to be scheduled, with its declared order.
#[doc(hidden)]
pub struct SystemConfig {
    pub(crate) system: Box<dyn ErasedSystem>,
    pub(crate) id: SystemId,
    pub(crate) access: Access,
    pub(crate) before: Vec<SystemId>,
    pub(crate) after: Vec<SystemId>,
}

/// Orders `configs`, returning the systems in execution order.
pub(crate) fn build(
    configs: Vec<SystemConfig>,
) -> Result<(Vec<Box<dyn ErasedSystem>>, Schedule), ScheduleError> {
    let graph = Graph::new(&configs)?;
    let order = graph.order()?;
    graph.check_writers()?;

    let schedule = Schedule {
        systems: order
            .iter()
            .map(|&index| ScheduledSystem {
                name: configs[index].id.name,
                after: graph
                    .edges
                    .iter()
                    .filter(|((_, to), _)| *to == index)
                    .map(|(&(from, _), &reason)| Dependency {
                        system: configs[from].id.name,
                        reason,
                    })
                    .collect(),
            })
            .collect(),
    };
    let mut systems: Vec<_> = configs.into_iter().map(Some).collect();
    let systems = order
        .into_iter()
        .filter_map(|index| systems[index].take())
        .map(|config| config.system)
        .collect();
    Ok((systems, schedule))
}

/// Dependencies between systems, identified by registration index.
struct Graph<'a> {
    configs: &'a [SystemConfig],
    /// Edges from an earlier to a later system, in a deterministic order.
    edges: BTreeMap<(usize, usize), Reason>,
}

impl<'a> Graph<'a> {
    fn new(configs: &'a [SystemConfig]) -> Result<Self, ScheduleError> {
        let mut graph = Self {
            configs,
            edges: BTreeMap::new(),
        };
        for config in configs {
            if let Some(component) = config.access.conflict {
                return Err(ScheduleError::ConflictingAccess {
                    system: config.id.name,
                    component,
                });
            }
        }

        // Explicit orders replace any derived order between the same pair.
        for (index, config) in configs.iter().enumerate() {
            for &label in &config.after {
                for other in graph.labelled(config, label)? {
                    graph.edges.insert((other, index), Reason::Explicit);
                }
            }
            for &label in &config.before {
                for other in graph.labelled(config, label)? {
                    graph.edges.insert((index, other), Reason::Explicit);
                }
            }
        }
        let explicit: HashSet<_> = graph
            .edges
            .keys()
            .flat_map(|&(from, to)| [(from, to), (to, from)])
            .collect();

        for (writer, writing) in configs.iter().enumerate() {
            for (reader, reading) in configs.iter().enumerate() {
                if writer == reader || explicit.contains(&(writer, reader)) {
                    continue;
                }
                for component in &writing.access.writes {
                    let name = writing.access.names[component];
                    if reading.access.reads.contains(component) {
                        graph
                            .edges
                            .entry((writer, reader))
                            .or_insert(Reason::Writes(name));
                    }
                    if reading.access.lagged.contains(component) {
                        graph
                            .edges
                            .entry((reader, writer))
                            .or_insert(Reason::ReadsLagged(name));
                    }
                }
            }
        }
        Ok(graph)
    }

    /// Returns the systems registered under `label`.
    fn labelled(
        &self,
        config: &SystemConfig,
        label: SystemId,
    ) -> Result<Vec<usize>, ScheduleError> {
        let matches: Vec<_> = (0..self.configs.len())
            .filter(|&index| self.configs[index].id.id == label.id)
            .collect();
        if matches.is_empty() {
            return Err(ScheduleError::UnknownLabel {
                system: config.id.name,
                label: label.name,
            });
        }
        Ok(matches)
    }

    fn successors(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        self.edges
            .keys()
            .filter(move |(from, _)| *from == index)
            .map(|&(_, to)| to)
    }

    /// Sorts the systems topologically, preferring registration order.
    fn order(&self) -> Result<Vec<usize>, ScheduleError> {
        let mut incoming = vec![0_usize; self.configs.len()];
        for &(_, to) in self.edges.keys() {
            incoming[to] += 1;
        }
        let mut ready: BinaryHeap<_> = (0..self.configs.len())
            .filter(|&index| incoming[index] == 0)
            .map(Reverse)
            .collect();
        let mut order = Vec::with_capacity(self.configs.len());
        while let Some(Reverse(index)) = ready.pop() {
            order.push(index);
            for next in self.successors(index) {
                incoming[next] -= 1;
                if incoming[next] == 0 {
                    ready.push(Reverse(next));
                }
            }
        }
        if order.len() == self.configs.len() {
            Ok(order)
        } else {
            Err(ScheduleError::Cycle(self.cycle(&incoming)))
        }
    }

    /// Finds a cycle among the systems left unsorted, which all still have
    /// an unsorted predecessor.
    fn cycle(&self, incoming: &[usize]) -> Vec<Dependency> {
        let mut path = vec![
            (0..incoming.len())
                .find(|&index| incoming[index] > 0)
                .expect("an unsorted system remains"),
        ];
        loop {
            let current = *path.last().expect("path is never empty");
            let previous = self
                .edges
                .keys()
                .find(|&&(from, to)| to == current && incoming[from] > 0)
                .map(|&(from, _)| from)
                .expect("unsorted systems have an unsorted predecessor");
            if let Some(start) = path.iter().position(|&index| index == previous) {
                // `path` walks backwards: each system runs after the next one.
                // Start from the earliest registered system for stable output.
                let mut cycle: Vec<_> = path[start..].to_vec();
                cycle.reverse();
                let first = (0..cycle.len())
                    .min_by_key(|&position| cycle[position])
                    .expect("a cycle has systems");
                cycle.rotate_left(first);
                return cycle
                    .iter()
                    .enumerate()
                    .map(|(position, &from)| {
                        let to = cycle[(position + 1) % cycle.len()];
                        Dependency {
                            system: self.configs[from].id.name,
                            reason: self.edges[&(from, to)],
                        }
                    })
                    .collect();
            }
            path.push(previous);
        }
    }

    /// Rejects systems writing the same component in no defined order.
    fn check_writers(&self) -> Result<(), ScheduleError> {
        let reachable: Vec<HashSet<usize>> = (0..self.configs.len())
            .map(|index| self.reachable(index))
            .collect();
        for (first, a) in self.configs.iter().enumerate() {
            for (second, b) in self.configs.iter().enumerate().skip(first + 1) {
                let ordered =
                    reachable[first].contains(&second) || reachable[second].contains(&first);
                if ordered {
                    continue;
                }
                let shared = a
                    .access
                    .writes
                    .intersection(&b.access.writes)
                    .map(|component| a.access.names[component])
                    .min();
                if let Some(component) = shared {
                    return Err(ScheduleError::AmbiguousWriters {
                        component,
                        first: a.id.name,
                        second: b.id.name,
                    });
                }
            }
        }
        Ok(())
    }

    fn reachable(&self, start: usize) -> HashSet<usize> {
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        while let Some(index) = stack.pop() {
            for next in self.successors(index) {
                if seen.insert(next) {
                    stack.push(next);
                }
            }
        }
        seen
    }
}
