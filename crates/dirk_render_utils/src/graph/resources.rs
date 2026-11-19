//! The images and buffers of a graph: imported or created, their tracked
//! states, and the pool transients are reused from.

use std::{collections::HashMap, fmt::Debug, marker::PhantomData, ops::BitOrAssign};

use dirk_rhi::{
    Buffer, BufferAccess, BufferDesc, BufferInfo, BufferUsage, Error, FRAMES_IN_FLIGHT, Frame,
    Image, ImageAccess, ImageDesc, ImageInfo, ImageUsage, Result, Rhi,
};

use super::access::{Access, Barrier, States};

/// A kind of resource the graph tracks: [`Image`] or [`Buffer`].
pub(crate) trait Resource: Sized {
    /// How passes access it.
    type Access: Access;
    /// Ways it may be used.
    type Usage: Copy + Default + Eq + Debug + BitOrAssign;
    /// Describes one to create.
    type Desc<'d>: Copy;
    /// Name of the kind, for errors.
    const KIND: &'static str;

    /// The usage `access` needs.
    fn usage_of(access: Self::Access) -> Self::Usage;

    /// Whether it was created with `usage`.
    fn supports(&self, usage: Self::Usage) -> bool;

    /// Creates one from `desc`, with `usage` added to the description's.
    fn create(rhi: &Rhi, desc: &Self::Desc<'_>, usage: Self::Usage) -> Result<Self>;

    /// Whether it is what [`Resource::create`] makes of `desc` and `usage`.
    fn matches(&self, desc: &Self::Desc<'_>, usage: Self::Usage) -> bool;
}

impl Resource for Image {
    type Access = ImageAccess;
    type Usage = ImageUsage;
    type Desc<'d> = ImageDesc<'d>;
    const KIND: &'static str = "image";

    fn usage_of(access: ImageAccess) -> ImageUsage {
        access.usage()
    }

    fn supports(&self, usage: ImageUsage) -> bool {
        self.info().usage.contains(usage)
    }

    fn create(rhi: &Rhi, desc: &ImageDesc<'_>, usage: ImageUsage) -> Result<Self> {
        rhi.create_image(&ImageDesc {
            usage: desc.usage | usage,
            ..*desc
        })
    }

    fn matches(&self, desc: &ImageDesc<'_>, usage: ImageUsage) -> bool {
        self.label() == desc.label
            && *self.info()
                == ImageInfo {
                    dimension: desc.dimension,
                    extent: desc.extent,
                    format: desc.format,
                    usage: desc.usage | usage,
                    mip_levels: desc.mip_levels,
                    array_layers: desc.array_layers,
                    samples: desc.samples,
                }
    }
}

impl Resource for Buffer {
    type Access = BufferAccess;
    type Usage = BufferUsage;
    type Desc<'d> = BufferDesc<'d>;
    const KIND: &'static str = "buffer";

    fn usage_of(access: BufferAccess) -> BufferUsage {
        access.usage()
    }

    fn supports(&self, usage: BufferUsage) -> bool {
        self.info().usage.contains(usage)
    }

    fn create(rhi: &Rhi, desc: &BufferDesc<'_>, usage: BufferUsage) -> Result<Self> {
        rhi.create_buffer(&BufferDesc {
            usage: desc.usage | usage,
            ..*desc
        })
    }

    fn matches(&self, desc: &BufferDesc<'_>, usage: BufferUsage) -> bool {
        self.label() == desc.label
            && *self.info()
                == BufferInfo {
                    size: desc.size,
                    usage: desc.usage | usage,
                    memory: desc.memory,
                }
    }
}

/// An [`Image`] or [`Buffer`] of a [`RenderGraph`](super::RenderGraph),
/// imported into it or created by it. Only valid with the graph that
/// returned it; passes of other graphs reject it.
pub struct Handle<R> {
    /// Identifies the graph.
    graph: u64,
    pub(crate) index: usize,
    resource: PhantomData<fn() -> R>,
}

impl<R> Clone for Handle<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for Handle<R> {}

impl<R> PartialEq for Handle<R> {
    fn eq(&self, other: &Self) -> bool {
        (self.graph, self.index) == (other.graph, other.index)
    }
}

impl<R> Eq for Handle<R> {}

impl<R> Debug for Handle<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Handle({} of graph {})", self.index, self.graph)
    }
}

/// An access a pass declared.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Use<A> {
    /// Index of the resource in the graph.
    pub(crate) resource: usize,
    pub(crate) access: A,
    /// Whether the pass reads the contents, such as an attachment it loads.
    pub(crate) reads: bool,
}

/// Every resource of one kind in a graph.
pub(crate) struct Resources<'a, R: Resource> {
    /// Identifies the graph, in handles.
    graph: u64,
    pub(crate) list: Vec<Tracked<'a, R>>,
    /// Index and boundary accesses of each import, by its address.
    imports: HashMap<usize, (usize, R::Access, R::Access)>,
    /// Sorted by pass. Earlier barriers widen as later passes read.
    pub(crate) barriers: Vec<Barrier<R::Access>>,
}

/// A resource and what the graph knows of it.
pub(crate) struct Tracked<'a, R: Resource> {
    label: &'a str,
    pub(crate) source: Source<'a, R>,
    /// What the declared accesses need.
    pub(crate) usage: R::Usage,
    states: States<R::Access>,
    mips: u32,
    /// The last pass declaring it.
    pub(crate) last_pass: Option<usize>,
}

/// Where a graph's resource comes from.
pub(crate) enum Source<'a, R: Resource> {
    /// Owned elsewhere, and left in `after` once the graph ends. `scratch`
    /// imports hold nothing needed after the graph.
    Imported {
        object: &'a R,
        after: R::Access,
        scratch: bool,
    },
    /// Taken from the transient pool, or created, when the graph executes.
    Transient(R::Desc<'a>),
}

impl<'a, R: Resource> Resources<'a, R> {
    /// No resources, in graph `graph`.
    pub(crate) fn new(graph: u64) -> Self {
        Self {
            graph,
            list: Vec::new(),
            imports: HashMap::new(),
            barriers: Vec::new(),
        }
    }

    /// Imports `object`, of `mips` levels, in `before` and leaves it in
    /// `after` once the graph ends. Importing it again returns the same
    /// handle.
    pub(crate) fn import(
        &mut self,
        object: &'a R,
        label: &'a str,
        mips: u32,
        before: R::Access,
        after: R::Access,
    ) -> Result<Handle<R>> {
        let invalid = |reason| Error::Invalid {
            object: "render graph import",
            label: label.to_owned(),
            reason,
        };
        let address = std::ptr::from_ref(object).addr();
        if let Some(&(index, first_before, first_after)) = self.imports.get(&address) {
            return if (first_before, first_after) == (before, after) {
                Ok(self.handle(index))
            } else {
                Err(invalid(format!(
                    "is already imported from {first_before:?} to {first_after:?}, not \
                     {before:?} to {after:?}"
                )))
            };
        }
        if after == R::Access::DISCARDED {
            return Err(invalid(format!("cannot be left in {after:?}")));
        }
        let tracked = Tracked {
            label,
            source: Source::Imported {
                object,
                after,
                scratch: false,
            },
            usage: R::Usage::default(),
            states: States::new(mips, before),
            mips,
            last_pass: None,
        };
        tracked
            .check(before)
            .and_then(|()| tracked.check(after))
            .map_err(invalid)?;
        let index = self.list.len();
        self.list.push(tracked);
        self.imports.insert(address, (index, before, after));
        Ok(self.handle(index))
    }

    /// Adds a transient described by `desc`, of `mips` levels.
    pub(crate) fn create(&mut self, desc: R::Desc<'a>, label: &'a str, mips: u32) -> Handle<R> {
        self.list.push(Tracked {
            label,
            source: Source::Transient(desc),
            usage: R::Usage::default(),
            states: States::new(mips, R::Access::DISCARDED),
            mips,
            last_pass: None,
        });
        self.handle(self.list.len() - 1)
    }

    /// The handle of the resource at `index`.
    fn handle(&self, index: usize) -> Handle<R> {
        Handle {
            graph: self.graph,
            index,
            resource: PhantomData,
        }
    }

    /// Adds a pass's `access` of `handle` to `uses`, or says why the pass
    /// cannot declare it.
    pub(crate) fn declare(
        &self,
        uses: &mut Vec<Use<R::Access>>,
        handle: Handle<R>,
        access: R::Access,
        reads: bool,
    ) -> Result<(), String> {
        if handle.graph != self.graph {
            return Err(format!("{} {handle:?} belongs to another graph", R::KIND));
        }
        let tracked = &self.list[handle.index];
        if !access.in_pass() {
            return Err(format!(
                "{} `{}`: passes do not declare {access:?}",
                R::KIND,
                tracked.label
            ));
        }
        tracked.check(access)?;
        uses.push(Use {
            resource: handle.index,
            access,
            reads,
        });
        Ok(())
    }

    /// Merges a pass's uses of the same resource, and checks each against
    /// the resource's state before the pass.
    pub(crate) fn prepare(&self, uses: Vec<Use<R::Access>>) -> Result<Vec<Use<R::Access>>, String> {
        let mut merged: Vec<Use<R::Access>> = Vec::with_capacity(uses.len());
        for usage in uses {
            let tracked = &self.list[usage.resource];
            match merged
                .iter_mut()
                .find(|other| other.resource == usage.resource)
            {
                Some(other) => {
                    other.access = other.access.merge(usage.access).ok_or_else(|| {
                        format!(
                            "{} `{}` is accessed both as {:?} and as {:?}",
                            R::KIND,
                            tracked.label,
                            other.access,
                            usage.access
                        )
                    })?;
                    other.reads |= usage.reads;
                }
                None => merged.push(usage),
            }
        }
        for usage in &merged {
            let tracked = &self.list[usage.resource];
            for previous in tracked.states.accesses() {
                if usage.reads && !previous.holds_data() {
                    return Err(format!(
                        "{} `{}` is read before anything writes it",
                        R::KIND,
                        tracked.label
                    ));
                }
                usage
                    .access
                    .check_follows(previous)
                    .map_err(|reason| format!("{} `{}` {reason}", R::KIND, tracked.label))?;
            }
        }
        Ok(merged)
    }

    /// Moves each resource to its use in pass `pass`, adding the barriers
    /// this needs.
    pub(crate) fn apply(&mut self, pass: usize, uses: &[Use<R::Access>]) {
        for usage in uses {
            let tracked = &mut self.list[usage.resource];
            tracked.usage |= R::usage_of(usage.access);
            tracked.last_pass = Some(pass);
            tracked.states.transition(
                usage.resource,
                pass,
                0..tracked.mips,
                usage.access,
                false,
                &mut self.barriers,
            );
        }
    }

    /// Moves each import to its boundary access after the last pass, `end`.
    pub(crate) fn finish(&mut self, end: usize) {
        for (index, tracked) in self.list.iter_mut().enumerate() {
            if let Source::Imported { after, .. } = tracked.source {
                let mips = 0..tracked.mips;
                tracked
                    .states
                    .transition(index, end, mips, after, true, &mut self.barriers);
            }
        }
    }

    /// The barriers recorded before pass `pass`.
    pub(crate) fn before(&self, pass: usize) -> &[Barrier<R::Access>] {
        let start = self.barriers.partition_point(|barrier| barrier.pass < pass);
        let end = self
            .barriers
            .partition_point(|barrier| barrier.pass <= pass);
        &self.barriers[start..end]
    }

    /// Takes each declared transient from `pool`, or creates it, for frame
    /// `frame`. Transients no pass declared get nothing.
    pub(crate) fn allocate(&self, frame: &Frame<'_>, pool: &mut Pool<R>) -> Result<Vec<Option<R>>> {
        pool.evict(frame.number());
        self.list
            .iter()
            .map(|tracked| match tracked.source {
                Source::Transient(desc) if tracked.usage != R::Usage::default() => pool
                    .take(frame.number(), |object| {
                        object.matches(&desc, tracked.usage)
                    })
                    .map_or_else(|| R::create(frame.rhi(), &desc, tracked.usage), Ok)
                    .map(Some),
                _ => Ok(None),
            })
            .collect()
    }

    /// Every resource: the imports, and the transients in `allocated`.
    pub(crate) fn objects<'t>(&'t self, allocated: &'t [Option<R>]) -> Table<'t, R> {
        Table {
            graph: self.graph,
            objects: self
                .list
                .iter()
                .zip(allocated)
                .map(|(tracked, allocated)| match tracked.source {
                    Source::Imported { object, .. } => Some(object),
                    Source::Transient(_) => allocated.as_ref(),
                })
                .collect(),
        }
    }
}

impl<R: Resource> Tracked<'_, R> {
    /// Checks that `access` names its shader stages and, for imports, that
    /// the resource has the usage it needs.
    fn check(&self, access: R::Access) -> Result<(), String> {
        if access
            .shader_stages()
            .is_some_and(|stages| stages.is_empty())
        {
            return Err(format!(
                "{} `{}`: {access:?} names no shader stage",
                R::KIND,
                self.label
            ));
        }
        let usage = R::usage_of(access);
        if let Source::Imported { object, .. } = self.source
            && !object.supports(usage)
        {
            return Err(format!(
                "{} `{}` lacks the {usage:?} usage {access:?} needs",
                R::KIND,
                self.label
            ));
        }
        Ok(())
    }
}

/// The objects of a graph's resources while it executes, by index.
pub(crate) struct Table<'t, R> {
    graph: u64,
    objects: Vec<Option<&'t R>>,
}

impl<'t, R> Table<'t, R> {
    /// The resource at `index`, which a pass declared or the graph imported.
    pub(crate) fn get(&self, index: usize) -> &'t R {
        self.objects[index].expect("declared transients are allocated when the graph executes")
    }

    /// Whether `handle` belongs to this table's graph.
    pub(crate) fn owns(&self, handle: Handle<R>) -> bool {
        handle.graph == self.graph
    }
}

/// Resources free for reuse, each with the number of the last frame that
/// used it.
pub(crate) struct Pool<T> {
    free: Vec<(u64, T)>,
}

impl<T> Pool<T> {
    /// Drops what an earlier frame could have reused but did not, which
    /// defers its free until the GPU is done.
    pub(crate) fn evict(&mut self, frame: u64) {
        self.free
            .retain(|(used, _)| used + FRAMES_IN_FLIGHT as u64 >= frame);
    }

    /// Takes a resource satisfying `matches` that frame `frame` can use: one
    /// whose last frame's work completed before `frame` began, so it can
    /// start from an undefined state without a barrier against that work.
    pub(crate) fn take(&mut self, frame: u64, matches: impl Fn(&T) -> bool) -> Option<T> {
        let index = self.free.iter().position(|(used, object)| {
            used + FRAMES_IN_FLIGHT as u64 <= frame && matches(object)
        })?;
        Some(self.free.swap_remove(index).1)
    }

    /// Returns `objects`, used by frame `frame`, to the pool.
    pub(crate) fn release(&mut self, frame: u64, objects: impl IntoIterator<Item = T>) {
        self.free
            .extend(objects.into_iter().map(|object| (frame, object)));
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self { free: Vec::new() }
    }
}

/// The transient images and buffers of earlier graphs, reused by later ones.
/// Keep one per [`Rhi`] and pass it to every
/// [`RenderGraph::execute`](super::RenderGraph::execute).
///
/// A transient used in frame F is reused from frame F + [`FRAMES_IN_FLIGHT`]
/// on, once beginning that frame waited for F's work, by a transient with
/// the same description and derived usage. One that frame did not reuse is
/// dropped, its native free deferred until the GPU is done with it. Steady
/// rendering thus keeps [`FRAMES_IN_FLIGHT`] sets of transients.
#[derive(Default)]
pub struct Transients {
    pub(crate) images: Pool<Image>,
    pub(crate) buffers: Pool<Buffer>,
}

impl Transients {
    /// An empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use dirk_rhi::FRAMES_IN_FLIGHT;

    use super::Pool;

    const IN_FLIGHT: u64 = FRAMES_IN_FLIGHT as u64;

    #[test]
    fn pooled_resources_are_reused_once_their_frame_completed() {
        let mut pool = Pool::default();
        pool.release(1, ["a", "b"]);
        // Frames still running frame 1's work cannot reuse its resources.
        assert_eq!(pool.take(IN_FLIGHT, |_| true), None);
        assert_eq!(pool.take(1 + IN_FLIGHT, |&name| name == "b"), Some("b"));
        assert_eq!(pool.take(1 + IN_FLIGHT, |&name| name == "b"), None);
    }

    #[test]
    fn resources_a_frame_did_not_reuse_are_evicted() {
        let mut pool = Pool::default();
        pool.release(1, ["a", "b"]);
        pool.evict(1 + IN_FLIGHT);
        assert_eq!(
            pool.take(1 + IN_FLIGHT, |&name| name == "a"),
            Some("a"),
            "the first frame that may reuse them keeps them"
        );
        pool.evict(2 + IN_FLIGHT);
        assert_eq!(pool.take(2 + IN_FLIGHT, |_| true), None);
    }
}
