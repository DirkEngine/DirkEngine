//! How the graph tracks each resource's accesses and derives the barriers
//! between them.

use std::{fmt::Debug, ops::Range};

use dirk_rhi::{BufferAccess, ImageAccess, ShaderStages};

/// An access of a resource, as the graph tracks it: [`ImageAccess`] or
/// [`BufferAccess`].
pub(crate) trait Access: Copy + Eq + Debug {
    /// The access before a resource's first use, with undefined contents.
    const DISCARDED: Self;

    /// Whether the access writes the resource.
    fn writes(self) -> bool;

    /// One access covering both reads, if they can share a state, such as
    /// sampling by more stages.
    fn merge(self, other: Self) -> Option<Self>;

    /// Whether the resource's contents are defined in this access.
    fn holds_data(self) -> bool;

    /// Whether a barrier from this access orders or transitions anything.
    fn orders(self) -> bool;

    /// The stages of a shader access, which must name at least one.
    fn shader_stages(self) -> Option<ShaderStages>;

    /// Whether passes declare this access. Boundary-only accesses are not.
    fn in_pass(self) -> bool;

    /// Why this access cannot be the first after `previous`, if it cannot.
    fn check_follows(self, previous: Self) -> Result<(), &'static str>;
}

impl Access for ImageAccess {
    const DISCARDED: Self = Self::Undefined;

    fn writes(self) -> bool {
        ImageAccess::writes(self)
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Sampled(a), Self::Sampled(b)) => Some(Self::Sampled(a | b)),
            (Self::StorageRead(a), Self::StorageRead(b)) => Some(Self::StorageRead(a | b)),
            (Self::DepthStencilRead(a), Self::DepthStencilRead(b)) => {
                Some(Self::DepthStencilRead(a | b))
            }
            (Self::CopySource, Self::CopySource) => Some(self),
            _ => None,
        }
    }

    fn holds_data(self) -> bool {
        !matches!(self, Self::Undefined | Self::Present)
    }

    fn orders(self) -> bool {
        // Every image barrier may change the layout.
        true
    }

    fn shader_stages(self) -> Option<ShaderStages> {
        match self {
            Self::Sampled(stages) | Self::StorageRead(stages) | Self::StorageWrite(stages) => {
                Some(stages)
            }
            _ => None,
        }
    }

    fn in_pass(self) -> bool {
        !matches!(
            self,
            Self::Undefined | Self::Present | Self::DepthStencilRead(_)
        )
    }

    fn check_follows(self, previous: Self) -> Result<(), &'static str> {
        if previous == Self::Present
            && !matches!(self, Self::ColorAttachment | Self::CopyDestination)
        {
            return Err(
                "was just acquired, so its first access must write it as a color \
                        attachment or copy destination",
            );
        }
        Ok(())
    }
}

impl Access for BufferAccess {
    const DISCARDED: Self = Self::None;

    fn writes(self) -> bool {
        BufferAccess::writes(self)
    }

    fn merge(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::ShaderRead(a), Self::ShaderRead(b)) => Some(Self::ShaderRead(a | b)),
            (Self::CopySource, Self::CopySource)
            | (Self::Index, Self::Index)
            | (Self::Indirect, Self::Indirect) => Some(self),
            _ => None,
        }
    }

    fn holds_data(self) -> bool {
        self != Self::None
    }

    fn orders(self) -> bool {
        // Buffers have no layout: nothing precedes their first access.
        self != Self::None
    }

    fn shader_stages(self) -> Option<ShaderStages> {
        match self {
            Self::ShaderRead(stages) | Self::ShaderWrite(stages) => Some(stages),
            _ => None,
        }
    }

    fn in_pass(self) -> bool {
        !matches!(self, Self::None | Self::HostRead)
    }

    fn check_follows(self, _previous: Self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Mip levels and array layers of a resource. Buffers have one of each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Subresources {
    pub(crate) mips: Range<u32>,
    pub(crate) layers: Range<u32>,
}

impl Subresources {
    /// Every one of `mips` levels and `layers` layers.
    pub(crate) const fn all(mips: u32, layers: u32) -> Self {
        Self {
            mips: 0..mips,
            layers: 0..layers,
        }
    }

    /// Whether these share a subresource with `other`.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        let overlap = |a: &Range<u32>, b: &Range<u32>| a.start < b.end && b.start < a.end;
        overlap(&self.mips, &other.mips) && overlap(&self.layers, &other.layers)
    }

    /// Whether every subresource of `other` is one of these.
    fn contains(&self, other: &Self) -> bool {
        let contains = |a: &Range<u32>, b: &Range<u32>| a.start <= b.start && b.end <= a.end;
        contains(&self.mips, &other.mips) && contains(&self.layers, &other.layers)
    }

    /// Splits these where a bound of `other` cuts through them, keeping the
    /// first part and returning the rest, if one does.
    fn cut(&mut self, other: &Self) -> Option<Self> {
        let inside = |range: &Range<u32>, at: u32| range.start < at && at < range.end;
        if let Some(at) = [other.mips.start, other.mips.end]
            .into_iter()
            .find(|&at| inside(&self.mips, at))
        {
            let rest = Self {
                mips: at..self.mips.end,
                layers: self.layers.clone(),
            };
            self.mips.end = at;
            return Some(rest);
        }
        let at = [other.layers.start, other.layers.end]
            .into_iter()
            .find(|&at| inside(&self.layers, at))?;
        let rest = Self {
            mips: self.mips.clone(),
            layers: at..self.layers.end,
        };
        self.layers.end = at;
        Some(rest)
    }
}

/// A barrier recorded before pass `pass`, or after the last pass for the
/// boundaries of imported resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Barrier<A> {
    pub(crate) pass: usize,
    /// Index of the resource in the graph.
    pub(crate) resource: usize,
    pub(crate) subresources: Subresources,
    pub(crate) before: A,
    pub(crate) after: A,
}

/// The access of each block of mips and layers of one resource. Buffers
/// have one block of one subresource.
pub(crate) struct States<A> {
    /// Disjoint blocks covering every subresource.
    blocks: Vec<Block<A>>,
}

/// Subresources in the same access, and the barrier that put them there.
#[derive(Clone, Debug)]
struct Block<A> {
    subresources: Subresources,
    access: A,
    /// Index of the barrier that established `access`, if one did.
    origin: Option<usize>,
}

impl<A: Access> States<A> {
    /// `subresources`, all in `access`.
    pub(crate) fn new(subresources: Subresources, access: A) -> Self {
        Self {
            blocks: vec![Block {
                subresources,
                access,
                origin: None,
            }],
        }
    }

    /// The access of each block sharing a subresource with `subresources`.
    pub(crate) fn accesses(&self, subresources: &Subresources) -> impl Iterator<Item = A> {
        self.blocks
            .iter()
            .filter(|block| block.subresources.overlaps(subresources))
            .map(|block| block.access)
    }

    /// Moves `subresources` of `resource` to `access` in pass `pass`, adding
    /// the barriers this needs to `barriers`.
    ///
    /// Writes always need a barrier, even after the same write. A read that
    /// can share a state with the previous read widens it instead: the
    /// barrier that established the previous read is widened to cover this
    /// one too, so later reads need no barrier of their own. Only when no
    /// barrier established it, as for an import's initial access, is a
    /// read-to-read barrier added. At a `boundary`, the subresources end in
    /// exactly `access`.
    pub(crate) fn transition(
        &mut self,
        resource: usize,
        pass: usize,
        subresources: &Subresources,
        access: A,
        boundary: bool,
        barriers: &mut Vec<Barrier<A>>,
    ) {
        self.split(subresources);
        let inside = |block: &&mut Block<A>| subresources.contains(&block.subresources);
        for block in self.blocks.iter_mut().filter(inside) {
            let old = block.access;
            let new = if boundary {
                access
            } else {
                old.merge(access).unwrap_or(access)
            };
            if new == old && (boundary || !old.writes()) {
                continue;
            }
            // Widening a barrier that also covers other blocks, split off
            // since, would change their access too.
            if old.merge(new) == Some(new)
                && let Some(origin) = block.origin
                && barriers[origin].after == old
                && barriers[origin].subresources == block.subresources
            {
                barriers[origin].after = new;
            } else if old.orders() {
                block.origin = Some(barriers.len());
                barriers.push(Barrier {
                    pass,
                    resource,
                    subresources: block.subresources.clone(),
                    before: old,
                    after: new,
                });
            } else {
                block.origin = None;
            }
            block.access = new;
        }
    }

    /// Splits blocks so that each is inside `subresources` or apart from
    /// them.
    fn split(&mut self, subresources: &Subresources) {
        let mut index = 0;
        while index < self.blocks.len() {
            let block = &mut self.blocks[index];
            if block.subresources.overlaps(subresources)
                && let Some(rest) = block.subresources.cut(subresources)
            {
                // The block keeps its first part, which may need another cut.
                let rest = Block {
                    subresources: rest,
                    ..block.clone()
                };
                self.blocks.push(rest);
            } else {
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dirk_rhi::{ImageAccess, ShaderStages};

    use super::{Barrier, States, Subresources};

    const FRAGMENT: ImageAccess = ImageAccess::Sampled(ShaderStages::FRAGMENT);

    /// `mips` of `layers`.
    fn range(mips: std::ops::Range<u32>, layers: std::ops::Range<u32>) -> Subresources {
        Subresources { mips, layers }
    }

    #[test]
    fn mip_transitions_only_barrier_the_mips_they_access() {
        let mut states = States::new(Subresources::all(4, 1), FRAGMENT);
        let mut barriers = Vec::new();
        let copy = ImageAccess::CopyDestination;
        states.transition(0, 0, &range(2..3, 0..1), copy, false, &mut barriers);
        assert_eq!(
            barriers,
            [Barrier {
                pass: 0,
                resource: 0,
                subresources: range(2..3, 0..1),
                before: FRAGMENT,
                after: copy,
            }]
        );
        let all = Subresources::all(4, 1);
        assert_eq!(
            states.accesses(&all).collect::<Vec<_>>(),
            [FRAGMENT, copy, FRAGMENT]
        );

        // Reading every mip moves each block from its own access.
        barriers.clear();
        states.transition(0, 1, &all, ImageAccess::CopySource, false, &mut barriers);
        let blocks: Vec<_> = barriers
            .iter()
            .map(|barrier| (barrier.subresources.mips.clone(), barrier.before))
            .collect();
        assert_eq!(blocks, [(0..2, FRAGMENT), (2..3, copy), (3..4, FRAGMENT)]);
    }

    #[test]
    fn layer_and_mip_transitions_only_barrier_their_block() {
        let all = Subresources::all(3, 6);
        let mut states = States::new(all.clone(), ImageAccess::Undefined);
        let mut barriers = Vec::new();
        // Render one face's top mip, then sample it while rendering the
        // next mip of the same face.
        let face = range(0..1, 2..3);
        let next = range(1..2, 2..3);
        let color = ImageAccess::ColorAttachment;
        states.transition(0, 0, &face, color, false, &mut barriers);
        states.transition(0, 1, &face, FRAGMENT, false, &mut barriers);
        states.transition(0, 1, &next, color, false, &mut barriers);
        let blocks: Vec<_> = barriers
            .iter()
            .map(|barrier| (barrier.pass, barrier.subresources.clone(), barrier.after))
            .collect();
        assert_eq!(
            blocks,
            [
                (0, face.clone(), color),
                (1, face, FRAGMENT),
                (1, next, color)
            ]
        );
        // The other faces stay untouched.
        assert!(
            states
                .accesses(&range(0..3, 0..2))
                .all(|access| access == ImageAccess::Undefined)
        );
    }

    #[test]
    fn reads_widen_only_barriers_covering_exactly_their_block() {
        let mut states = States::new(Subresources::all(1, 2), ImageAccess::CopyDestination);
        let mut barriers = Vec::new();
        let vertex = ImageAccess::Sampled(ShaderStages::VERTEX);
        let both = ImageAccess::Sampled(ShaderStages::VERTEX | ShaderStages::FRAGMENT);
        states.transition(
            0,
            0,
            &Subresources::all(1, 2),
            FRAGMENT,
            false,
            &mut barriers,
        );
        // A later read of one layer cannot widen the barrier of both.
        states.transition(0, 1, &range(0..1, 1..2), vertex, false, &mut barriers);
        assert_eq!(barriers[0].after, FRAGMENT);
        assert_eq!(
            (
                barriers[1].subresources.clone(),
                barriers[1].before,
                barriers[1].after
            ),
            (range(0..1, 1..2), FRAGMENT, both)
        );
    }
}
