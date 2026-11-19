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
        ImageAccess::shader_stages(self)
    }

    fn in_pass(self) -> bool {
        !matches!(self, Self::Undefined | Self::Present)
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
        BufferAccess::shader_stages(self)
    }

    fn in_pass(self) -> bool {
        !matches!(self, Self::None | Self::HostRead)
    }

    fn check_follows(self, _previous: Self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// A barrier recorded before pass `pass`, or after the last pass for the
/// boundaries of imported resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Barrier<A> {
    pub(crate) pass: usize,
    /// Index of the resource in the graph.
    pub(crate) resource: usize,
    /// The mip levels; `0..1` for buffers.
    pub(crate) mips: Range<u32>,
    pub(crate) before: A,
    pub(crate) after: A,
}

/// The access of each span of mips of one resource. Buffers have one span of
/// one level.
pub(crate) struct States<A> {
    /// Sorted spans covering every mip level.
    spans: Vec<Span<A>>,
}

/// Mips in the same access, and the barrier that put them there.
#[derive(Clone, Debug)]
struct Span<A> {
    mips: Range<u32>,
    access: A,
    /// Index of the barrier that established `access`, if one did.
    origin: Option<usize>,
}

impl<A: Access> States<A> {
    /// `mips` levels, all in `access`.
    pub(crate) fn new(mips: u32, access: A) -> Self {
        Self {
            spans: vec![Span {
                mips: 0..mips,
                access,
                origin: None,
            }],
        }
    }

    /// The access of each span.
    pub(crate) fn accesses(&self) -> impl Iterator<Item = A> {
        self.spans.iter().map(|span| span.access)
    }

    /// Moves `mips` of `resource` to `access` in pass `pass`, adding the
    /// barriers this needs to `barriers`.
    ///
    /// Writes always need a barrier, even after the same write. A read that
    /// can share a state with the previous read widens it instead: the
    /// barrier that established the previous read is widened to cover this
    /// one too, so later reads need no barrier of their own. Only when no
    /// barrier established it, as for an import's initial access, is a
    /// read-to-read barrier added. At a `boundary`, the mips end in exactly
    /// `access`.
    pub(crate) fn transition(
        &mut self,
        resource: usize,
        pass: usize,
        mips: Range<u32>,
        access: A,
        boundary: bool,
        barriers: &mut Vec<Barrier<A>>,
    ) {
        self.split(mips.start);
        self.split(mips.end);
        let inside =
            |span: &&mut Span<A>| mips.start <= span.mips.start && span.mips.end <= mips.end;
        for span in self.spans.iter_mut().filter(inside) {
            let old = span.access;
            let new = if boundary {
                access
            } else {
                old.merge(access).unwrap_or(access)
            };
            if new == old && (boundary || !old.writes()) {
                continue;
            }
            if old.merge(new) == Some(new)
                && let Some(origin) = span.origin
                && barriers[origin].after == old
            {
                barriers[origin].after = new;
            } else if old.orders() {
                span.origin = Some(barriers.len());
                barriers.push(Barrier {
                    pass,
                    resource,
                    mips: span.mips.clone(),
                    before: old,
                    after: new,
                });
            } else {
                span.origin = None;
            }
            span.access = new;
        }
    }

    /// Splits the span containing mip `at` so that a span starts there.
    fn split(&mut self, at: u32) {
        let Some(index) = self
            .spans
            .iter()
            .position(|span| span.mips.start < at && at < span.mips.end)
        else {
            return;
        };
        let span = &mut self.spans[index];
        let tail = Span {
            mips: at..span.mips.end,
            ..span.clone()
        };
        span.mips.end = at;
        self.spans.insert(index + 1, tail);
    }
}

#[cfg(test)]
mod tests {
    use dirk_rhi::{ImageAccess, ShaderStages};

    use super::{Access, Barrier, States};

    const FRAGMENT: ImageAccess = ImageAccess::Sampled(ShaderStages::FRAGMENT);

    #[test]
    fn mip_transitions_only_barrier_the_mips_they_access() {
        let mut states = States::new(4, FRAGMENT);
        let mut barriers = Vec::new();
        states.transition(
            0,
            0,
            2..3,
            ImageAccess::CopyDestination,
            false,
            &mut barriers,
        );
        assert_eq!(
            barriers,
            [Barrier {
                pass: 0,
                resource: 0,
                mips: 2..3,
                before: FRAGMENT,
                after: ImageAccess::CopyDestination,
            }]
        );
        assert_eq!(
            states.accesses().collect::<Vec<_>>(),
            [FRAGMENT, ImageAccess::CopyDestination, FRAGMENT]
        );

        // Reading every mip moves each span from its own access.
        barriers.clear();
        states.transition(0, 1, 0..4, ImageAccess::CopySource, false, &mut barriers);
        let spans: Vec<_> = barriers
            .iter()
            .map(|barrier| (barrier.mips.clone(), barrier.before))
            .collect();
        assert_eq!(
            spans,
            [
                (0..2, FRAGMENT),
                (2..3, ImageAccess::CopyDestination),
                (3..4, FRAGMENT)
            ]
        );
    }

    #[test]
    fn swapchain_images_are_written_before_anything_else() {
        let present = ImageAccess::Present;
        assert!(ImageAccess::ColorAttachment.check_follows(present).is_ok());
        assert!(ImageAccess::CopyDestination.check_follows(present).is_ok());
        assert!(FRAGMENT.check_follows(present).is_err());
        // A swapchain import no pass wrote would be presented undefined.
        assert!(present.check_follows(present).is_err());
    }
}
