use dirk_rhi::{ImageAccess, ShaderStages};

use super::{Access, Barrier, States, Subresources};

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

#[test]
fn swapchain_images_are_written_before_anything_else() {
    let present = ImageAccess::Present;
    assert!(ImageAccess::ColorAttachment.check_follows(present).is_ok());
    assert!(ImageAccess::CopyDestination.check_follows(present).is_ok());
    assert!(FRAGMENT.check_follows(present).is_err());
    // A swapchain import no pass wrote would be presented undefined.
    assert!(present.check_follows(present).is_err());
}
