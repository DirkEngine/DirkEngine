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
