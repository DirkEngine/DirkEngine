use ash::vk;

use super::Swapchain;

#[test]
fn extents_follow_the_surface_or_clamp_to_it() {
    let capabilities = |current| vk::SurfaceCapabilitiesKHR {
        current_extent: current,
        min_image_extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
        max_image_extent: vk::Extent2D {
            width: 4096,
            height: 4096,
        },
        ..Default::default()
    };
    let extent = |width, height| vk::Extent2D { width, height };
    // Fixed surfaces dictate the extent.
    let fixed = capabilities(extent(800, 600));
    assert_eq!(Swapchain::extent(&fixed, (1, 1)), Some(extent(800, 600)));
    // Others take the wanted one, clamped.
    let free = capabilities(extent(u32::MAX, u32::MAX));
    assert_eq!(
        Swapchain::extent(&free, (640, 9000)),
        Some(extent(640, 4096))
    );
    // Minimized windows have no extent to present at.
    let minimized = capabilities(extent(0, 0));
    assert_eq!(Swapchain::extent(&minimized, (640, 480)), None);
}
