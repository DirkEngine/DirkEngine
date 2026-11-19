//! What of presentation runs without a display. Acquiring, presenting and
//! recreating swapchains need one; `examples/present.rs` checks them by hand.

use std::sync::Arc;

use dirk_rhi::{Error, Result};
use dirk_rhi_test_shaders::support::with_rhi;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};

/// A window that is already gone.
struct Closed;

impl HasWindowHandle for Closed {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

impl HasDisplayHandle for Closed {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

#[test]
fn surfaces_need_an_rhi_created_for_a_display() -> Result<()> {
    with_rhi(|rhi| {
        match rhi.create_surface(Arc::new(Closed)) {
            Err(Error::Invalid { reason, .. }) => assert!(reason.contains("display"), "{reason}"),
            Err(error) => panic!("expected an invalid request, got {error}"),
            Ok(_) => panic!("created a surface without a display"),
        }
        Ok(())
    })
}
