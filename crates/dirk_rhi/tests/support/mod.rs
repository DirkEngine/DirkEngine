//! Shared setup for native tests.

use dirk_rhi::{Error, Result, Rhi, RhiCreateInfo};

/// Runs `test` on an RHI with validation enabled, then fails if validation
/// reported any error. Skips the test, with a note, on machines without a
/// Vulkan driver or a suitable device.
pub fn with_rhi(test: impl FnOnce(&mut Rhi) -> Result<()>) -> Result<()> {
    let Some(mut rhi) = rhi(&RhiCreateInfo {
        application_name: "dirk_rhi tests",
        validation: true,
    })?
    else {
        return Ok(());
    };
    test(&mut rhi)?;
    assert_eq!(
        rhi.validation_error_count(),
        0,
        "validation reported errors; they are logged to the `vulkan` tracing target"
    );
    Ok(())
}

/// Creates an RHI, or returns `None` after printing why when the machine
/// cannot run native tests.
pub fn rhi(info: &RhiCreateInfo<'_>) -> Result<Option<Rhi>> {
    match Rhi::new(info) {
        Ok(rhi) => Ok(Some(rhi)),
        Err(
            error @ (Error::NoSuitableDevice(_)
            | Error::Native {
                operation: "loading the Vulkan library",
                ..
            }),
        ) => {
            eprintln!("skipping native test: {error}");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
