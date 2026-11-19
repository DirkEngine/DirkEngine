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
        rhi.validation_error_count().unwrap_or(0),
        0,
        "validation reported errors; they are logged to the `vulkan` tracing target"
    );
    Ok(())
}

/// Creates an RHI, or returns `None` after printing why when the machine
/// cannot run native tests.
///
/// Machines that must run native tests, such as CI machines with a software
/// driver, set `DIRK_REQUIRE_GPU=1` to fail instead, and to fail if
/// validation is requested but its layer is not installed.
pub fn rhi(info: &RhiCreateInfo<'_>) -> Result<Option<Rhi>> {
    let required = std::env::var_os("DIRK_REQUIRE_GPU").is_some_and(|required| required == "1");
    match Rhi::new(info) {
        Ok(rhi) => {
            assert!(
                !(required && info.validation && rhi.validation_error_count().is_none()),
                "DIRK_REQUIRE_GPU=1 requires validation, but its layer is not installed"
            );
            Ok(Some(rhi))
        }
        Err(
            error @ (Error::NoSuitableDevice(_)
            | Error::Native {
                operation: "loading the Vulkan library",
                ..
            }),
        ) if !required => {
            eprintln!("skipping native test: {error}");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
