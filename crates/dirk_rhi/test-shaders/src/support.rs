//! Shared setup for the native tests of `dirk_rhi` and the crates built on
//! it.

use dirk_rhi::{Error, Result, Rhi, RhiCreateInfo};

/// Runs `test` on an RHI with validation enabled.
///
/// # Panics
///
/// Panics if validation reported an error.
///
/// # Errors
///
/// Returns the RHI's errors and `test`'s. Returns `Ok` without running
/// `test`, after printing why, on machines without a
/// graphics driver or a suitable device.
pub fn with_rhi(test: impl FnOnce(&mut Rhi) -> Result<()>) -> Result<()> {
    let Some(mut rhi) = rhi(&RhiCreateInfo {
        application_name: "dirk_rhi tests",
        validation: true,
        ..RhiCreateInfo::default()
    })?
    else {
        return Ok(());
    };
    test(&mut rhi)?;
    rhi.wait_idle()?;
    assert_eq!(
        rhi.validation_error_count().unwrap_or(0),
        0,
        "validation reported errors; they are logged to the `vulkan` or `metal` tracing target"
    );
    Ok(())
}

/// Creates an RHI, or returns `None` after printing why when the machine
/// cannot run native tests. Machines that must run them, such as CI machines
/// with a software driver, set the `DIRK_REQUIRE_GPU` environment variable
/// to `1` to fail instead, and to fail if validation is requested but its
/// layer is not installed.
///
/// # Panics
///
/// Panics if `DIRK_REQUIRE_GPU` is `1` and validation is requested but does
/// not run.
///
/// # Errors
///
/// Returns the errors of [`Rhi::new`] other than missing drivers or
/// devices.
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
