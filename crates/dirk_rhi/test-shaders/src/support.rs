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
/// graphics driver or a suitable device, unless [`rhi`] must create one.
pub fn with_rhi<E: From<Error>>(test: impl FnOnce(&mut Rhi) -> Result<(), E>) -> Result<(), E> {
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
        rhi.validation_error_count(),
        0,
        "validation reported errors; they are logged to the `vulkan` or `metal` tracing target"
    );
    Ok(())
}

/// Creates an RHI, or returns `None` after printing why when the machine
/// cannot run native tests, unless the `DIRK_REQUIRE_GPU` environment
/// variable is `1`.
///
/// # Errors
///
/// Returns the errors of [`Rhi::new`] other than missing drivers or
/// devices.
pub fn rhi(info: &RhiCreateInfo<'_>) -> Result<Option<Rhi>> {
    match Rhi::new(info) {
        Ok(rhi) => Ok(Some(rhi)),
        // Machines that must run native tests, such as CI machines with a
        // software driver, set `DIRK_REQUIRE_GPU=1` to fail instead.
        Err(
            error @ (Error::NoSuitableDevice(_)
            | Error::Native {
                operation: "loading the Vulkan library",
                ..
            }),
        ) if std::env::var_os("DIRK_REQUIRE_GPU").is_none_or(|required| required != "1") => {
            eprintln!("skipping native test: {error}");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
