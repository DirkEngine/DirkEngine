use crate::{
    Error, Extent3d, Format, ImageDesc, ImageDimension, ImageUsage, Result, Rhi, RhiCreateInfo,
    SampleCount,
};

/// Runs `test` on an RHI without validation, to inspect its bookkeeping.
/// Skips it, with a note, on machines without a Vulkan driver or a suitable
/// device, unless `DIRK_REQUIRE_GPU=1` makes that fail.
fn with_rhi(test: impl FnOnce(&mut Rhi) -> Result<()>) -> Result<()> {
    let required = std::env::var_os("DIRK_REQUIRE_GPU").is_some_and(|required| required == "1");
    match Rhi::new(&RhiCreateInfo::default()) {
        Ok(mut rhi) => test(&mut rhi),
        Err(
            error @ (Error::NoSuitableDevice(_)
            | Error::Native {
                operation: "loading the Vulkan library",
                ..
            }),
        ) if !required => {
            eprintln!("skipping native test: {error}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

#[test]
fn destroying_a_dropped_image_forgets_its_handle() -> Result<()> {
    with_rhi(|rhi| {
        let image = rhi.create_image(&ImageDesc {
            label: "test image",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(4, 4),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::COPY_DST,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        })?;
        let id = image.raw.id();
        drop(image);
        // Destroys the image with its first transition still queued; a new
        // image the driver gives the same handle must not match either entry.
        rhi.wait_idle()?;
        assert!(
            !rhi.device.dropped_images.lock().contains(&id),
            "the destroyed image is still listed as dropped"
        );
        assert!(
            rhi.device
                .initializations
                .lock()
                .iter()
                .all(|transition| transition.image != id),
            "the destroyed image's first transition is still queued"
        );
        Ok(())
    })
}
