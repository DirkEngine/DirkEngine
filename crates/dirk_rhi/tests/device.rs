//! Device creation and capability queries.

use dirk_rhi::{Error, Format, FormatFeatures, ImageUsage, Result, RhiCreateInfo, SampleCounts};
use dirk_rhi_test_shaders::support::{rhi, with_rhi};

#[test]
fn capabilities_report_what_every_device_guarantees() -> Result<()> {
    with_rhi(|rhi| {
        let capabilities = rhi.capabilities();
        assert_ne!(capabilities.name, "");
        let limits = capabilities.limits;
        assert!(limits.max_buffer_size >= 1 << 30);
        assert!(limits.max_image_dimension_2d >= 4096);
        assert!(limits.max_image_array_layers >= 256);
        assert!((1..=16).contains(&limits.max_sampler_anisotropy));

        let rgba = capabilities.format(Format::Rgba8Unorm);
        assert!(rgba.usage.contains(
            ImageUsage::COPY_SRC
                | ImageUsage::COPY_DST
                | ImageUsage::SAMPLED
                | ImageUsage::STORAGE
                | ImageUsage::COLOR_ATTACHMENT
        ));
        assert!(rgba.features.contains(FormatFeatures::all()));
        assert!(
            rgba.sample_counts
                .contains(SampleCounts::X1 | SampleCounts::X4)
        );

        let depth = capabilities.format(Format::Depth16Unorm);
        assert!(depth.usage.contains(ImageUsage::DEPTH_STENCIL_ATTACHMENT));
        assert!(!depth.usage.contains(ImageUsage::COLOR_ATTACHMENT));
        Ok(())
    })
}

#[test]
fn application_names_with_nul_bytes_are_rejected() {
    let result = rhi(&RhiCreateInfo {
        application_name: "dirk\0rhi",
        validation: false,
        display: None,
    });
    assert!(
        matches!(
            result,
            Ok(None)
                | Err(Error::Invalid {
                    object: "application name",
                    ..
                })
        ),
        "expected the name to be rejected"
    );
}
