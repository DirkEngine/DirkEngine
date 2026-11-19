//! Resource creation, its validation and host access.
#![expect(
    unsafe_code,
    reason = "Host reads and writes of a buffer are unsafe; the tests wait for the GPU first."
)]

use dirk_rhi::{
    BufferDesc, BufferUsage, Error, Extent3d, Features, Format, ImageDesc, ImageDimension,
    ImageUsage, ImageView, Memory, Result, Rhi, SampleCount, SamplerDesc, ViewType,
};
use dirk_rhi_test_shaders::support::with_rhi;

/// A single-mip 2D image description.
fn image_2d(format: Format, usage: ImageUsage) -> ImageDesc<'static> {
    ImageDesc {
        label: "test image",
        dimension: ImageDimension::D2,
        extent: Extent3d::new_2d(16, 16),
        format,
        usage,
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::X1,
        rest: None,
    }
}

/// Asserts that `result` was rejected as an invalid `object` whose reason
/// mentions `reason`.
#[expect(clippy::panic, reason = "Test helpers fail the test by panicking.")]
fn assert_invalid<T>(result: Result<T>, object: &str, reason: &str) {
    match result {
        Err(Error::Invalid {
            object: rejected,
            reason: why,
            ..
        }) => {
            assert_eq!(rejected, object);
            assert!(why.contains(reason), "`{why}` does not mention `{reason}`");
        }
        Err(error) => panic!("expected an invalid {object}, got {error}"),
        Ok(_) => panic!("expected an invalid {object}, but it was created"),
    }
}

#[test]
fn buffers_are_validated_at_creation() -> Result<()> {
    with_rhi(|rhi| {
        let desc = BufferDesc {
            label: "test buffer",
            size: 256,
            usage: BufferUsage::STORAGE,
            memory: Memory::Device,
        };
        let buffer = rhi.create_buffer(&desc)?;
        assert_eq!(buffer.info().size, 256);
        assert_eq!(buffer.label(), "test buffer");

        assert_invalid(
            rhi.create_buffer(&BufferDesc { size: 0, ..desc }),
            "buffer",
            "zero",
        );
        let too_large = rhi.capabilities().limits.max_buffer_size + 1;
        assert_invalid(
            rhi.create_buffer(&BufferDesc {
                size: too_large,
                ..desc
            }),
            "buffer",
            "limit",
        );
        assert_invalid(
            rhi.create_buffer(&BufferDesc {
                usage: BufferUsage::empty(),
                ..desc
            }),
            "buffer",
            "usage",
        );
        Ok(())
    })
}

#[test]
fn host_visible_buffers_round_trip_any_range() -> Result<()> {
    with_rhi(|rhi| {
        for memory in [Memory::Upload, Memory::Readback] {
            // An odd size exercises padding to whole non-coherent atoms.
            let mut buffer = rhi.create_buffer(&BufferDesc {
                label: "host buffer",
                size: 1001,
                usage: BufferUsage::COPY_SRC,
                memory,
            })?;
            let data: Vec<u8> = (0..=255).cycle().take(1001).collect();
            let mut read = vec![0; 997];
            // SAFETY: the buffer was never submitted to the GPU.
            unsafe {
                buffer.write(0, &data)?;
                buffer.write(1000, &[7])?;
                buffer.read(3, &mut read)?;
            }
            assert_eq!(read[..], data[3..1000]);
            let mut last = [0];
            // SAFETY: the buffer was never submitted to the GPU.
            unsafe { buffer.read(1000, &mut last)? };
            assert_eq!(last, [7]);

            // SAFETY: rejected before memory is touched.
            assert_invalid(unsafe { buffer.write(1000, &[0, 0]) }, "buffer", "exceeds");
            // SAFETY: rejected before memory is touched.
            assert_invalid(
                unsafe { buffer.read(u64::MAX, &mut last) },
                "buffer",
                "exceeds",
            );
        }
        Ok(())
    })
}

#[test]
fn device_local_buffers_reject_host_access() -> Result<()> {
    with_rhi(|rhi| {
        let mut buffer = rhi.create_buffer(&BufferDesc {
            label: "device buffer",
            size: 64,
            usage: BufferUsage::STORAGE,
            memory: Memory::Device,
        })?;
        // SAFETY: rejected before memory is touched.
        assert_invalid(unsafe { buffer.write(0, &[1]) }, "buffer", "device-local");
        Ok(())
    })
}

#[test]
fn only_images_used_through_a_view_have_one() -> Result<()> {
    with_rhi(|rhi| {
        let transfer = rhi.create_image(&image_2d(
            Format::Rgba8Unorm,
            ImageUsage::COPY_SRC | ImageUsage::COPY_DST,
        ))?;
        assert!(transfer.view().is_none());

        let sampled = rhi.create_image(&image_2d(
            Format::Rgba8Unorm,
            ImageUsage::COPY_DST | ImageUsage::SAMPLED,
        ))?;
        assert_eq!(sampled.view().map(ImageView::view_type), Some(ViewType::D2));

        let layers = rhi.create_image(&ImageDesc {
            array_layers: 4,
            ..image_2d(Format::Rgba16Float, ImageUsage::COLOR_ATTACHMENT)
        })?;
        assert_eq!(
            layers.view().map(ImageView::view_type),
            Some(ViewType::D2Array)
        );

        let cube = ImageDesc {
            dimension: ImageDimension::Cube,
            array_layers: 6,
            mip_levels: 5,
            ..image_2d(Format::Rgba8Srgb, ImageUsage::SAMPLED)
        };
        let image = rhi.create_image(&cube)?;
        assert_eq!(image.view().map(ImageView::view_type), Some(ViewType::Cube));
        let cube_array = rhi.create_image(&ImageDesc {
            array_layers: 12,
            ..cube
        });
        if rhi
            .capabilities()
            .features
            .contains(Features::IMAGE_CUBE_ARRAY)
        {
            assert_eq!(
                cube_array?.view().map(ImageView::view_type),
                Some(ViewType::CubeArray)
            );
        } else {
            assert_invalid(cube_array, "image", "cube array");
        }

        let depth = rhi.create_image(&image_2d(
            Format::Depth16Unorm,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
        ))?;
        assert_eq!(depth.view().map(ImageView::view_type), Some(ViewType::D2));
        Ok(())
    })
}

#[test]
fn images_are_validated_at_creation() -> Result<()> {
    with_rhi(|rhi| {
        let color = image_2d(Format::Rgba8Unorm, ImageUsage::SAMPLED);
        let reject = |rhi: &Rhi, desc: ImageDesc<'_>, reason: &str| {
            assert_invalid(rhi.create_image(&desc), "image", reason);
        };
        reject(
            rhi,
            ImageDesc {
                extent: Extent3d::new_2d(0, 16),
                ..color
            },
            "empty",
        );
        reject(
            rhi,
            ImageDesc {
                mip_levels: 6,
                ..color
            },
            "mip levels",
        );
        reject(
            rhi,
            ImageDesc {
                extent: Extent3d {
                    depth: 2,
                    ..color.extent
                },
                ..color
            },
            "depth of one",
        );
        reject(
            rhi,
            ImageDesc {
                dimension: ImageDimension::Cube,
                array_layers: 4,
                ..color
            },
            "six layers",
        );
        let limit = rhi.capabilities().limits.max_image_dimension_2d;
        reject(
            rhi,
            ImageDesc {
                extent: Extent3d::new_2d(limit + 1, 1),
                ..color
            },
            "limit",
        );
        reject(
            rhi,
            ImageDesc {
                usage: ImageUsage::empty(),
                ..color
            },
            "usage is empty",
        );
        reject(
            rhi,
            image_2d(Format::Depth16Unorm, ImageUsage::COLOR_ATTACHMENT),
            "does not support",
        );
        reject(
            rhi,
            ImageDesc {
                dimension: ImageDimension::D3,
                ..image_2d(Format::Depth16Unorm, ImageUsage::SAMPLED)
            },
            "not 3D",
        );
        reject(
            rhi,
            ImageDesc {
                samples: SampleCount::X4,
                rest: None,
                ..color
            },
            "multisampled",
        );
        Ok(())
    })
}

#[test]
fn multisampled_attachments_are_created() -> Result<()> {
    with_rhi(|rhi| {
        // Vulkan and Metal guarantee four samples for color attachments.
        let image = rhi.create_image(&ImageDesc {
            samples: SampleCount::X4,
            rest: None,
            ..image_2d(Format::Rgba8Unorm, ImageUsage::COLOR_ATTACHMENT)
        })?;
        assert_eq!(image.info().samples, SampleCount::X4);
        let layered = rhi.create_image(&ImageDesc {
            samples: SampleCount::X4,
            rest: None,
            array_layers: 2,
            ..image_2d(Format::Rgba8Unorm, ImageUsage::COLOR_ATTACHMENT)
        });
        assert_invalid(layered, "image", "single-layer");
        Ok(())
    })
}

#[test]
fn samplers_are_validated_at_creation() -> Result<()> {
    with_rhi(|rhi| {
        let max = rhi.capabilities().limits.max_sampler_anisotropy;
        rhi.create_sampler(&SamplerDesc {
            label: "anisotropic",
            max_anisotropy: max,
            ..SamplerDesc::default()
        })?;
        assert_invalid(
            rhi.create_sampler(&SamplerDesc {
                max_anisotropy: max + 1,
                ..SamplerDesc::default()
            }),
            "sampler",
            "anisotropy",
        );
        assert_invalid(
            rhi.create_sampler(&SamplerDesc {
                min_lod: 2.0,
                max_lod: 1.0,
                ..SamplerDesc::default()
            }),
            "sampler",
            "level of detail",
        );
        Ok(())
    })
}
