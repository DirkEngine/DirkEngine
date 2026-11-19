//! Frames, queues, completions, copies and deferred destruction.

use std::time::Duration;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CommandEncoder, CopyQueue, Error, Extent3d, FRAMES_IN_FLIGHT, Filter, Format, Graphics, Image,
    ImageAccess, ImageBarrier, ImageCopy, ImageDesc, ImageDimension, ImageRange, ImageUsage,
    Memory, PerFrame, QueueTransfer, QueueType, Result, Rhi, RhiCreateInfo, SampleCount,
    ShaderStages,
};
use dirk_rhi_test_shaders::support::{self, with_rhi};

const TIMEOUT: Duration = Duration::from_secs(10);

fn buffer(rhi: &Rhi, size: u64, usage: BufferUsage, memory: Memory) -> Result<Buffer> {
    rhi.create_buffer(&BufferDesc {
        label: "test buffer",
        size,
        usage,
        memory,
    })
}

fn image(rhi: &Rhi, extent: Extent3d, mip_levels: u32) -> Result<Image> {
    rhi.create_image(&ImageDesc {
        label: "test image",
        dimension: ImageDimension::D2,
        extent,
        format: Format::Rgba8Unorm,
        usage: ImageUsage::COPY_SRC | ImageUsage::COPY_DST,
        mip_levels,
        array_layers: 1,
        samples: SampleCount::X1,
        rest: None,
    })
}

/// Reads `len` bytes of a readback buffer whose writes completed.
fn read(buffer: &Buffer, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut data = vec![0; len];
    // SAFETY: callers wait for the work writing the buffer, which ends with a
    // barrier to host reads.
    unsafe { buffer.read(offset, &mut data)? };
    Ok(data)
}

/// Asserts that `result` was rejected with a reason mentioning `reason`.
fn assert_invalid<T>(result: Result<T>, reason: &str) {
    match result {
        Err(Error::Invalid { reason: why, .. }) => {
            assert!(why.contains(reason), "`{why}` does not mention `{reason}`");
        }
        Err(error) => panic!("expected an invalid request, got {error}"),
        Ok(_) => panic!("expected an invalid request, but it was accepted"),
    }
}

#[test]
fn copies_round_trip_from_the_copy_queue_to_the_graphics_queue() -> Result<()> {
    with_rhi(|rhi| {
        const SIZE: usize = 32 * 16 * 4;
        let extent = Extent3d::new_2d(32, 16);
        let data: Vec<u8> = (0..=250).cycle().take(SIZE).collect();
        let size = SIZE as u64;
        let mut staging = buffer(rhi, size, BufferUsage::COPY_SRC, Memory::Upload)?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, &data)? };
        let uploaded = buffer(
            rhi,
            size,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            Memory::Device,
        )?;
        let texture = image(rhi, extent, 1)?;
        let readback = buffer(rhi, 2 * size, BufferUsage::COPY_DST, Memory::Readback)?;
        let region = texture.info().mip_region(0);
        let packed = |offset| BufferImageCopy::packed(Format::Rgba8Unorm, offset, region);
        let transfer = Some(QueueTransfer {
            from: QueueType::Copy,
            to: QueueType::Graphics,
        });
        let handoff = (
            [BufferBarrier {
                buffer: &uploaded,
                before: BufferAccess::CopyDestination,
                after: BufferAccess::CopySource,
                transfer,
            }],
            [ImageBarrier {
                image: &texture,
                range: ImageRange::ALL,
                before: ImageAccess::CopyDestination,
                after: ImageAccess::CopySource,
                transfer,
            }],
        );

        let frame = rhi.begin_frame()?;
        let mut copy = frame.encoder::<CopyQueue>("upload")?;
        copy.copy_buffer(
            &staging,
            &uploaded,
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        )?;
        // SAFETY: the image is new and the barriers describe the copies.
        unsafe {
            copy.barrier(
                &[],
                &[ImageBarrier {
                    image: &texture,
                    range: ImageRange::ALL,
                    before: ImageAccess::Undefined,
                    after: ImageAccess::CopyDestination,
                    transfer: None,
                }],
            )?;
        }
        copy.copy_buffer_to_image(&staging, &texture, &[packed(0)])?;
        // SAFETY: releases what the copies wrote.
        unsafe { copy.barrier(&handoff.0, &handoff.1)? };
        let upload = frame.submit(copy)?;
        assert_eq!(upload.queue(), QueueType::Copy);

        let mut graphics = frame.encoder::<Graphics>("readback")?;
        graphics.wait_for(upload)?;
        // SAFETY: acquires what the copy queue released, then orders the
        // readback copies before host reads.
        unsafe { graphics.barrier(&handoff.0, &handoff.1)? };
        graphics.copy_buffer(
            &uploaded,
            &readback,
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        )?;
        graphics.copy_image_to_buffer(&texture, &readback, &[packed(size)])?;
        // SAFETY: orders the copies before host reads.
        unsafe {
            graphics.barrier(
                &[BufferBarrier {
                    buffer: &readback,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::HostRead,
                    transfer: None,
                }],
                &[],
            )?;
        }
        let done = frame.submit(graphics)?;
        frame.finish();

        rhi.wait(done, TIMEOUT)?;
        assert!(rhi.is_complete(upload)? && rhi.is_complete(done)?);
        assert_eq!(read(&readback, 0, data.len())?, data);
        assert_eq!(read(&readback, size, data.len())?, data);
        Ok(())
    })
}

#[test]
fn copies_within_one_buffer_or_image_need_disjoint_regions() -> Result<()> {
    with_rhi(|rhi| {
        let mut shared = buffer(
            rhi,
            64,
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            Memory::Readback,
        )?;
        let data: Vec<u8> = (0..64).collect();
        // SAFETY: the buffer was never submitted.
        unsafe { shared.write(0, &data)? };
        let texture = image(rhi, Extent3d::new_2d(8, 8), 2)?;
        let info = *texture.info();

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("disjoint copies")?;
        let copy = |src_offset, dst_offset| BufferCopy {
            src_offset,
            dst_offset,
            size: 16,
        };
        assert_invalid(
            encoder.copy_buffer(&shared, &shared, &[copy(0, 8)]),
            "overlaps",
        );
        assert_invalid(
            encoder.copy_buffer(&shared, &shared, &[copy(0, 32), copy(32, 48)]),
            "overlaps",
        );
        assert_invalid(
            encoder.copy_buffer(&shared, &shared, &[copy(0, 32), copy(16, 40)]),
            "destinations",
        );
        encoder.copy_buffer(&shared, &shared, &[copy(0, 32), copy(16, 48)])?;
        // Mips of one image are disjoint, so a mip can be blitted from the one
        // above it; the same mip cannot.
        let mip = |level| ImageCopy {
            src: info.mip_region(level),
            dst: info.mip_region(level),
        };
        assert_invalid(
            encoder.copy_image(&texture, &texture, &[mip(1)]),
            "overlaps",
        );
        // Nor can disjoint halves of one mip, which is in one layout.
        let mut halves = mip(0);
        (halves.src.extent.width, halves.dst.extent.width) = (4, 4);
        halves.dst.origin.x = 4;
        assert_invalid(
            encoder.copy_image(&texture, &texture, &[halves]),
            "overlaps",
        );
        let downsample = ImageCopy {
            src: info.mip_region(0),
            dst: info.mip_region(1),
        };
        assert_invalid(
            encoder.copy_image(&texture, &texture, &[downsample]),
            "changes size",
        );
        encoder.blit_image(&texture, &texture, &[downsample], Filter::Linear)?;
        // SAFETY: orders the buffer copy before host reads.
        unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &shared,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::HostRead,
                    transfer: None,
                }],
                &[],
            )?;
        }
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, TIMEOUT)?;

        let result = read(&shared, 0, 64)?;
        assert_eq!(result[32..48], data[0..16]);
        assert_eq!(result[48..64], data[16..32]);
        Ok(())
    })
}

#[test]
fn copy_queue_barriers_dropped_unsubmitted_leave_new_images_to_the_graphics_queue() -> Result<()> {
    with_rhi(|rhi| {
        let texture = image(rhi, Extent3d::new_2d(4, 4), 1)?;
        let readback = buffer(rhi, 64, BufferUsage::COPY_DST, Memory::Readback)?;
        let frame = rhi.begin_frame()?;
        let mut copy = frame.encoder::<CopyQueue>("abandoned upload")?;
        // SAFETY: the image is new, and the encoder is never submitted.
        unsafe {
            copy.barrier(
                &[],
                &[ImageBarrier {
                    image: &texture,
                    range: ImageRange::ALL,
                    before: ImageAccess::Undefined,
                    after: ImageAccess::CopyDestination,
                    transfer: None,
                }],
            )?;
        }
        drop(copy);
        // The graphics queue's next submission still moves the image to
        // rest, where the copy finds it.
        let mut encoder = frame.encoder::<Graphics>("read")?;
        let region = texture.info().mip_region(0);
        let copy = BufferImageCopy::packed(Format::Rgba8Unorm, 0, region);
        encoder.copy_image_to_buffer(&texture, &readback, &[copy])?;
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, TIMEOUT)
    })
}

#[test]
fn blits_scale_regions_with_filtering() -> Result<()> {
    with_rhi(|rhi| {
        // A 4x4 image of four quadrants: red, green, blue and white.
        let quadrant = |x, y| match (x < 2, y < 2) {
            (true, true) => [255, 0, 0, 255],
            (false, true) => [0, 255, 0, 255],
            (true, false) => [0, 0, 255, 255],
            (false, false) => [255; 4],
        };
        let texels: Vec<[u8; 4]> = (0..16)
            .map(|index| quadrant(index % 4, index / 4))
            .collect();
        let mut staging = buffer(rhi, 64, BufferUsage::COPY_SRC, Memory::Upload)?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, texels.as_flattened())? };
        let texture = image(rhi, Extent3d::new_2d(4, 4), 3)?;
        let readback = buffer(rhi, 20, BufferUsage::COPY_DST, Memory::Readback)?;
        let info = *texture.info();
        let mip = |level| ImageCopy {
            src: info.mip_region(level - 1),
            dst: info.mip_region(level),
        };
        let packed = |offset, level| {
            BufferImageCopy::packed(Format::Rgba8Unorm, offset, info.mip_region(level))
        };

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("blits")?;
        // Each copy and blit moves the mips it touches from rest and back.
        encoder.copy_buffer_to_image(&staging, &texture, &[packed(0, 0)])?;
        encoder.blit_image(&texture, &texture, &[mip(1)], Filter::Linear)?;
        encoder.blit_image(&texture, &texture, &[mip(2)], Filter::Linear)?;
        encoder.copy_image_to_buffer(&texture, &readback, &[packed(0, 1), packed(16, 2)])?;
        // SAFETY: orders the copies before host reads.
        unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &readback,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::HostRead,
                    transfer: None,
                }],
                &[],
            )?;
        }
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, TIMEOUT)?;

        // Each texel of mip 1 filters one quadrant, and mip 2 all of them.
        let pixels = read(&readback, 0, 20)?;
        let expected: Vec<u8> = (0..4)
            .flat_map(|index| quadrant(index % 2 * 2, index / 2 * 2))
            .collect();
        assert_eq!(pixels[..16], expected);
        let average = &pixels[16..];
        assert!(
            average[..3]
                .iter()
                .all(|channel| channel.abs_diff(128) <= 1)
                && average[3] == 255,
            "{average:?} is not the average of the quadrants"
        );
        Ok(())
    })
}

#[test]
fn generated_mips_filter_every_level_and_layer_in_linear_space() -> Result<()> {
    with_rhi(|rhi| {
        // Two layers of 7x5 in one uniform color each, so every mip of a
        // layer keeps its color however odd sizes are filtered; and a 2x1
        // black and white image, whose mip averages them in linear space. A
        // 1x1 image has no mips to generate.
        let colors = [[200, 100, 50, 255], [10, 220, 120, 128]];
        let layers = Extent3d::new_2d(7, 5);
        let pair = Extent3d::new_2d(2, 1);
        let create = |label, extent: Extent3d, array_layers| {
            rhi.create_image(&ImageDesc {
                label,
                dimension: ImageDimension::D2,
                extent,
                format: Format::Rgba8Srgb,
                usage: ImageUsage::COPY_SRC | ImageUsage::COPY_DST,
                mip_levels: extent.width.max(extent.height).bit_width(),
                array_layers,
                samples: SampleCount::X1,
                rest: None,
            })
        };
        let layered = create("layers", layers, 2)?;
        let black_and_white = create("black and white", pair, 1)?;
        let single = create("single mip", Extent3d::new_2d(1, 1), 1)?;
        let mut texels: Vec<u8> = colors.iter().flat_map(|color| color.repeat(35)).collect();
        let pair_offset = texels.len() as u64;
        texels.extend([0, 0, 0, 255, 255, 255, 255, 255]);
        let mut staging = buffer(
            rhi,
            texels.len() as u64,
            BufferUsage::COPY_SRC,
            Memory::Upload,
        )?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, &texels)? };
        // Every mip of both layers, then mip 1 of the pair.
        let mips: Vec<_> = (0..layered.info().mip_levels)
            .map(|level| (&layered, layered.info().mip_region(level)))
            .chain([(&black_and_white, black_and_white.info().mip_region(1))])
            .collect();
        let mut offsets = Vec::new();
        let mut size = 0;
        for (_, region) in &mips {
            offsets.push(size);
            size += u64::from(region.extent.width * region.extent.height * region.layer_count) * 4;
        }
        let readback = buffer(rhi, size, BufferUsage::COPY_DST, Memory::Readback)?;
        let packed = |offset, region| BufferImageCopy::packed(Format::Rgba8Srgb, offset, region);

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("mips")?;
        let base = |image: &Image| image.info().mip_region(0);
        encoder.copy_buffer_to_image(&staging, &layered, &[packed(0, base(&layered))])?;
        let pair_base = packed(pair_offset, base(&black_and_white));
        encoder.copy_buffer_to_image(&staging, &black_and_white, &[pair_base])?;
        // Each image is at rest, where generating its mips leaves it.
        for image in [&layered, &black_and_white, &single] {
            encoder.generate_mips(image)?;
        }
        for ((image, region), &offset) in mips.iter().zip(&offsets) {
            encoder.copy_image_to_buffer(image, &readback, &[packed(offset, *region)])?;
        }
        // SAFETY: the readback buffer was only copied into.
        unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &readback,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::HostRead,
                    transfer: None,
                }],
                &[],
            )?;
        }
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, TIMEOUT)?;

        // Rounding may differ by one; filtering in sRGB space would give 128
        // instead of 188 for the pair.
        let expected: Vec<u8> = mips[..mips.len() - 1]
            .iter()
            .flat_map(|(_, region)| {
                let texels = (region.extent.width * region.extent.height) as usize;
                colors.iter().flat_map(move |color| color.repeat(texels))
            })
            .chain([188, 188, 188, 255])
            .collect();
        let pixels = read(&readback, 0, expected.len())?;
        assert!(
            pixels
                .iter()
                .zip(&expected)
                .all(|(texel, expected)| texel.abs_diff(*expected) <= 1),
            "generated {pixels:?} instead of {expected:?}"
        );
        Ok(())
    })
}

#[test]
fn safe_copies_find_images_in_their_resting_access() -> Result<()> {
    with_rhi(|rhi| {
        const SIZE: usize = 8 * 8 * 4;
        let data: Vec<u8> = (0..=255).cycle().take(SIZE).collect();
        let mut staging = buffer(rhi, SIZE as u64, BufferUsage::COPY_SRC, Memory::Upload)?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, &data)? };
        let readback = buffer(
            rhi,
            2 * SIZE as u64,
            BufferUsage::COPY_DST,
            Memory::Readback,
        )?;
        let desc = |rest| ImageDesc {
            label: "resting",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(8, 8),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED | ImageUsage::COPY_SRC | ImageUsage::COPY_DST,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest,
        };
        let sampled = rhi.create_image(&desc(None))?;
        let fragment = ImageAccess::Sampled(ShaderStages::all());
        assert_eq!(sampled.info().rest, fragment);
        let copied = rhi.create_image(&desc(Some(ImageAccess::CopySource)))?;
        assert_eq!(copied.info().rest, ImageAccess::CopySource);
        assert_invalid(
            rhi.create_image(&desc(Some(ImageAccess::ColorAttachment))),
            "cannot rest in",
        );
        let region = sampled.info().mip_region(0);
        let packed = |offset| BufferImageCopy::packed(Format::Rgba8Unorm, offset, region);

        // Each frame's copies move the new images from their first,
        // undefined contents, and from rest to rest, with no barrier given.
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("fill")?;
        encoder.copy_buffer_to_image(&staging, &sampled, &[packed(0)])?;
        encoder.copy_image(
            &sampled,
            &copied,
            &[ImageCopy {
                src: region,
                dst: region,
            }],
        )?;
        frame.submit(encoder)?;
        frame.finish();
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("read")?;
        encoder.copy_image_to_buffer(&sampled, &readback, &[packed(0)])?;
        encoder.copy_image_to_buffer(&copied, &readback, &[packed(SIZE as u64)])?;
        // SAFETY: orders the copies before host reads.
        unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &readback,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::HostRead,
                    transfer: None,
                }],
                &[],
            )?;
        }
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, TIMEOUT)?;
        assert_eq!(read(&readback, 0, SIZE)?, data);
        assert_eq!(read(&readback, SIZE as u64, SIZE)?, data);

        // A raw barrier moving the image elsewhere is remembered until it
        // moves it back to rest.
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("moved")?;
        let barrier = |before, after| ImageBarrier {
            image: &sampled,
            range: ImageRange::ALL,
            before,
            after,
            transfer: None,
        };
        // SAFETY: the image is at rest, and moved back before submitting.
        unsafe { encoder.barrier(&[], &[barrier(fragment, ImageAccess::CopySource)])? };
        assert_invalid(
            encoder.copy_buffer_to_image(&staging, &sampled, &[packed(0)]),
            "where a barrier left it",
        );
        encoder.copy_image_to_buffer(&sampled, &readback, &[packed(0)])?;
        // SAFETY: the copy read it in `CopySource`.
        unsafe { encoder.barrier(&[], &[barrier(ImageAccess::CopySource, fragment)])? };
        encoder.copy_buffer_to_image(&staging, &sampled, &[packed(0)])?;
        // The copy queue cannot reach resting accesses.
        let mut copy_queue = frame.encoder::<CopyQueue>("copies")?;
        assert_invalid(
            copy_queue.copy_buffer_to_image(&staging, &copied, &[packed(0)]),
            "copy queue cannot reach",
        );
        frame.submit(encoder)?;
        frame.finish();
        Ok(())
    })
}

#[test]
fn recording_rejects_invalid_requests() -> Result<()> {
    with_rhi(|rhi| {
        let source = buffer(rhi, 64, BufferUsage::COPY_SRC, Memory::Device)?;
        let destination = buffer(rhi, 64, BufferUsage::COPY_DST, Memory::Device)?;
        let texture = image(rhi, Extent3d::new_2d(4, 4), 1)?;
        let copy = |size| BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<CopyQueue>("invalid")?;
        assert_invalid(
            encoder.copy_buffer(&destination, &source, &[copy(64)]),
            "COPY_SRC",
        );
        assert_invalid(
            encoder.copy_buffer(&source, &destination, &[copy(6)]),
            "four bytes",
        );
        assert_invalid(
            encoder.copy_buffer(&source, &destination, &[copy(128)]),
            "bounds",
        );
        assert_invalid(
            encoder.copy_buffer(&source, &destination, &[]),
            "no regions",
        );
        let mut region = texture.info().mip_region(0);
        region.extent.width = 5;
        assert_invalid(
            encoder.copy_buffer_to_image(
                &source,
                &texture,
                &[BufferImageCopy::packed(Format::Rgba8Unorm, 0, region)],
            ),
            "outside mip 0",
        );
        // SAFETY: rejected before recording.
        let sampled = unsafe {
            encoder.barrier(
                &[],
                &[ImageBarrier {
                    image: &texture,
                    range: ImageRange::ALL,
                    before: ImageAccess::Undefined,
                    after: ImageAccess::Sampled(ShaderStages::FRAGMENT),
                    transfer: None,
                }],
            )
        };
        assert_invalid(sampled, "SAMPLED");
        // SAFETY: rejected before recording.
        let storage = unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &destination,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::ShaderRead(ShaderStages::COMPUTE),
                    transfer: None,
                }],
                &[],
            )
        };
        assert_invalid(storage, "STORAGE");
        // SAFETY: rejected before recording.
        let stageless = unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &destination,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::ShaderRead(ShaderStages::empty()),
                    transfer: None,
                }],
                &[],
            )
        };
        assert_invalid(stageless, "names no shader stage");
        // SAFETY: rejected before recording.
        let unrelated = unsafe {
            encoder.barrier(
                &[BufferBarrier {
                    buffer: &destination,
                    before: BufferAccess::CopyDestination,
                    after: BufferAccess::CopyDestination,
                    transfer: Some(QueueTransfer {
                        from: QueueType::Graphics,
                        to: QueueType::Graphics,
                    }),
                }],
                &[],
            )
        };
        assert_invalid(unrelated, "itself");
        frame.finish();

        Ok(())
    })
}

#[test]
fn objects_of_another_device_are_rejected() -> Result<()> {
    with_rhi(|rhi| {
        let Some(mut other) = support::rhi(&RhiCreateInfo {
            application_name: "second dirk_rhi test device",
            validation: true,
            ..RhiCreateInfo::default()
        })?
        else {
            return Ok(());
        };
        let foreign = buffer(&other, 64, BufferUsage::COPY_DST, Memory::Device)?;
        let source = buffer(rhi, 64, BufferUsage::COPY_SRC, Memory::Device)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<CopyQueue>("foreign copy")?;
        assert_invalid(
            encoder.copy_buffer(
                &source,
                &foreign,
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 64,
                }],
            ),
            "another device",
        );
        // This device's own submission reaches as far as the other's below.
        frame.submit(frame.encoder::<Graphics>("own")?)?;
        frame.finish();
        drop(foreign);

        // A completion from another device.
        let frame = other.begin_frame()?;
        let elsewhere = frame.submit(frame.encoder::<Graphics>("elsewhere")?)?;
        frame.finish();
        assert_invalid(rhi.wait(elsewhere, TIMEOUT), "not submitted");
        assert_invalid(rhi.is_complete(elsewhere), "not submitted");
        other.wait_idle()?;
        assert_eq!(other.validation_error_count().unwrap_or(0), 0);
        Ok(())
    })
}

#[test]
fn dropped_resources_outlive_the_work_using_them() -> Result<()> {
    with_rhi(|rhi| {
        // Large enough to still be copying when the buffers are dropped.
        const SIZE: u64 = 256 << 20;
        let usage = BufferUsage::COPY_SRC | BufferUsage::COPY_DST;
        let source = buffer(rhi, SIZE, usage, Memory::Device)?;
        let destination = buffer(rhi, SIZE, usage, Memory::Device)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("long copy")?;
        let copy = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: SIZE,
        };
        encoder.copy_buffer(&source, &destination, &[copy])?;
        let done = frame.submit(encoder)?;
        // Dropped while the copies may still run: validation reports any
        // destruction of a buffer in use.
        drop((source, destination));
        frame.finish();
        let pending = !rhi.is_complete(done)?;
        for _ in 0..=FRAMES_IN_FLIGHT {
            rhi.begin_frame()?.finish();
        }
        assert!(rhi.is_complete(done)?);
        if !pending {
            eprintln!("the copies finished before the buffers were dropped");
        }
        Ok(())
    })
}

#[test]
fn per_frame_values_follow_frame_slots() -> Result<()> {
    with_rhi(|rhi| {
        let values = PerFrame::new(rhi, |slot| slot);
        let mut seen = Vec::new();
        for _ in 0..3 * FRAMES_IN_FLIGHT {
            seen.push(*values.get(&rhi.begin_frame()?));
        }
        for (index, slot) in seen.iter().enumerate() {
            assert_eq!(*slot, seen[index % FRAMES_IN_FLIGHT]);
        }
        assert_ne!(seen[0], seen[1]);
        Ok(())
    })
}

/// Records a copy of `size` bytes at `offset` of `source` into `readback`
/// at `at`, made visible to host reads.
fn read_back(
    encoder: &mut CommandEncoder<'_, Graphics>,
    (source, offset, size): (&Buffer, u64, u64),
    readback: &Buffer,
    at: u64,
) -> Result<()> {
    let copy = BufferCopy {
        src_offset: offset,
        dst_offset: at,
        size,
    };
    encoder.copy_buffer(source, readback, &[copy])?;
    // SAFETY: orders the copy before host reads.
    unsafe {
        encoder.barrier(
            &[BufferBarrier {
                buffer: readback,
                before: BufferAccess::CopyDestination,
                after: BufferAccess::HostRead,
                transfer: None,
            }],
            &[],
        )
    }
}

#[test]
fn frame_allocations_are_not_reused_while_their_frame_runs() -> Result<()> {
    with_rhi(|rhi| {
        const LEN: u32 = 1 << 12;
        const SIZE: u64 = LEN as u64 * 4;
        let frames = 3 * FRAMES_IN_FLIGHT;
        let data = |frame: usize| -> Vec<u32> {
            let seed = u32::try_from(frame).expect("few frames");
            (0..LEN).map(|i| i * 7 + seed).collect()
        };
        let readback = buffer(
            rhi,
            SIZE * frames as u64,
            BufferUsage::COPY_DST,
            Memory::Readback,
        )?;
        let mut places = Vec::new();
        let mut last = None;
        // No frame waits for its copies, so frames in flight overlap.
        for index in 0..frames {
            let frame = rhi.begin_frame()?;
            let slice = frame.alloc(&data(index))?;
            assert_eq!((slice.len(), slice.size()), (LEN, SIZE));
            places.push((std::ptr::from_ref(slice.buffer()), slice.offset()));
            let mut encoder = frame.encoder::<Graphics>("read back allocation")?;
            let source = (slice.buffer(), slice.offset(), SIZE);
            read_back(&mut encoder, source, &readback, SIZE * index as u64)?;
            last = Some(frame.submit(encoder)?);
        }
        rhi.wait(last.expect("frames were submitted"), TIMEOUT)?;

        // Frames in flight allocate from different memory, which later
        // frames of their slots reuse.
        for (index, place) in places.iter().enumerate() {
            assert_eq!(*place, places[index % FRAMES_IN_FLIGHT]);
        }
        assert_ne!(places[0].0, places[1].0);
        let read = read(&readback, 0, LEN as usize * 4 * frames)?;
        for (index, read) in read.chunks(LEN as usize * 4).enumerate() {
            assert_eq!(
                read,
                bytemuck::cast_slice::<u32, u8>(&data(index)),
                "frame {index}"
            );
        }
        Ok(())
    })
}

#[test]
fn frame_allocations_too_large_for_frame_memory_get_buffers_of_their_own() -> Result<()> {
    with_rhi(|rhi| {
        let large = vec![
            0x5a_u8;
            usize::try_from(RhiCreateInfo::default().frame_memory).expect("fits") + 4
        ];
        let readback = buffer(rhi, 1 << 10, BufferUsage::COPY_DST, Memory::Readback)?;
        let frame = rhi.begin_frame()?;
        let small = frame.alloc(&[1_u8, 2, 3])?;
        let oversized = frame.alloc(&large)?;
        // Three-word elements start at a multiple of their size and 16.
        let after = frame.alloc(&[[4_u32, 5, 6], [7, 8, 9]])?;
        assert!(!std::ptr::eq(small.buffer(), oversized.buffer()));
        assert_eq!(oversized.offset(), 0);
        assert!(std::ptr::eq(small.buffer(), after.buffer()));
        assert_eq!(after.offset() % 48, 0);

        let mut encoder = frame.encoder::<Graphics>("read back allocations")?;
        let tail = large.len() as u64 - 4;
        let oversized = (oversized.buffer(), oversized.offset() + tail, 4);
        read_back(&mut encoder, oversized, &readback, 0)?;
        read_back(
            &mut encoder,
            (after.buffer(), after.offset(), 24),
            &readback,
            4,
        )?;
        let done = frame.submit(encoder)?;
        // The oversized buffer is freed once the copy is done: validation
        // reports freeing it earlier.
        frame.finish();
        rhi.wait(done, TIMEOUT)?;

        let words: Vec<u8> = [4_u32, 5, 6, 7, 8, 9]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        assert_eq!(read(&readback, 0, 28)?, [&[0x5a; 4][..], &words].concat());
        Ok(())
    })
}
