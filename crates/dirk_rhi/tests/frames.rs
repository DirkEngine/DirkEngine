//! Frames, queues, completions, copies and deferred destruction.

use std::time::Duration;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CopyQueue, Error, Extent3d, FRAMES_IN_FLIGHT, Features, Filter, Format, Graphics, Image,
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
        let downsample = ImageCopy {
            src: info.mip_region(0),
            dst: info.mip_region(1),
        };
        assert_invalid(
            encoder.copy_image(&texture, &texture, &[downsample]),
            "changes size",
        );
        // SAFETY: the image is new; mip 0 becomes a source and mip 1 a
        // destination.
        unsafe {
            encoder.barrier(
                &[],
                &[
                    ImageBarrier {
                        image: &texture,
                        range: ImageRange::mip(0),
                        before: ImageAccess::Undefined,
                        after: ImageAccess::CopySource,
                        transfer: None,
                    },
                    ImageBarrier {
                        image: &texture,
                        range: ImageRange::mip(1),
                        before: ImageAccess::Undefined,
                        after: ImageAccess::CopyDestination,
                        transfer: None,
                    },
                ],
            )?;
        }
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
        let barrier = |level, before, after| ImageBarrier {
            image: &texture,
            range: ImageRange::mip(level),
            before,
            after,
            transfer: None,
        };
        let packed = |offset, level| {
            BufferImageCopy::packed(Format::Rgba8Unorm, offset, info.mip_region(level))
        };

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("blits")?;
        // SAFETY: the barriers describe each mip's accesses in order.
        unsafe {
            encoder.barrier(
                &[],
                &[barrier(
                    0,
                    ImageAccess::Undefined,
                    ImageAccess::CopyDestination,
                )],
            )?;
            encoder.copy_buffer_to_image(&staging, &texture, &[packed(0, 0)])?;
            encoder.barrier(
                &[],
                &[
                    barrier(0, ImageAccess::CopyDestination, ImageAccess::CopySource),
                    barrier(1, ImageAccess::Undefined, ImageAccess::CopyDestination),
                ],
            )?;
            encoder.blit_image(&texture, &texture, &[mip(1)], Filter::Linear)?;
            encoder.barrier(
                &[],
                &[
                    barrier(1, ImageAccess::CopyDestination, ImageAccess::CopySource),
                    barrier(2, ImageAccess::Undefined, ImageAccess::CopyDestination),
                ],
            )?;
            encoder.blit_image(&texture, &texture, &[mip(2)], Filter::Linear)?;
            encoder.barrier(
                &[],
                &[barrier(
                    2,
                    ImageAccess::CopyDestination,
                    ImageAccess::CopySource,
                )],
            )?;
            encoder.copy_image_to_buffer(&texture, &readback, &[packed(0, 1), packed(16, 2)])?;
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
        // black and white image, whose mip averages them in linear space.
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
            })
        };
        let layered = create("layers", layers, 2)?;
        let black_and_white = create("black and white", pair, 1)?;
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
        let prepare = |image| ImageBarrier {
            image,
            range: ImageRange::ALL,
            before: ImageAccess::Undefined,
            after: ImageAccess::CopyDestination,
            transfer: None,
        };

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("mips")?;
        // SAFETY: the images are new.
        unsafe { encoder.barrier(&[], &[prepare(&layered), prepare(&black_and_white)])? };
        let base = |image: &Image| image.info().mip_region(0);
        encoder.copy_buffer_to_image(&staging, &layered, &[packed(0, base(&layered))])?;
        let pair_base = packed(pair_offset, base(&black_and_white));
        encoder.copy_buffer_to_image(&staging, &black_and_white, &[pair_base])?;
        assert_invalid(
            encoder.generate_mips(&layered, ImageAccess::Undefined, ImageAccess::CopySource),
            "cannot generate mips from Undefined",
        );
        for image in [&layered, &black_and_white] {
            encoder.generate_mips(image, ImageAccess::CopyDestination, ImageAccess::CopySource)?;
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
            display: None,
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
        frame.finish();
        drop(foreign);

        // A completion from another device, beyond anything this one submitted.
        let frame = other.begin_frame()?;
        let elsewhere = frame.submit(frame.encoder::<Graphics>("elsewhere")?)?;
        frame.finish();
        assert_invalid(rhi.wait(elsewhere, TIMEOUT), "not submitted");
        assert_invalid(rhi.is_complete(elsewhere), "not submitted");
        other.wait_idle()?;
        assert_eq!(other.validation_error_count(), 0);
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
        let frames = 3 * FRAMES_IN_FLIGHT;
        let mut values = PerFrame::try_new(rhi, |_| {
            buffer(rhi, 4, BufferUsage::COPY_SRC, Memory::Upload)
        })?;
        let readback = buffer(
            rhi,
            4 * frames as u64,
            BufferUsage::COPY_DST,
            Memory::Readback,
        )?;
        let mut slots = Vec::new();
        let mut last = None;
        for _ in 0..frames {
            let frame = rhi.begin_frame()?;
            let number = frame.number();
            slots.push(std::ptr::from_ref(values.get(&frame)));
            let value = u32::try_from(number).expect("few frames").to_le_bytes();
            values.write(&frame, 0, &value)?;
            let mut encoder = frame.encoder::<Graphics>("collect value")?;
            encoder.copy_buffer(
                values.get(&frame),
                &readback,
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 4 * (number - 1),
                    size: 4,
                }],
            )?;
            // SAFETY: orders the copy before host reads.
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
            last = Some(frame.submit(encoder)?);
        }
        rhi.wait(last.expect("frames were submitted"), TIMEOUT)?;

        for (index, slot) in slots.iter().enumerate() {
            assert_eq!(*slot, slots[index % FRAMES_IN_FLIGHT]);
        }
        assert_ne!(slots[0], slots[1]);
        let collected = read(&readback, 0, 4 * frames)?;
        let expected: Vec<u8> = (1..=u32::try_from(frames).expect("few frames"))
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(collected, expected);
        Ok(())
    })
}

#[test]
fn timestamps_are_read_once_their_frame_slot_comes_round() -> Result<()> {
    with_rhi(|rhi| {
        if !rhi.capabilities().features.contains(Features::TIMESTAMPS) {
            eprintln!("skipping: the device has no timestamps");
            return Ok(());
        }
        let mut timestamps = rhi.create_timestamps("frame", 3)?;
        let src = buffer(rhi, 1 << 20, BufferUsage::COPY_SRC, Memory::Device)?;
        let dst = buffer(rhi, 1 << 20, BufferUsage::COPY_DST, Memory::Device)?;
        let frame = rhi.begin_frame()?;
        assert_eq!(timestamps.begin(&frame)?, [None; 3]);
        let mut encoder = frame.encoder::<Graphics>("timed")?;
        encoder.write_timestamp(&mut timestamps, 0)?;
        encoder.copy_buffer(
            &src,
            &dst,
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 1 << 20,
            }],
        )?;
        encoder.write_timestamp(&mut timestamps, 1)?;
        frame.submit(encoder)?;
        frame.finish();
        // Later frames use the other slots.
        for _ in 1..FRAMES_IN_FLIGHT {
            let frame = rhi.begin_frame()?;
            assert_eq!(timestamps.begin(&frame)?, [None; 3]);
        }
        let frame = rhi.begin_frame()?;
        let [Some(start), Some(end), None] = *timestamps.begin(&frame)? else {
            panic!("the first frame wrote its first two timestamps only");
        };
        assert!(start <= end, "{start:?} is after {end:?}");
        // The slot's timestamps were cleared for this frame.
        frame.finish();
        for _ in 0..FRAMES_IN_FLIGHT {
            let frame = rhi.begin_frame()?;
            assert_eq!(timestamps.begin(&frame)?, [None; 3]);
        }
        Ok(())
    })
}

#[test]
fn timestamp_writes_are_validated() -> Result<()> {
    with_rhi(|rhi| {
        if !rhi.capabilities().features.contains(Features::TIMESTAMPS) {
            eprintln!("skipping: the device has no timestamps");
            return Ok(());
        }
        assert_invalid(rhi.create_timestamps("none", 0), "count is zero");
        let mut timestamps = rhi.create_timestamps("frame", 1)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("timed")?;
        assert_invalid(encoder.write_timestamp(&mut timestamps, 0), "call `begin`");
        timestamps.begin(&frame)?;
        assert_invalid(encoder.write_timestamp(&mut timestamps, 1), "outside");
        encoder.write_timestamp(&mut timestamps, 0)?;
        assert_invalid(
            encoder.write_timestamp(&mut timestamps, 0),
            "already written",
        );
        frame.submit(encoder)?;
        Ok(())
    })
}
