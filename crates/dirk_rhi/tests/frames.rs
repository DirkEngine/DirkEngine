//! Frames, queues, completions, copies and deferred destruction.

mod support;

use std::time::Duration;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    CopyQueue, Error, Extent3d, FRAMES_IN_FLIGHT, Filter, Format, Graphics, Image, ImageAccess,
    ImageBarrier, ImageCopy, ImageDesc, ImageDimension, ImageRange, ImageUsage, Memory, PerFrame,
    QueueTransfer, QueueType, Result, Rhi, RhiCreateInfo, SampleCount, ShaderStages,
};
use support::with_rhi;

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
