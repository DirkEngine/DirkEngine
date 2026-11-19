//! Staged uploads, read back by graphics work under validation.

use std::time::Duration;

use dirk_render_utils::Uploads;
use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage,
    Completion, Error, Extent3d, FRAMES_IN_FLIGHT, Format, Frame, Graphics, Image, ImageDesc,
    ImageDimension, ImageRegion, ImageUsage, Memory, Origin3d, Result, Rhi, RhiCreateInfo,
    SampleCount, ShaderStages,
};
use dirk_rhi_test_shaders::support::with_rhi;

const TIMEOUT: Duration = Duration::from_secs(10);

/// Host-readable copies of uploaded resources, made by graphics work.
struct Readback {
    done: Completion,
    buffers: Vec<Buffer>,
}

impl Readback {
    /// Copies each buffer, accessed as given, into host memory in `frame`.
    fn buffers(frame: &Frame<'_>, sources: &[(&Buffer, BufferAccess)]) -> Result<Self> {
        let buffers = sources
            .iter()
            .map(|(buffer, _)| Self::create(frame.rhi(), buffer.info().size))
            .collect::<Result<Vec<_>>>()?;
        let mut encoder = frame.encoder::<Graphics>("buffer readback")?;
        let to_copies: Vec<_> = sources
            .iter()
            .map(|&(buffer, access)| BufferBarrier {
                buffer,
                before: access,
                after: BufferAccess::CopySource,
                transfer: None,
            })
            .collect();
        // SAFETY: the uploads left each buffer in its given access.
        unsafe { encoder.barrier(&to_copies, &[])? };
        for ((source, _), readback) in sources.iter().zip(&buffers) {
            let copy = BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: source.info().size,
            };
            encoder.copy_buffer(source, readback, &[copy])?;
        }
        Self::submit(frame, encoder, buffers)
    }

    /// Copies every mip of `image` into host memory in `frame`, tightly
    /// packed.
    fn image(frame: &Frame<'_>, image: &Image) -> Result<Self> {
        let info = *image.info();
        let regions: Vec<_> = (0..info.mip_levels)
            .map(|level| info.mip_region(level))
            .collect();
        let buffers = regions
            .iter()
            .map(|region| {
                let copy = BufferImageCopy::packed(info.format, 0, *region);
                let size = u64::from(copy.bytes_per_row)
                    * u64::from(copy.rows_per_image)
                    * u64::from(region.layer_count);
                Self::create(frame.rhi(), size)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut encoder = frame.encoder::<Graphics>("image readback")?;
        for (region, readback) in regions.into_iter().zip(&buffers) {
            let copy = BufferImageCopy::packed(info.format, 0, region);
            encoder.copy_image_to_buffer(image, readback, &[copy])?;
        }
        Self::submit(frame, encoder, buffers)
    }

    /// Waits for the copies and returns each buffer's bytes.
    fn wait(self, rhi: &Rhi) -> Result<Vec<Vec<u8>>> {
        rhi.wait(self.done, TIMEOUT)?;
        self.buffers
            .iter()
            .map(|buffer| {
                let mut data = vec![0; usize::try_from(buffer.info().size).unwrap_or(0)];
                // SAFETY: the copies completed, ending with a barrier to host
                // reads.
                unsafe { buffer.read(0, &mut data)? };
                Ok(data)
            })
            .collect()
    }

    fn create(rhi: &Rhi, size: u64) -> Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "readback",
            size,
            usage: BufferUsage::COPY_DST,
            memory: Memory::Readback,
        })
    }

    /// Makes the copies into `buffers` visible to the host and submits them.
    fn submit(
        frame: &Frame<'_>,
        mut encoder: dirk_rhi::CommandEncoder<'_, Graphics>,
        buffers: Vec<Buffer>,
    ) -> Result<Self> {
        let to_host: Vec<_> = buffers
            .iter()
            .map(|buffer| BufferBarrier {
                buffer,
                before: BufferAccess::CopyDestination,
                after: BufferAccess::HostRead,
                transfer: None,
            })
            .collect();
        // SAFETY: the readback buffers were only copied into.
        unsafe { encoder.barrier(&to_host, &[])? };
        Ok(Self {
            done: frame.submit(encoder)?,
            buffers,
        })
    }
}

/// `len` bytes distinct for each `seed`.
fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|index| u8::try_from((index * 7 + seed * 31) % 251).unwrap_or(0))
        .collect()
}

/// Uploads `data` as copy sources in `frame` and starts reading them back.
fn upload_buffers(frame: &mut Frame<'_>, data: &[Vec<u8>]) -> Result<Readback> {
    let mut uploads = Uploads::new(frame);
    let buffers = data
        .iter()
        .map(|data| {
            uploads.buffer(
                "upload",
                BufferUsage::COPY_SRC,
                BufferAccess::CopySource,
                data,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    uploads.submit()?;
    let sources: Vec<_> = buffers
        .iter()
        .map(|buffer| (buffer, BufferAccess::CopySource))
        .collect();
    Readback::buffers(frame, &sources)
}

/// Asserts that each readback starts with its data, padded to four bytes.
fn assert_read(read: &[Vec<u8>], data: &[Vec<u8>]) {
    for (read, data) in read.iter().zip(data) {
        assert_eq!(read.len(), data.len().next_multiple_of(4));
        assert_eq!(&read[..data.len()], data);
    }
}

#[test]
fn buffers_of_any_length_round_trip() -> Result<()> {
    with_rhi(|rhi| {
        let indices: Vec<u8> = [1_u16, 2, 3].iter().flat_map(|i| i.to_ne_bytes()).collect();
        let bytes = [pattern(1, 1), pattern(13, 2), pattern(1001, 3)];

        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        let index_buffer = uploads.buffer(
            "indices",
            BufferUsage::INDEX | BufferUsage::COPY_SRC,
            BufferAccess::Index,
            &indices,
        )?;
        let buffers = bytes
            .iter()
            .map(|data| {
                uploads.buffer(
                    "bytes",
                    BufferUsage::COPY_SRC,
                    BufferAccess::CopySource,
                    data,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Staged as given, before rounding up to whole words.
        let staged = indices.len() + bytes.iter().map(Vec::len).sum::<usize>();
        assert_eq!(uploads.staged(), staged as u64);
        uploads.submit()?;
        let mut sources = vec![(&index_buffer, BufferAccess::Index)];
        sources.extend(
            buffers
                .iter()
                .map(|buffer| (buffer, BufferAccess::CopySource)),
        );
        let readback = Readback::buffers(&frame, &sources)?;
        frame.finish();

        let mut expected = vec![indices];
        expected.extend(bytes);
        assert_read(&readback.wait(rhi)?, &expected);
        Ok(())
    })
}

#[test]
fn layered_and_mipped_images_round_trip() -> Result<()> {
    with_rhi(|rhi| {
        // Three layers of three mips, then odd two-byte texel rows whose
        // smallest mip is not a multiple of four bytes long.
        let images = [
            (Format::Rgba8Unorm, Extent3d::new_2d(8, 4), 3, 3),
            (Format::R16Float, Extent3d::new_2d(5, 3), 2, 1),
        ];

        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        let mut sources = Vec::new();
        for (index, (format, extent, mip_levels, array_layers)) in images.into_iter().enumerate() {
            let desc = ImageDesc {
                label: "uploaded image",
                dimension: ImageDimension::D2,
                extent,
                format,
                usage: ImageUsage::SAMPLED | ImageUsage::COPY_SRC,
                mip_levels,
                array_layers,
                samples: SampleCount::X1,
                rest: None,
            };
            let mips: Vec<_> = (0..mip_levels)
                .map(|level| {
                    let extent = extent.mip(level);
                    let len =
                        extent.width * extent.height * array_layers * format.bytes_per_texel();
                    pattern(len as usize, index * 8 + level as usize)
                })
                .collect();
            let slices: Vec<_> = mips.iter().map(Vec::as_slice).collect();
            sources.push((uploads.image(&desc, &slices)?, mips));
        }
        uploads.submit()?;
        let readbacks = sources
            .iter()
            .map(|(image, _)| Readback::image(&frame, image))
            .collect::<Result<Vec<_>>>()?;
        frame.finish();

        for (readback, (_, mips)) in readbacks.into_iter().zip(&sources) {
            assert_eq!(&readback.wait(rhi)?, mips);
        }
        Ok(())
    })
}

#[test]
fn mipmapped_images_generate_every_mip_from_the_first() -> Result<()> {
    with_rhi(|rhi| {
        // A uniform color keeps its value in every mip of any size.
        let color = [200, 100, 50, 255];
        let desc = ImageDesc {
            label: "mipmapped",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(5, 3),
            format: Format::Rgba8Srgb,
            usage: ImageUsage::SAMPLED,
            mip_levels: 3,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };

        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        let image = uploads.mipmapped_image(&desc, &color.repeat(15))?;
        uploads.submit()?;
        let readback = Readback::image(&frame, &image)?;
        frame.finish();

        let expected = [15, 2, 1].map(|texels| color.repeat(texels));
        assert_eq!(readback.wait(rhi)?, expected);
        Ok(())
    })
}

#[test]
fn uploads_stage_in_frame_memory_without_overwriting_frames_in_flight() -> Result<()> {
    with_rhi(|rhi| {
        let frame_memory = usize::try_from(RhiCreateInfo::default().frame_memory).expect("fits");
        let mut pending = Vec::new();
        // Frames overlap on the GPU: none waits for its readback, so a slot's
        // staged data must survive until its copies ran.
        for number in 0..3 * FRAMES_IN_FLIGHT {
            let mut data: Vec<_> = (0..3)
                .map(|index| pattern(40 + 24 * index + number, 3 * number + index))
                .collect();
            // One upload does not fit the frame's memory.
            if number == 1 {
                data.push(pattern(frame_memory + 1, number));
            }
            let mut frame = rhi.begin_frame()?;
            pending.push((upload_buffers(&mut frame, &data)?, data));
        }
        for (readback, data) in pending {
            assert_read(&readback.wait(rhi)?, &data);
        }
        Ok(())
    })
}

#[test]
fn the_copy_queue_hands_uploads_to_graphics_work() -> Result<()> {
    with_rhi(|rhi| {
        if !rhi.capabilities().dedicated_copy_queue {
            eprintln!("skipping: the device has no dedicated copy queue");
            return Ok(());
        }
        let read = BufferAccess::ShaderRead(ShaderStages::VERTEX | ShaderStages::FRAGMENT);
        let vertices = pattern(96, 1);
        let texels = pattern(4 * 4 * 4, 2);
        let desc = ImageDesc {
            label: "texture",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(4, 4),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED | ImageUsage::COPY_SRC,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };

        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        let usage = BufferUsage::STORAGE | BufferUsage::COPY_SRC;
        let mesh = uploads.buffer("vertices", usage, read, &vertices)?;
        let texture = uploads.image(&desc, &[&texels])?;
        let done = uploads.submit()?;
        assert_eq!(
            done.map(Completion::queue),
            Some(dirk_rhi::QueueType::Graphics)
        );
        // Graphics work submitted after the uploads uses them without waiting.
        let buffer = Readback::buffers(&frame, &[(&mesh, read)])?;
        let image = Readback::image(&frame, &texture)?;
        frame.finish();

        assert_eq!(buffer.wait(rhi)?, [vertices]);
        assert_eq!(image.wait(rhi)?, [texels]);
        Ok(())
    })
}

#[test]
fn image_writes_replace_regions_of_new_and_sampled_images() -> Result<()> {
    with_rhi(|rhi| {
        let desc = ImageDesc {
            label: "written",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(4, 4),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED | ImageUsage::COPY_SRC,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };
        let region = |x, y, width, height| ImageRegion {
            mip_level: 0,
            base_layer: 0,
            layer_count: 1,
            origin: Origin3d { x, y, z: 0 },
            extent: Extent3d::new_2d(width, height),
        };
        let mut expected = pattern(64, 1);
        // Writes `width` by `height` texels of `seed`'s pattern at `x`, `y`
        // into `expected`, returning them.
        let mut patch = |x: usize, y: usize, width: usize, height: usize, seed| {
            let texels = pattern(width * height * 4, seed);
            for (row, source) in texels.chunks_exact(width * 4).enumerate() {
                let start = ((y + row) * 4 + x) * 4;
                expected[start..start + width * 4].copy_from_slice(source);
            }
            texels
        };
        let writes = [
            (region(0, 0, 2, 2), patch(0, 0, 2, 2, 2)),
            (region(1, 1, 3, 2), patch(1, 1, 3, 2, 3)),
            (region(2, 0, 1, 4), patch(2, 0, 1, 4, 4)),
        ];

        // Written in the batch creating it, then twice in a later frame.
        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        let image = uploads.image(&desc, &[&pattern(64, 1)])?;
        uploads.write_image(&image, writes[0].0, &writes[0].1)?;
        assert_invalid(
            uploads.write_image(&image, writes[1].0, &writes[1].1[1..]),
            "the region holds 23 bytes instead of 24",
        );
        uploads.submit()?;
        frame.finish();
        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        for (region, texels) in &writes[1..] {
            uploads.write_image(&image, *region, texels)?;
        }
        uploads.submit()?;
        let readback = Readback::image(&frame, &image)?;
        frame.finish();

        assert_eq!(readback.wait(rhi)?, [expected]);
        Ok(())
    })
}

#[test]
fn image_uploads_need_every_texel_of_each_mip_given() -> Result<()> {
    with_rhi(|rhi| {
        let desc = ImageDesc {
            label: "texture",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(4, 4),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED,
            mip_levels: 2,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };
        let (mip0, mip1) = ([0; 64], [0; 16]);

        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        assert_invalid(uploads.image(&desc, &[]), "0 mips given");
        assert_invalid(
            uploads.image(&desc, &[&mip0, &mip1, &mip1]),
            "3 mips given for an image of 2",
        );
        assert_invalid(
            uploads.image(&desc, &[&mip0[1..]]),
            "mip 0 holds 63 bytes instead of 64",
        );
        assert_invalid(
            uploads.image(&desc, &[&mip0, &mip0]),
            "mip 1 holds 64 bytes instead of 16",
        );
        assert!(
            uploads.submit()?.is_none(),
            "rejected uploads record nothing"
        );
        Ok(())
    })
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
