//! Native upload checks; run explicitly with `cargo nextest run -p dirk_render_utils --run-ignored only`.
use dirk_render_utils::upload::UploadBatch;
use dirk_rhi::{
    AccessTypes, BufferDesc, BufferImageCopy, BufferUsages, DependencyInfo, Extent3d, Graphics,
    ImageAspects, ImageBarrier, ImageDesc, ImageDimension, ImageState, ImageUsages, MemoryBarrier,
    MemoryDomain, Origin3d, PipelineStages, Rhi, RhiCreateInfo, SampleCount, ShaderStages,
    SubmitInfo, TextureFormat, UploadLayout,
};

const LAYERS: u32 = 2;

/// Distinct opaque texel for each mip and layer.
fn texel(mip: u32, layer: u32) -> [u8; 4] {
    let tag = u8::try_from(mip * LAYERS + layer).unwrap_or(u8::MAX);
    [tag, 100 + tag, 200 - tag, 255]
}

/// Copies both mips of every layer to host memory, returning padded rows per mip.
fn read_back(
    rhi: &mut Rhi,
    image: &dirk_rhi::Image,
    stages: ShaderStages,
) -> anyhow::Result<Vec<(u32, UploadLayout, Vec<u8>)>> {
    let mut readbacks = Vec::new();
    let mut commands = rhi.create_encoder::<Graphics>("read back uploaded layers")?;
    // SAFETY: the upload finished; readback resources outlive the flush.
    unsafe {
        commands.barrier(&DependencyInfo {
            memory_barriers: &[],
            buffer_barriers: &[],
            image_barriers: &[ImageBarrier {
                image,
                old_state: ImageState::ShaderRead(stages),
                new_state: ImageState::CopySource,
                aspects: ImageAspects::COLOR,
                base_mip_level: 0,
                mip_level_count: 2,
                base_array_layer: 0,
                array_layer_count: LAYERS,
                queue_transfer: None,
            }],
        })?;
        for (mip, size) in [(0, 4), (1, 2)] {
            let layout =
                UploadLayout::new(size, size, TextureFormat::Rgba8Unorm, rhi.capabilities())?;
            let layer_bytes = u64::from(layout.bytes_per_row.get()) * u64::from(size);
            let buffer = rhi.create_buffer(&BufferDesc {
                label: "uploaded layer readback",
                size: layer_bytes * u64::from(LAYERS),
                usage: BufferUsages::COPY_DST,
                memory: MemoryDomain::Readback,
            })?;
            commands.copy_image_to_buffer(
                image,
                &buffer,
                &[BufferImageCopy {
                    buffer_offset: 0,
                    buffer_bytes_per_row: layout.bytes_per_row,
                    buffer_rows_per_image: layout.rows,
                    mip_level: mip,
                    base_array_layer: 0,
                    array_layer_count: LAYERS,
                    image_origin: Origin3d::default(),
                    extent: Extent3d::new_2d(size, size),
                    aspects: ImageAspects::COLOR,
                }],
            )?;
            readbacks.push((mip, layout, buffer));
        }
        commands.barrier(&DependencyInfo {
            memory_barriers: &[MemoryBarrier {
                src_stages: PipelineStages::COPY,
                src_access: AccessTypes::COPY_WRITE,
                dst_stages: PipelineStages::HOST,
                dst_access: AccessTypes::HOST_READ,
            }],
            buffer_barriers: &[],
            image_barriers: &[],
        })?;
        rhi.queue::<Graphics>()
            .submit(vec![commands.finish()?], &SubmitInfo::default())?;
    }
    rhi.flush()?;
    let mut host = Vec::new();
    for (mip, layout, buffer) in readbacks {
        let mut bytes = vec![0; usize::try_from(buffer.size())?];
        // SAFETY: flush waited for all GPU writes, with an explicit host-read dependency.
        unsafe {
            buffer.read(0, &mut bytes)?;
        }
        host.push((mip, layout, bytes));
    }
    Ok(host)
}

#[test]
#[ignore = "requires a Vulkan or Metal device; exercised explicitly on native validation hosts"]
fn image_uploads_fill_every_layer_for_the_requested_stages() -> anyhow::Result<()> {
    let mut rhi = Rhi::new(&RhiCreateInfo {
        engine_name: "DirkEngine",
        engine_version: (0, 1, 0),
        application_name: "layered upload validation",
        application_version: (0, 1, 0),
        validation: true,
        compatible_surface: None,
    })?;
    let image = rhi.create_image(&ImageDesc {
        label: "4x4 two-layer image with 2x2 mip1",
        dimension: ImageDimension::TwoD,
        extent: Extent3d::new_2d(4, 4),
        format: TextureFormat::Rgba8Unorm,
        usage: ImageUsages::COPY_DST | ImageUsages::COPY_SRC | ImageUsages::SAMPLED,
        mip_levels: 2,
        array_layers: LAYERS,
        samples: SampleCount::One,
    })?;
    let levels = [(0, 16), (1, 4)].map(|(mip, texels)| {
        (0..LAYERS)
            .flat_map(|layer| texel(mip, layer).repeat(texels))
            .collect::<Vec<_>>()
    });
    let stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    let mut uploads = UploadBatch::new();
    // SAFETY: the image is new, unused, and retained until the final flush.
    unsafe {
        uploads.image_for_stages(&rhi, &image, &[&levels[0], &levels[1]], stages)?;
    }
    uploads.finish(&mut rhi)?;

    for (mip, layout, bytes) in read_back(&mut rhi, &image, stages)? {
        let rows = bytes.chunks_exact(layout.bytes_per_row.get() as usize);
        for (row, bytes) in rows.enumerate() {
            let layer = u32::try_from(row)? / layout.rows.get();
            let texels = &bytes[..layout.row_bytes as usize];
            assert_eq!(texels, texel(mip, layer).repeat(texels.len() / 4));
        }
    }
    drop(image);
    rhi.flush()?;
    assert_eq!(rhi.validation_error_count(), 0, "native validation errors");
    Ok(())
}
