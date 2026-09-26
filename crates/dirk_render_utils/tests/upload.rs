//! Explicit native checks; run with `cargo nextest run -p dirk_render_utils --run-ignored only`.
use dirk_render_utils::upload::UploadBatch;
use dirk_rhi::{
    AccessTypes, BufferDesc, BufferUsages, DependencyInfo, Graphics, MemoryBarrier, MemoryDomain,
    PipelineStages, ResourceAccess, Rhi, RhiCreateInfo, SubmitInfo,
};

#[test]
#[ignore = "requires a Vulkan or Metal device; exercised explicitly on native validation hosts"]
fn batched_upload_acquires_on_graphics_and_survives_retirement() -> dirk_rhi::Result<()> {
    let mut rhi = Rhi::new(&RhiCreateInfo {
        engine_name: "DirkEngine",
        engine_version: (0, 1, 0),
        application_name: "RHI upload validation",
        application_version: (0, 1, 0),
        validation: true,
        compatible_surface: None,
    })?;
    let expected = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut uploads = UploadBatch::new();
    let gpu = uploads.buffer(
        &rhi,
        &expected,
        BufferUsages::COPY_SRC,
        ResourceAccess::CopySource,
    )?;
    let readback = rhi.create_buffer(&BufferDesc {
        label: "readback",
        size: 16,
        usage: BufferUsages::COPY_DST,
        memory: MemoryDomain::Readback,
    })?;
    let mut commands = rhi.create_encoder::<Graphics>("read back uploaded bytes")?;
    // SAFETY: fresh output, acquired source; host access happens after submission completion.
    unsafe {
        commands.copy_buffer(
            &gpu,
            &readback,
            &[dirk_rhi::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 16,
            }],
        )?;
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
    }
    let (acquire, transfer) = uploads.submit(&mut rhi)?.expect("nonempty upload");
    // Dropped owners, including staging inside UploadBatch::buffer, are kept by the retirement cycle.
    drop(gpu);
    let commands = commands.finish()?;
    let done = unsafe {
        rhi.queue::<Graphics>().submit(
            vec![acquire, commands],
            &SubmitInfo {
                surface_frames: &[],
                wait_for: &[&transfer],
            },
        )?
    };
    rhi.finish_cycle()?;
    done.wait(u64::MAX)?;
    let mut actual = [0; 16];
    unsafe {
        readback.read(0, &mut actual)?;
    }
    assert_eq!(actual, expected);
    drop((done, transfer, readback));
    rhi.finish_cycle()?;
    rhi.finish_cycle()?;
    rhi.wait_idle()?;
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan or Metal device; exercised explicitly on native validation hosts"]
fn clear_only_graph_produces_pixels_and_idle_preserves_active_retirements() -> anyhow::Result<()> {
    use dirk_render_utils::graph::{AttachmentInfo, ImportedTexture, RenderGraph};
    use dirk_rhi::{
        Extent3d, ImageDesc, ImageDimension, ImageState, ImageUsages, SampleCount, TextureFormat,
    };
    let mut rhi = Rhi::new(&RhiCreateInfo {
        engine_name: "DirkEngine",
        engine_version: (0, 1, 0),
        application_name: "graph clear validation",
        application_version: (0, 1, 0),
        validation: true,
        compatible_surface: None,
    })?;
    let image = rhi.create_image(&ImageDesc {
        label: "offscreen target",
        dimension: ImageDimension::TwoD,
        extent: Extent3d::new_2d(4, 4),
        format: TextureFormat::Rgba8Unorm,
        usage: ImageUsages::COLOR_ATTACHMENT | ImageUsages::COPY_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::One,
    })?;
    let view = rhi.view(&image)?;
    let layout = dirk_rhi::UploadLayout::new(4, 4, TextureFormat::Rgba8Unorm, rhi.capabilities())?;
    let readback = rhi.create_buffer(&BufferDesc {
        label: "clear readback",
        size: u64::from(layout.bytes_per_row.get()) * 4,
        usage: BufferUsages::COPY_DST,
        memory: MemoryDomain::Readback,
    })?;
    let mut graph = RenderGraph::new();
    let target = graph.import_texture(ImportedTexture {
        image: &image,
        view: &view,
        initial_state: ImageState::Undefined,
        final_state: ImageState::CopySource,
    })?;
    graph
        .add_pass("clear without draws")
        .write_color_attachment(target, AttachmentInfo::clear_color(1.0, 0.0, 0.0, 1.0));
    let mut commands = rhi.create_encoder::<Graphics>("clear and readback")?;
    // SAFETY: fresh image, graph-exported copy state, host read only after completion.
    unsafe {
        graph.run(&rhi, &mut commands)?;
        commands.copy_image_to_buffer(
            &image,
            &readback,
            &[dirk_rhi::BufferImageCopy {
                buffer_offset: 0,
                buffer_bytes_per_row: layout.bytes_per_row,
                buffer_rows_per_image: layout.rows,
                mip_level: 0,
                base_array_layer: 0,
                array_layer_count: 1,
                image_origin: dirk_rhi::Origin3d::default(),
                extent: Extent3d::new_2d(4, 4),
                aspects: dirk_rhi::ImageAspects::COLOR,
            }],
        )?;
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
    }
    drop((view, image));
    // Window resize takes this same idle path while current-cycle recordings may still exist.
    rhi.wait_idle()?;
    unsafe {
        rhi.queue::<Graphics>()
            .submit(vec![commands.finish()?], &SubmitInfo::default())?;
    }
    rhi.flush()?;
    let mut pixels = vec![0; layout.bytes_per_row.get() as usize * 4];
    unsafe {
        readback.read(0, &mut pixels)?;
    }
    for row in pixels.chunks_exact(layout.bytes_per_row.get() as usize) {
        assert_eq!(&row[..16], [255, 0, 0, 255].repeat(4));
    }
    Ok(())
}
