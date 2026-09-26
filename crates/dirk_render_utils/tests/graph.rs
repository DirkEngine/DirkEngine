//! Native graph checks; run explicitly with `cargo nextest run -p dirk_render_utils --run-ignored only`.
use dirk_render_utils::graph::{
    AttachmentInfo, ImportedTexture, RenderGraph, SubresourceRange, TextureWrite,
};
use dirk_rhi::{
    AccessTypes, BufferDesc, BufferUsages, DependencyInfo, Extent3d, Graphics, Image, ImageAspects,
    ImageDesc, ImageDimension, ImageState, ImageUsages, ImageView, ImageViewDesc, ImageViewType,
    MemoryBarrier, MemoryDomain, PipelineStages, Rhi, RhiCreateInfo, SampleCount, SubmitInfo,
    TextureFormat,
};

struct MipTarget {
    image: Image,
    view: ImageView,
}

impl MipTarget {
    fn new(rhi: &Rhi) -> dirk_rhi::Result<Self> {
        let image = rhi.create_image(&ImageDesc {
            label: "8x8 image with 4x4 mip1",
            dimension: ImageDimension::TwoD,
            extent: Extent3d::new_2d(8, 8),
            format: TextureFormat::Rgba8Unorm,
            usage: ImageUsages::COLOR_ATTACHMENT | ImageUsages::COPY_SRC,
            mip_levels: 2,
            array_layers: 1,
            samples: SampleCount::One,
        })?;
        let view = rhi.create_image_view(&ImageViewDesc {
            label: "mip1 attachment",
            image: &image,
            view_type: ImageViewType::TwoD,
            aspects: ImageAspects::COLOR,
            base_mip_level: 1,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        })?;
        Ok(Self { image, view })
    }

    fn clear(&self, range: SubresourceRange) -> anyhow::Result<RenderGraph<'_>> {
        let mut graph = RenderGraph::new();
        let target = graph.import_texture(ImportedTexture {
            image: &self.image,
            view: &self.view,
            initial_state: ImageState::Undefined,
            final_state: ImageState::CopySource,
        })?;
        graph.add_pass("clear mip1").write_range(
            target,
            TextureWrite::ColorAttachment {
                info: AttachmentInfo::clear_color(1.0, 0.0, 0.0, 1.0),
                resolve: None,
            },
            range,
        );
        Ok(graph)
    }
}

#[test]
#[ignore = "requires a Vulkan or Metal device; exercised explicitly on native validation hosts"]
fn nonzero_attachment_mip_validates_ranges_and_clears_its_full_extent() -> anyhow::Result<()> {
    let mut rhi = Rhi::new(&RhiCreateInfo {
        engine_name: "DirkEngine",
        engine_version: (0, 1, 0),
        application_name: "graph attachment mip validation",
        application_version: (0, 1, 0),
        validation: true,
        compatible_surface: None,
    })?;
    let target = MipTarget::new(&rhi)?;
    for (range, message) in [
        (SubresourceRange::WHOLE, "one mip and one layer"),
        (
            SubresourceRange::WHOLE.with_mips(0, 1),
            "view must match its declared subresource range",
        ),
    ] {
        let mut commands = rhi.create_encoder::<Graphics>("invalid attachment declaration")?;
        // SAFETY: graph validation rejects these declarations before recording any work.
        let error = unsafe { target.clear(range)?.run(&rhi, &mut commands) }
            .expect_err("mismatched attachment range");
        assert!(format!("{error:#}").contains(message));
    }

    let full_view = rhi.view(&target.image)?;
    let mut graph = target.clear(SubresourceRange::WHOLE.with_mips(1, 1))?;
    assert!(
        graph
            .import_texture(ImportedTexture {
                image: &target.image,
                view: &full_view,
                initial_state: ImageState::Undefined,
                final_state: ImageState::CopySource,
            })
            .is_err()
    );

    let layout = dirk_rhi::UploadLayout::new(4, 4, TextureFormat::Rgba8Unorm, rhi.capabilities())?;
    let readback = rhi.create_buffer(&BufferDesc {
        label: "mip1 clear readback",
        size: u64::from(layout.bytes_per_row.get()) * 4,
        usage: BufferUsages::COPY_DST,
        memory: MemoryDomain::Readback,
    })?;
    let mut commands = rhi.create_encoder::<Graphics>("clear and read back mip1")?;
    // SAFETY: fresh image, exported copy state, resources retained until completion.
    unsafe {
        graph.run(&rhi, &mut commands)?;
        commands.copy_image_to_buffer(
            &target.image,
            &readback,
            &[dirk_rhi::BufferImageCopy {
                buffer_offset: 0,
                buffer_bytes_per_row: layout.bytes_per_row,
                buffer_rows_per_image: layout.rows,
                mip_level: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                image_origin: dirk_rhi::Origin3d::default(),
                extent: Extent3d::new_2d(4, 4),
                aspects: ImageAspects::COLOR,
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
        rhi.queue::<Graphics>()
            .submit(vec![commands.finish()?], &SubmitInfo::default())?;
    }
    rhi.flush()?;
    let mut pixels = vec![0; layout.bytes_per_row.get() as usize * 4];
    // SAFETY: flush waited for all GPU writes, with an explicit host-read dependency.
    unsafe {
        readback.read(0, &mut pixels)?;
    }
    for row in pixels.chunks_exact(layout.bytes_per_row.get() as usize) {
        assert_eq!(&row[..16], [255, 0, 0, 255].repeat(4));
    }
    drop((full_view, target, readback));
    rhi.flush()?;
    assert_eq!(rhi.validation_error_count(), 0, "native validation errors");
    Ok(())
}
