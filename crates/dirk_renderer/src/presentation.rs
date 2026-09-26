//! Samples the linear scene output and encodes it once for the window's format.
use dirk_rhi::{
    AddressMode, BindGroupDesc, BindGroupEntry, BindGroupLayout, BindGroupLayoutDesc,
    BindingResource, ColorTargetState, ColorWrites, CullMode, Extent3d, FilterMode,
    GraphicsPipeline, GraphicsPipelineDesc, PipelineLayout, PipelineLayoutDesc, RasterState, Rhi,
    SampleCount, Sampler, SamplerDesc, ShaderStages, TextureFormat,
};

use crate::{
    Result,
    frame_graph::{AttachmentInfo, RenderGraph, TextureHandle},
    shaders::{PresentFS, PresentUnormFS, PresentVS, metadata::Shader},
};

pub(crate) struct Presenter {
    format: TextureFormat,
    pipeline: GraphicsPipeline,
    layout: PipelineLayout,
    bindings: BindGroupLayout,
    sampler: Sampler,
}

impl Presenter {
    pub(crate) fn new(rhi: &Rhi, format: TextureFormat) -> Result<Self> {
        let bindings = rhi.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "presentation texture",
            entries: PresentFS::SET_LAYOUTS[0],
        })?;
        let layout = rhi.create_pipeline_layout(&PipelineLayoutDesc {
            label: "presentation",
            bind_group_layouts: &[&bindings],
        })?;
        let vertex = PresentVS::create(rhi)?;
        let fragment = if matches!(format, TextureFormat::Rgba8Srgb | TextureFormat::Bgra8Srgb) {
            PresentFS::create(rhi)?
        } else {
            PresentUnormFS::create(rhi)?
        };
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: "presentation",
            layout: &layout,
            vertex: &vertex,
            fragment: Some(&fragment),
            vertex_buffers: &[],
            raster: RasterState {
                cull_mode: CullMode::None,
                topology: dirk_rhi::PrimitiveTopology::TriangleList,
                front_face: dirk_rhi::FrontFace::CounterClockwise,
            },
            color_targets: &[ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::RED
                    | ColorWrites::GREEN
                    | ColorWrites::BLUE
                    | ColorWrites::ALPHA,
            }],
            depth: None,
            depth_bias: dirk_rhi::DepthBiasState::default(),
            primitive_restart: None,
            alpha_to_coverage: false,
            samples: SampleCount::One,
        })?;
        let sampler = rhi.create_sampler(&SamplerDesc {
            label: "presentation",
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            max_anisotropy: 1,
            lod_min: 0.0,
            lod_max: 0.0,
        })?;
        Ok(Self {
            format,
            pipeline,
            layout,
            bindings,
            sampler,
        })
    }

    pub(crate) fn set_target_format(&mut self, rhi: &Rhi, format: TextureFormat) -> Result<()> {
        if self.format != format {
            *self = Self::new(rhi, format)?;
        }
        Ok(())
    }

    pub(crate) fn add_pass<'a>(
        &'a self,
        rhi: &'a Rhi,
        graph: &mut RenderGraph<'a>,
        source: TextureHandle,
        target: TextureHandle,
        extent: Extent3d,
    ) {
        graph
            .add_pass("encode scene for presentation")
            .read_sampled(source, ShaderStages::FRAGMENT)
            .write_color_attachment(target, AttachmentInfo::load_store())
            .execute(Box::new(move |cmd, ctx| {
                let source = ctx.resolve(source)?;
                let binding = rhi.create_bind_group(&BindGroupDesc {
                    label: "presentation texture",
                    layout: &self.bindings,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::SampledImage {
                            view: &source.view,
                            sampler: &self.sampler,
                        },
                    }],
                })?;
                // Texture extents are bounded by the backend, well within exact f32 integers.
                #[allow(clippy::cast_precision_loss)]
                cmd.set_viewport(dirk_rhi::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                })?;
                cmd.set_scissor(dirk_rhi::Rect {
                    x: 0,
                    y: 0,
                    width: extent.width,
                    height: extent.height,
                })?;
                // SAFETY: the graph declares the sampled texture and attachment. The
                // window owns the sampler/pipeline; RHI retirement retains this binding.
                unsafe {
                    cmd.bind_graphics_pipeline(&self.pipeline)?;
                    cmd.bind_groups(&self.layout, 0, &[&binding], &[])?;
                    cmd.draw(3, 1, 0, 0)?;
                }
                Ok(())
            }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::ImportedTexture;
    use dirk_rhi::{
        AccessTypes, BufferDesc, BufferUsages, DependencyInfo, Graphics, ImageDesc, ImageDimension,
        ImageState, ImageUsages, MemoryBarrier, MemoryDomain, PipelineStages, RhiCreateInfo,
        SubmitInfo,
    };

    #[test]
    #[ignore = "requires a native Vulkan or Metal device"]
    fn presentation_encodes_midtones_once_for_srgb_and_unorm_targets() -> anyhow::Result<()> {
        let mut rhi = Rhi::new(&RhiCreateInfo {
            engine_name: "DirkEngine",
            engine_version: (0, 1, 0),
            application_name: "presentation color validation",
            application_version: (0, 1, 0),
            validation: true,
            compatible_surface: None,
        })?;
        let extent = Extent3d::new_2d(4, 4);
        let mut presenter = Presenter::new(&rhi, TextureFormat::Rgba8Unorm)?;
        for source_format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Srgb] {
            for target_format in [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Srgb] {
                let desc = |format| ImageDesc {
                    label: "presentation color test",
                    dimension: ImageDimension::TwoD,
                    extent,
                    format,
                    usage: ImageUsages::COLOR_ATTACHMENT
                        | ImageUsages::SAMPLED
                        | ImageUsages::COPY_SRC,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: SampleCount::One,
                };
                let source = rhi.create_image(&desc(source_format))?;
                let target = rhi.create_image(&desc(target_format))?;
                let source_view = rhi.view(&source)?;
                let target_view = rhi.view(&target)?;
                presenter.set_target_format(&rhi, target_format)?;
                let layout = dirk_rhi::UploadLayout::new(4, 4, target_format, rhi.capabilities())?;
                let readback = rhi.create_buffer(&BufferDesc {
                    label: "encoded midtone",
                    size: u64::from(layout.bytes_per_row.get()) * 4,
                    usage: BufferUsages::COPY_DST,
                    memory: MemoryDomain::Readback,
                })?;
                let mut graph = RenderGraph::new();
                let src = graph.import_texture(ImportedTexture {
                    image: &source,
                    view: &source_view,
                    initial_state: ImageState::Undefined,
                    final_state: ImageState::ShaderRead(ShaderStages::FRAGMENT),
                })?;
                let dst = graph.import_texture(ImportedTexture {
                    image: &target,
                    view: &target_view,
                    initial_state: ImageState::Undefined,
                    final_state: ImageState::CopySource,
                })?;
                graph.add_pass("linear midtone").write_color_attachment(
                    src,
                    AttachmentInfo::clear_color(0.18, 0.18, 0.18, 1.0),
                );
                graph
                    .add_pass("clear target")
                    .write_color_attachment(dst, AttachmentInfo::clear_color(0.0, 0.0, 0.0, 1.0));
                presenter.add_pass(&rhi, &mut graph, src, dst, extent);
                let mut cmd = rhi.create_encoder::<Graphics>("midtone readback")?;
                // SAFETY: fresh images; graph establishes source/target dependencies;
                // CPU readback happens only after submission completion.
                unsafe {
                    graph.run(&rhi, &mut cmd)?;
                    cmd.copy_image_to_buffer(
                        &target,
                        &readback,
                        &[dirk_rhi::BufferImageCopy {
                            buffer_offset: 0,
                            buffer_bytes_per_row: layout.bytes_per_row,
                            buffer_rows_per_image: layout.rows,
                            mip_level: 0,
                            base_array_layer: 0,
                            array_layer_count: 1,
                            image_origin: dirk_rhi::Origin3d::default(),
                            extent,
                            aspects: dirk_rhi::ImageAspects::COLOR,
                        }],
                    )?;
                    cmd.barrier(&DependencyInfo {
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
                        .submit(vec![cmd.finish()?], &SubmitInfo::default())?
                        .wait(u64::MAX)?;
                }
                let mut pixels = vec![0; layout.bytes_per_row.get() as usize * 4];
                // SAFETY: GPU completion was awaited above.
                unsafe {
                    readback.read(0, &mut pixels)?;
                }
                for row in pixels.chunks_exact(layout.bytes_per_row.get() as usize) {
                    for pixel in row[..16].chunks_exact(4) {
                        // Quantizing the intermediate scene image permits one byte of error.
                        assert!(
                            pixel[..3].iter().all(|v| v.abs_diff(118) <= 1),
                            "{source_format:?} -> {target_format:?}: {pixel:?}"
                        );
                        assert_eq!(pixel[3], 255);
                    }
                }
                rhi.finish_cycle()?;
            }
        }
        rhi.wait_idle()?;
        Ok(())
    }
}
