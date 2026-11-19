//! Render graphs executed under validation, read back by the host.

use std::{cell::Cell, time::Duration};

use dirk_render_utils::{Handle, RenderGraph, Transients, Uploader};
use dirk_rhi::{
    Buffer, BufferAccess, BufferCopy, BufferDesc, BufferImageCopy, BufferUsage, ColorTarget,
    CompareOp, Completion, CullMode, DepthBias, DepthState, Error, Extent3d, FRAMES_IN_FLIGHT,
    Filter, Format, FrontFace, Graphics, GraphicsPipeline, GraphicsPipelineDesc, Image,
    ImageAccess, ImageDesc, ImageDimension, ImageRange, ImageRegion, ImageUsage, ImageView, LoadOp,
    Memory, Origin3d, PushData, Result, Rhi, SampleCount, Sampler, SamplerDesc, Shader,
    ShaderStages, Subresource, TextureIndex, Topology,
};
use dirk_rhi_test_shaders::support::with_rhi;
use dirk_rhi_test_shaders::{Fill, FillFs, FillVs, Textured, TexturedFs, TexturedVs, Vertex};

const SIZE: u32 = 8;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLACK: LoadOp<[f32; 4]> = LoadOp::Clear([0.0, 0.0, 0.0, 1.0]);
const FRAGMENT: ShaderStages = ShaderStages::FRAGMENT;

/// Renders a solid color into a transient over a transient depth image,
/// then samples it into a target.
struct Scene {
    fill: GraphicsPipeline<Fill>,
    textured: GraphicsPipeline<Textured>,
    /// A quad covering the target. Only the host writes it, so passes read
    /// it without declaring it.
    quad: Buffer,
    sampler: Sampler,
}

impl Scene {
    fn new(rhi: &Rhi) -> Result<Self> {
        let corner = |x: f32, y: f32| Vertex {
            position: [x * 2.0 - 1.0, y * 2.0 - 1.0],
            uv: [x, y],
        };
        let quad = [(0, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 1)]
            .map(|(x, y): (u8, u8)| corner(f32::from(x), f32::from(y)));
        let mut vertices = rhi.create_buffer(&BufferDesc {
            label: "quad",
            size: size_of_val(&quad) as u64,
            usage: BufferUsage::STORAGE,
            memory: Memory::Upload,
        })?;
        // SAFETY: the buffer was never submitted.
        unsafe { vertices.write(0, bytemuck::cast_slice(&quad))? };
        let depth = DepthState {
            format: Format::Depth32Float,
            write: true,
            compare: CompareOp::Less,
            bias: DepthBias::default(),
        };
        Ok(Self {
            fill: Self::pipeline::<FillVs, FillFs, Fill>(rhi, Some(depth))?,
            textured: Self::pipeline::<TexturedVs, TexturedFs, Textured>(rhi, None)?,
            quad: vertices,
            sampler: rhi.create_sampler(&SamplerDesc {
                label: "nearest",
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_filter: Filter::Nearest,
                ..SamplerDesc::default()
            })?,
        })
    }

    /// A pipeline drawing to one `Rgba8Unorm` target and `depth`.
    fn pipeline<V: Shader<Push = P>, F: Shader<Push = P>, P: PushData>(
        rhi: &Rhi,
        depth: Option<DepthState>,
    ) -> Result<GraphicsPipeline<P>> {
        rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: F::ENTRY_POINT,
            vertex: &rhi.create_shader::<V>()?,
            fragment: Some(&rhi.create_shader::<F>()?),
            topology: Topology::default(),
            cull_mode: CullMode::default(),
            front_face: FrontFace::default(),
            color_targets: &[ColorTarget::new(Format::Rgba8Unorm)],
            depth,
            samples: SampleCount::X1,
        })
    }

    /// Adds the passes rendering `color` into `target`. `scene` receives the
    /// transient's texture index when the sampling pass records.
    fn add<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        target: Handle<Image>,
        color: [f32; 4],
        scene: &'a Cell<Option<TextureIndex>>,
    ) -> Result<()> {
        let desc = |label, format| ImageDesc {
            label,
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(SIZE, SIZE),
            format,
            usage: ImageUsage::empty(),
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };
        let rendered = graph.create_image(&desc("scene", Format::Rgba8Unorm));
        let depth = graph.create_image(&desc("depth", Format::Depth32Float));
        graph
            .pass("scene")
            .color(rendered, BLACK)
            .depth(depth, LoadOp::Clear(1.0))
            .record(move |pass| pass.draw(&self.fill, &Fill { color, depth: 0.5 }, 0..3, 0..1))?;
        graph
            .pass("post")
            .color(target, BLACK)
            .sampled(rendered, FRAGMENT)
            .record(move |pass| {
                let texture = pass
                    .image(rendered)?
                    .view()
                    .and_then(ImageView::index)
                    .expect("sampled transients have a texture index");
                scene.set(Some(texture));
                let push = Textured {
                    vertices: self.quad.index().expect("storage buffers have an index"),
                    texture,
                    sampler: self.sampler.index(),
                };
                pass.draw(&self.textured, &push, 0..6, 0..1)
            })
    }

    /// Renders `color` into an 8x8 target in a new frame and returns its
    /// pixels and the transient's texture index.
    fn render(
        &self,
        rhi: &mut Rhi,
        transients: &mut Transients,
        target: &Image,
        readback: &Buffer,
        color: [f32; 4],
    ) -> Result<(Vec<u8>, TextureIndex)> {
        let scene = Cell::new(None);
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("scene")?;
        let mut graph = RenderGraph::new();
        // Like a depth buffer: rendered anew each frame.
        let handle = graph.import_scratch(target)?;
        self.add(&mut graph, handle, color, &scene)?;
        copy_out(&mut graph, handle, readback)?;
        graph.execute(&frame, &mut encoder, transients)?;
        let done = frame.submit(encoder)?;
        frame.finish();
        let pixels = read(rhi, done, readback)?;
        Ok((pixels, scene.get().expect("the sampling pass recorded")))
    }
}

/// An 8x8 color target and a buffer to read it back into.
fn target(rhi: &Rhi) -> Result<(Image, Buffer)> {
    let image = rhi.create_image(&ImageDesc {
        label: "target",
        dimension: ImageDimension::D2,
        extent: Extent3d::new_2d(SIZE, SIZE),
        format: Format::Rgba8Unorm,
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::COPY_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::X1,
        rest: None,
    })?;
    Ok((image, readback(rhi, u64::from(SIZE * SIZE * 4))?))
}

/// A host-readable buffer of `size` bytes.
fn readback(rhi: &Rhi, size: u64) -> Result<Buffer> {
    rhi.create_buffer(&BufferDesc {
        label: "readback",
        size,
        usage: BufferUsage::COPY_DST,
        memory: Memory::Readback,
    })
}

/// Adds a pass copying mip 0 of `image` into `readback`, which the graph
/// leaves for host reads.
fn copy_out<'a>(
    graph: &mut RenderGraph<'a>,
    image: Handle<Image>,
    readback: &'a Buffer,
) -> Result<()> {
    let buffer = graph.import_buffer(readback, BufferAccess::None, BufferAccess::HostRead)?;
    graph
        .pass("read back")
        .image(image, ImageAccess::CopySource)
        .buffer(buffer, BufferAccess::CopyDestination)
        .record(move |pass| {
            let (image, buffer) = (pass.image(image)?, pass.buffer(buffer)?);
            let info = image.info();
            let copy = BufferImageCopy::packed(info.format, 0, info.mip_region(0));
            pass.copy_image_to_buffer(image, buffer, &[copy])
        })
}

/// Waits for `done`, then reads `readback`.
fn read(rhi: &Rhi, done: Completion, readback: &Buffer) -> Result<Vec<u8>> {
    rhi.wait(done, Duration::from_secs(10))?;
    let mut bytes = vec![0; usize::try_from(readback.info().size).unwrap_or(0)];
    // SAFETY: the graph left the buffer for host reads and its work
    // completed.
    unsafe { readback.read(0, &mut bytes)? };
    Ok(bytes)
}

/// The 8-bit pixel of `color`, whose channels are zero or one.
fn unorm(color: [f32; 4]) -> [u8; 4] {
    color.map(|channel| if channel > 0.5 { 255 } else { 0 })
}

#[test]
fn rendered_transients_are_sampled_into_an_import_and_copied_out() -> Result<()> {
    with_rhi(|rhi| {
        let scene = Scene::new(rhi)?;
        let (target, readback) = target(rhi)?;
        let mut transients = Transients::new();
        let (pixels, _) = scene.render(rhi, &mut transients, &target, &readback, RED)?;
        assert!(pixels.chunks(4).all(|pixel| pixel == unorm(RED)));
        Ok(())
    })
}

#[test]
fn transients_are_reused_once_their_frame_completed() -> Result<()> {
    with_rhi(|rhi| {
        let scene = Scene::new(rhi)?;
        let (target, readback) = target(rhi)?;
        let mut transients = Transients::new();
        let mut textures = Vec::new();
        for frame in 0..=FRAMES_IN_FLIGHT {
            let color = if frame % 2 == 0 { RED } else { GREEN };
            let (pixels, texture) =
                scene.render(rhi, &mut transients, &target, &readback, color)?;
            assert!(pixels.chunks(4).all(|pixel| pixel == unorm(color)));
            textures.push(texture);
        }
        assert_ne!(textures[0], textures[1], "frame 1 is still in flight");
        assert_eq!(textures[0], textures[FRAMES_IN_FLIGHT]);
        Ok(())
    })
}

#[test]
fn transfer_only_transients_have_no_view() -> Result<()> {
    with_rhi(|rhi| {
        let texels: Vec<u8> = (0..16).collect();
        let mut staging = rhi.create_buffer(&BufferDesc {
            label: "staging",
            size: 16,
            usage: BufferUsage::COPY_SRC,
            memory: Memory::Upload,
        })?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, &texels)? };
        let staging = &staging;
        let readback = readback(rhi, 16)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("transfers")?;
        let mut graph = RenderGraph::new();
        let buffer = graph.create_buffer(&BufferDesc {
            label: "texels",
            size: 16,
            usage: BufferUsage::empty(),
            memory: Memory::Device,
        });
        let image = graph.create_image(&ImageDesc {
            label: "copied",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(2, 2),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::empty(),
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        });
        graph
            .pass("stage")
            .buffer(buffer, BufferAccess::CopyDestination)
            .record(move |pass| {
                let copy = BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 16,
                };
                let buffer = pass.buffer(buffer)?;
                pass.copy_buffer(staging, buffer, &[copy])
            })?;
        graph
            .pass("copy in")
            .buffer(buffer, BufferAccess::CopySource)
            .image(image, ImageAccess::CopyDestination)
            .record(move |pass| {
                let (buffer, image) = (pass.buffer(buffer)?, pass.image(image)?);
                assert!(image.view().is_none(), "transfer-only images have no view");
                let info = image.info();
                let copy = BufferImageCopy::packed(info.format, 0, info.mip_region(0));
                pass.copy_buffer_to_image(buffer, image, &[copy])
            })?;
        copy_out(&mut graph, image, &readback)?;
        graph.execute(&frame, &mut encoder, &mut Transients::new())?;
        let done = frame.submit(encoder)?;
        frame.finish();
        assert_eq!(read(rhi, done, &readback)?, texels);
        Ok(())
    })
}

#[test]
fn imported_images_start_and_end_at_rest() -> Result<()> {
    with_rhi(|rhi| {
        let texels: Vec<u8> = (0..16).map(|texel| texel * 3).collect();
        let readback = readback(rhi, 16)?;
        let mut uploader = Uploader::new(rhi, 64)?;
        let mut transients = Transients::new();
        let desc = |label, usage| ImageDesc {
            label,
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(2, 2),
            format: Format::Rgba8Unorm,
            usage,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };
        let sampled = ImageUsage::SAMPLED | ImageUsage::COPY_SRC;
        let mut frame = rhi.begin_frame()?;
        let mut uploads = uploader.begin(&mut frame);
        let texture = uploads.image(&desc("uploaded", sampled), &[&texels])?;
        uploads.submit()?;
        let mut encoder = frame.encoder::<Graphics>("imports")?;

        // An import no pass uses stays at rest.
        let mut graph = RenderGraph::new();
        let image = graph.import(&texture)?;
        assert_eq!(
            graph.import(&texture)?,
            image,
            "importing again returns the same handle"
        );
        let scratch = graph.import_scratch(&texture);
        assert!(matches!(scratch, Err(Error::Invalid { .. })));
        graph.execute(&frame, &mut encoder, &mut transients)?;
        frame.submit(encoder)?;
        frame.finish();

        // Later frames import it with no state given, finding it at rest.
        for _ in 0..FRAMES_IN_FLIGHT {
            let frame = rhi.begin_frame()?;
            let mut encoder = frame.encoder::<Graphics>("sampled")?;
            let mut graph = RenderGraph::new();
            let image = graph.import(&texture)?;
            graph
                .pass("sample")
                .sampled(image, FRAGMENT)
                .record(|_| Ok(()))?;
            copy_out(&mut graph, image, &readback)?;
            graph.execute(&frame, &mut encoder, &mut transients)?;
            let done = frame.submit(encoder)?;
            frame.finish();
            assert_eq!(read(rhi, done, &readback)?, texels);
        }

        // An image nothing wrote yet is imported from its undefined first
        // contents, which its first pass overwrites.
        let fresh = rhi.create_image(&desc("fresh", sampled | ImageUsage::COLOR_ATTACHMENT))?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("first import")?;
        let mut graph = RenderGraph::new();
        let image = graph.import(&fresh)?;
        graph
            .pass("clear")
            .color(image, LoadOp::Clear(RED))
            .record(|_| Ok(()))?;
        copy_out(&mut graph, image, &readback)?;
        graph.execute(&frame, &mut encoder, &mut transients)?;
        let done = frame.submit(encoder)?;
        frame.finish();
        assert!(
            read(rhi, done, &readback)?
                .chunks(4)
                .all(|pixel| pixel == unorm(RED))
        );
        Ok(())
    })
}

#[test]
fn passes_render_a_mip_chain_sampling_each_previous_mip() -> Result<()> {
    with_rhi(|rhi| {
        let scene = Scene::new(rhi)?;
        let readback = readback(rhi, 4)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("mip chain")?;
        let mut graph = RenderGraph::new();
        let chain = graph.create_image(&ImageDesc {
            label: "chain",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(SIZE, SIZE),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::empty(),
            mip_levels: 4,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        });
        let mip = |mip_level| Subresource {
            mip_level,
            layer: 0,
        };
        let fill = &scene.fill;
        let depth = graph.create_image(&ImageDesc {
            label: "depth",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(SIZE, SIZE),
            format: Format::Depth32Float,
            usage: ImageUsage::empty(),
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        });
        graph
            .pass("mip 0")
            .color(chain.subresource(mip(0)), BLACK)
            .depth(depth, LoadOp::Clear(1.0))
            .record(move |pass| {
                let green = Fill {
                    color: GREEN,
                    depth: 0.5,
                };
                pass.draw(fill, &green, 0..3, 0..1)
            })?;
        let scene = &scene;
        for level in 1..4 {
            graph
                .pass("downsample")
                .sampled(chain.slice(ImageRange::mip(level - 1)), FRAGMENT)
                .color(chain.subresource(mip(level)), BLACK)
                .record(move |pass| {
                    let texture = pass
                        .image(chain)?
                        .subview(mip(level - 1))
                        .and_then(ImageView::index)
                        .expect("rendered mips have sampled subviews");
                    let push = Textured {
                        vertices: scene.quad.index().expect("storage buffers have an index"),
                        texture,
                        sampler: scene.sampler.index(),
                    };
                    pass.draw(&scene.textured, &push, 0..6, 0..1)
                })?;
        }
        let buffer = graph.import_buffer(&readback, BufferAccess::None, BufferAccess::HostRead)?;
        graph
            .pass("read back")
            .image(chain.slice(ImageRange::mip(3)), ImageAccess::CopySource)
            .buffer(buffer, BufferAccess::CopyDestination)
            .record(move |pass| {
                let (image, buffer) = (pass.image(chain)?, pass.buffer(buffer)?);
                let region = ImageRegion {
                    mip_level: 3,
                    base_layer: 0,
                    layer_count: 1,
                    origin: Origin3d::default(),
                    extent: Extent3d::new_2d(1, 1),
                };
                let copy = BufferImageCopy::packed(Format::Rgba8Unorm, 0, region);
                pass.copy_image_to_buffer(image, buffer, &[copy])
            })?;
        graph.execute(&frame, &mut encoder, &mut Transients::new())?;
        let done = frame.submit(encoder)?;
        frame.finish();
        assert_eq!(read(rhi, done, &readback)?, unorm(GREEN));
        Ok(())
    })
}
