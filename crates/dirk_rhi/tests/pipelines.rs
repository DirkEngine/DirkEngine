//! Shaders, graphics pipelines, render passes, and draws reading the
//! bindless heap.

use dirk_rhi_test_shaders::support;

use std::time::Duration;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferDesc, BufferImageCopy, BufferUsage, ColorAttachment,
    ColorTarget, CommandEncoder, CompareOp, CullMode, DepthAttachment, DepthBias, DepthState,
    Error, Extent3d, FRAMES_IN_FLIGHT, Filter, Format, FrontFace, Graphics, GraphicsPipeline,
    GraphicsPipelineDesc, Image, ImageDesc, ImageDimension, ImageUsage, ImageView, IndexFormat,
    Indices, LoadOp, Memory, PushData, Rect, RenderPassDesc, Result, Rhi, SampleCount, SamplerDesc,
    Shader, StoreOp, Topology,
};
use dirk_rhi_test_shaders::{Fill, FillFs, FillVs, Textured, TexturedFs, TexturedVs, Vertex};
use support::with_rhi;

const SIZE: u32 = 8;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// An 8x8 2D attachment of `format` with `samples`.
fn attachment(rhi: &Rhi, format: Format, usage: ImageUsage, samples: SampleCount) -> Result<Image> {
    rhi.create_image(&ImageDesc {
        label: "attachment",
        dimension: ImageDimension::D2,
        extent: Extent3d::new_2d(SIZE, SIZE),
        format,
        usage,
        mip_levels: 1,
        array_layers: 1,
        samples,
        rest: None,
    })
}

/// A pipeline rendering to `color_targets` with shaders `V` and `F`.
fn graphics_pipeline<V: Shader, F: Shader, P: PushData>(
    rhi: &Rhi,
    color_targets: &[ColorTarget],
    depth: Option<DepthState>,
    samples: SampleCount,
) -> Result<GraphicsPipeline<P>> {
    let vertex = rhi.create_shader::<V, P>()?;
    let fragment = rhi.create_shader::<F, P>()?;
    rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
        label: F::ENTRY_POINT,
        vertex: &vertex,
        fragment: Some(&fragment),
        topology: Topology::default(),
        cull_mode: CullMode::default(),
        front_face: FrontFace::default(),
        color_targets,
        depth,
        samples,
    })
}

/// Clears an 8x8 `Rgba8Unorm` target to black, lets `record` render to it
/// and returns its pixels.
fn render(
    rhi: &mut Rhi,
    record: impl FnOnce(&mut CommandEncoder<'_>, &Image) -> Result<()>,
) -> Result<Vec<[u8; 4]>> {
    let target = attachment(
        rhi,
        Format::Rgba8Unorm,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::COPY_SRC,
        SampleCount::X1,
    )?;
    let readback = rhi.create_buffer(&BufferDesc {
        label: "readback",
        size: u64::from(SIZE * SIZE * 4),
        usage: BufferUsage::COPY_DST,
        memory: Memory::Readback,
    })?;
    let frame = rhi.begin_frame()?;
    let mut encoder = frame.encoder::<Graphics>("render")?;
    record(&mut encoder, &target)?;
    encoder.copy_image_to_buffer(
        &target,
        &readback,
        &[BufferImageCopy::packed(
            Format::Rgba8Unorm,
            0,
            target.info().mip_region(0),
        )],
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
    let done = frame.submit(encoder)?;
    frame.finish();
    rhi.wait(done, Duration::from_secs(10))?;
    let mut pixels = vec![[0; 4]; (SIZE * SIZE) as usize];
    // SAFETY: the copy completed, followed by a barrier to host reads.
    unsafe { readback.read(0, pixels.as_flattened_mut())? };
    Ok(pixels)
}

/// The 8-bit color of `color`.
fn unorm(color: [f32; 4]) -> [u8; 4] {
    // Test colors are exactly zero or one.
    color.map(|channel| if channel > 0.5 { 255 } else { 0 })
}

/// A color attachment cleared to black.
fn cleared(image: &Image) -> ColorAttachment<'_> {
    ColorAttachment {
        image,
        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        store: StoreOp::Store,
        resolve: None,
    }
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
fn draws_and_indexed_draws_fill_their_scissor() -> Result<()> {
    with_rhi(|rhi| {
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let mut indices = rhi.create_buffer(&BufferDesc {
            label: "indices",
            size: 8,
            usage: BufferUsage::INDEX,
            memory: Memory::Upload,
        })?;
        // SAFETY: the buffer was never submitted.
        unsafe { indices.write(0, bytemuck::cast_slice(&[7_u16, 0, 1, 2]))? };
        let half = SIZE / 2;
        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "halves",
                colors: &[cleared(target)],
                depth: None,
            })?;
            pass.set_scissor(Rect {
                x: 0,
                y: 0,
                width: half,
                height: SIZE,
            })?;
            let red = Fill {
                color: RED,
                depth: 0.5,
            };
            pass.draw(&pipeline, &red, 0..3, 0..1)?;
            pass.set_scissor(Rect {
                x: half,
                y: 0,
                width: half,
                height: SIZE,
            })?;
            let green = Fill {
                color: GREEN,
                depth: 0.5,
            };
            let indices = Indices {
                buffer: &indices,
                format: IndexFormat::U16,
                offset: 2,
                range: 0..3,
                base_vertex: 0,
            };
            pass.draw_indexed(&pipeline, &green, indices, 0..1)?;
            Ok(())
        })?;
        for (index, pixel) in pixels.iter().enumerate() {
            let left = index % (SIZE as usize) < half as usize;
            assert_eq!(
                *pixel,
                unorm(if left { RED } else { GREEN }),
                "pixel {index}"
            );
        }
        Ok(())
    })
}

#[test]
fn the_depth_test_keeps_the_nearest_fill() -> Result<()> {
    with_rhi(|rhi| {
        let format = Format::Depth32Float;
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            Some(DepthState {
                format,
                write: true,
                compare: CompareOp::Less,
                bias: DepthBias::default(),
            }),
            SampleCount::X1,
        )?;
        let depth = attachment(
            rhi,
            format,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            SampleCount::X1,
        )?;
        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "depth test",
                colors: &[cleared(target)],
                depth: Some(DepthAttachment {
                    image: &depth,
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Discard,
                }),
            })?;
            let near = Fill {
                color: RED,
                depth: 0.25,
            };
            let far = Fill {
                color: GREEN,
                depth: 0.75,
            };
            pass.draw(&pipeline, &near, 0..3, 0..1)?;
            pass.draw(&pipeline, &far, 0..3, 0..1)?;
            Ok(())
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(RED)));
        Ok(())
    })
}

#[test]
fn culling_follows_the_winding_on_screen() -> Result<()> {
    with_rhi(|rhi| {
        let vertex = rhi.create_shader::<FillVs, Fill>()?;
        let fragment = rhi.create_shader::<FillFs, Fill>()?;
        let targets = [ColorTarget::new(Format::Rgba8Unorm)];
        let culling_back = |front_face| {
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                label: "culling",
                vertex: &vertex,
                fragment: Some(&fragment),
                topology: Topology::TriangleList,
                cull_mode: CullMode::Back,
                front_face,
                color_targets: &targets,
                depth: None,
                samples: SampleCount::X1,
            })
        };
        let clockwise = culling_back(FrontFace::Clockwise)?;
        let counter_clockwise = culling_back(FrontFace::CounterClockwise)?;
        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "culling",
                colors: &[cleared(target)],
                depth: None,
            })?;
            // The fill triangle runs from the top left to the top right and
            // the bottom left: clockwise on screen, so front-facing only for
            // the first pipeline.
            let fill = |color| Fill { color, depth: 0.5 };
            pass.draw(&clockwise, &fill(GREEN), 0..3, 0..1)?;
            pass.draw(&counter_clockwise, &fill(RED), 0..3, 0..1)?;
            Ok(())
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(GREEN)));
        Ok(())
    })
}

#[test]
fn multisampled_targets_resolve_when_the_pass_ends() -> Result<()> {
    with_rhi(|rhi| {
        let samples = SampleCount::X4;
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            samples,
        )?;
        let multisampled = attachment(
            rhi,
            Format::Rgba8Unorm,
            ImageUsage::COLOR_ATTACHMENT,
            samples,
        )?;
        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "resolve",
                colors: &[ColorAttachment {
                    image: &multisampled,
                    load: LoadOp::Discard,
                    store: StoreOp::Discard,
                    resolve: Some(target),
                }],
                depth: None,
            })?;
            let blue = Fill {
                color: BLUE,
                depth: 0.5,
            };
            pass.draw(&pipeline, &blue, 0..3, 0..1)?;
            Ok(())
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(BLUE)));
        Ok(())
    })
}

#[test]
fn pipelines_are_validated_at_creation() -> Result<()> {
    with_rhi(|rhi| {
        let vertex = rhi.create_shader::<FillVs, Fill>()?;
        let fragment = rhi.create_shader::<FillFs, Fill>()?;
        let color = [ColorTarget::new(Format::Rgba8Unorm)];
        let desc = GraphicsPipelineDesc {
            label: "invalid",
            vertex: &vertex,
            fragment: Some(&fragment),
            topology: Topology::default(),
            cull_mode: CullMode::default(),
            front_face: FrontFace::default(),
            color_targets: &color,
            depth: None,
            samples: SampleCount::X1,
        };
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: &fragment,
                fragment: Some(&vertex),
                ..desc
            }),
            "not a vertex and a fragment shader",
        );
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                fragment: None,
                ..desc
            }),
            "need a fragment shader",
        );
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                color_targets: &[ColorTarget::new(Format::Depth32Float)],
                ..desc
            }),
            "cannot be a color target",
        );
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                color_targets: &[],
                depth: Some(DepthState {
                    format: Format::Rgba8Unorm,
                    write: true,
                    compare: CompareOp::Less,
                    bias: DepthBias::default(),
                }),
                ..desc
            }),
            "cannot be a depth target",
        );
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                color_targets: &[ColorTarget::new(Format::Rgba8Unorm); 9],
                ..desc
            }),
            "exceed the limit",
        );
        assert_invalid(
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                color_targets: &[],
                ..desc
            }),
            "needs a color or depth target",
        );
        Ok(())
    })
}

#[test]
fn draws_and_passes_are_validated() -> Result<()> {
    with_rhi(|rhi| {
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba16Float)],
            None,
            SampleCount::X1,
        )?;
        let target = attachment(
            rhi,
            Format::Rgba8Unorm,
            ImageUsage::COLOR_ATTACHMENT,
            SampleCount::X1,
        )?;
        let smaller = rhi.create_image(&ImageDesc {
            extent: Extent3d::new_2d(SIZE / 2, SIZE / 2),
            ..desc_of(&target)
        })?;
        let mipped = rhi.create_image(&ImageDesc {
            mip_levels: 2,
            ..desc_of(&target)
        })?;
        let indices: Buffer = rhi.create_buffer(&BufferDesc {
            label: "indices",
            size: 6,
            usage: BufferUsage::INDEX,
            memory: Memory::Device,
        })?;
        let fill = Fill {
            color: RED,
            depth: 0.5,
        };

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("invalid draws")?;
        assert_invalid(
            encoder
                .render_pass(&RenderPassDesc {
                    label: "mismatched",
                    colors: &[cleared(&target), cleared(&smaller)],
                    depth: None,
                })
                .map(|_| ()),
            "unlike",
        );
        assert_invalid(
            encoder
                .render_pass(&RenderPassDesc {
                    label: "mipped",
                    colors: &[cleared(&mipped)],
                    depth: None,
                })
                .map(|_| ()),
            "single-mip",
        );
        let mut pass = encoder.render_pass(&RenderPassDesc {
            label: "valid",
            colors: &[cleared(&target)],
            depth: None,
        })?;
        assert_invalid(pass.draw(&pipeline, &fill, 0..3, 0..1), "not this pass's");
        let rgba8 = graphics_pipeline::<FillVs, FillFs, Fill>(
            frame.rhi(),
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let index = |offset, range| Indices {
            buffer: &indices,
            format: IndexFormat::U16,
            offset,
            range,
            base_vertex: 0,
        };
        assert_invalid(
            pass.draw_indexed(&rgba8, &fill, index(2, 0..3), 0..1),
            "exceed",
        );
        assert_invalid(
            pass.draw_indexed(&rgba8, &fill, index(1, 0..2), 0..1),
            "misaligned",
        );
        assert_invalid(
            pass.set_scissor(Rect {
                x: 4,
                y: 0,
                width: SIZE,
                height: 1,
            }),
            "render area",
        );
        pass.end();
        frame.finish();
        Ok(())
    })
}

/// The description `image` was created with.
#[test]
fn leaked_render_passes_end_before_the_next_command() -> Result<()> {
    with_rhi(|rhi| {
        let pixels = render(rhi, |encoder, target| {
            let pass = encoder.render_pass(&RenderPassDesc {
                label: "leaked",
                colors: &[ColorAttachment {
                    load: LoadOp::Clear(RED),
                    ..cleared(target)
                }],
                depth: None,
            })?;
            std::mem::forget(pass);
            Ok(())
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(RED)));
        Ok(())
    })
}

fn desc_of(image: &Image) -> ImageDesc<'static> {
    let info = image.info();
    ImageDesc {
        label: "variant",
        dimension: info.dimension,
        extent: info.extent,
        format: info.format,
        usage: info.usage,
        mip_levels: info.mip_levels,
        array_layers: info.array_layers,
        samples: info.samples,
        rest: None,
    }
}

#[test]
fn shaders_read_storage_buffers_of_different_types_through_the_heap() -> Result<()> {
    use dirk_rhi_test_shaders::{ColorFs, Instanced, InstancedVs};

    with_rhi(|rhi| {
        let storage = |label, data: &[u8]| {
            let mut buffer = rhi.create_buffer(&BufferDesc {
                label,
                size: data.len() as u64,
                usage: BufferUsage::STORAGE,
                memory: Memory::Upload,
            })?;
            // SAFETY: the buffer was never submitted.
            unsafe { buffer.write(0, data)? };
            Ok::<_, Error>(buffer)
        };
        // One triangle covering the target, and colors of 4 bytes, not the
        // 16 of a vertex, so each array is read with its own stride.
        let triangle = [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)].map(|(x, y)| Vertex {
            position: [x, y],
            uv: [0.0; 2],
        });
        let colors = [RED, GREEN, BLUE].map(|color| u32::from_le_bytes(unorm(color)));
        let vertices = storage("triangle", bytemuck::cast_slice(&triangle))?;
        let colors = storage("colors", bytemuck::cast_slice(&colors))?;
        let push = Instanced {
            vertices: vertices.index().expect("storage buffers have an index"),
            colors: colors.index().expect("storage buffers have an index"),
        };
        let pipeline = graphics_pipeline::<InstancedVs, ColorFs, Instanced>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;

        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "instanced triangle",
                colors: &[cleared(target)],
                depth: None,
            })?;
            pass.draw(&pipeline, &push, 0..3, 2..3)
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(BLUE)));
        Ok(())
    })
}

#[test]
fn draws_pull_vertices_and_sample_textures_through_the_heap() -> Result<()> {
    with_rhi(|rhi| {
        // A 2x2 texture, red and green above blue and white.
        let texels = [RED, GREEN, BLUE, [1.0; 4]].map(unorm);
        let texture = rhi.create_image(&ImageDesc {
            label: "checker",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(2, 2),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED | ImageUsage::COPY_DST,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        })?;
        let mut staging = rhi.create_buffer(&BufferDesc {
            label: "texels",
            size: 16,
            usage: BufferUsage::COPY_SRC,
            memory: Memory::Upload,
        })?;
        // SAFETY: the buffer was never submitted.
        unsafe { staging.write(0, texels.as_flattened())? };
        let sampler = rhi.create_sampler(&SamplerDesc {
            label: "nearest",
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_filter: Filter::Nearest,
            ..SamplerDesc::default()
        })?;
        // Two triangles covering the target; clip-space y points down.
        let corner = |x: f32, y: f32| Vertex {
            position: [x * 2.0 - 1.0, y * 2.0 - 1.0],
            uv: [x, y],
        };
        let quad = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]
        .map(|(x, y)| corner(x, y));
        let mut vertices = rhi.create_buffer(&BufferDesc {
            label: "quad",
            size: size_of_val(&quad) as u64,
            usage: BufferUsage::STORAGE,
            memory: Memory::Upload,
        })?;
        // SAFETY: the buffer was never submitted.
        unsafe { vertices.write(0, bytemuck::cast_slice(&quad))? };
        let push = Textured {
            vertices: vertices.index().expect("storage buffers have an index"),
            texture: texture
                .view()
                .and_then(ImageView::index)
                .expect("sampled images have an index"),
            sampler: sampler.index(),
        };
        let pipeline = graphics_pipeline::<TexturedVs, TexturedFs, Textured>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;

        let pixels = render(rhi, |encoder, target| {
            encoder.copy_buffer_to_image(
                &staging,
                &texture,
                &[BufferImageCopy::packed(
                    Format::Rgba8Unorm,
                    0,
                    texture.info().mip_region(0),
                )],
            )?;
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "textured quad",
                colors: &[cleared(target)],
                depth: None,
            })?;
            pass.draw(&pipeline, &push, 0..6, 0..1)
        })?;
        for (index, pixel) in pixels.iter().enumerate() {
            let (x, y) = (index % SIZE as usize, index / SIZE as usize);
            let texel =
                usize::from(x >= SIZE as usize / 2) + 2 * usize::from(y >= SIZE as usize / 2);
            assert_eq!(*pixel, texels[texel], "pixel ({x}, {y})");
        }
        Ok(())
    })
}

#[test]
fn heap_slots_are_reused_once_the_gpu_is_done() -> Result<()> {
    with_rhi(|rhi| {
        let frame = rhi.begin_frame()?;
        let first = frame.rhi().create_sampler(&SamplerDesc::default())?;
        let slot = first.index();
        // The frame's work may still read the dropped sampler's slot.
        drop(first);
        let second = frame.rhi().create_sampler(&SamplerDesc::default())?;
        assert_ne!(second.index(), slot);
        frame.finish();
        for _ in 0..FRAMES_IN_FLIGHT {
            rhi.begin_frame()?.finish();
        }
        assert_eq!(rhi.create_sampler(&SamplerDesc::default())?.index(), slot);
        Ok(())
    })
}

#[test]
fn only_shader_visible_resources_have_indices() -> Result<()> {
    with_rhi(|rhi| {
        let buffer = |usage| {
            rhi.create_buffer(&BufferDesc {
                label: "buffer",
                size: 16,
                usage,
                memory: Memory::Device,
            })
        };
        assert!(buffer(BufferUsage::COPY_DST)?.index::<u32>().is_none());
        assert!(buffer(BufferUsage::STORAGE)?.index::<u32>().is_some());

        let image = |dimension, array_layers, usage| {
            rhi.create_image(&ImageDesc {
                label: "image",
                dimension,
                extent: Extent3d::new_2d(4, 4),
                format: Format::Rgba8Unorm,
                usage,
                mip_levels: 1,
                array_layers,
                samples: SampleCount::X1,
                rest: None,
            })
        };
        let indices = |image: &Image| {
            let view = image.view().expect("these images have a view");
            (view.index().is_some(), view.storage_index().is_some())
        };
        let both = ImageUsage::SAMPLED | ImageUsage::STORAGE;
        let flat = image(ImageDimension::D2, 1, both)?;
        assert_eq!(indices(&flat), (true, true));
        let target = image(ImageDimension::D2, 1, ImageUsage::COLOR_ATTACHMENT)?;
        assert_eq!(indices(&target), (false, false));
        // Cubes are only sampled through the heap.
        let cube = image(ImageDimension::Cube, 6, both)?;
        assert_eq!(indices(&cube), (true, false));
        Ok(())
    })
}
