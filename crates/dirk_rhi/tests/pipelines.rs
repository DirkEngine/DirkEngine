//! Shaders, graphics pipelines, render passes, and draws reading the
//! bindless heap.

use dirk_rhi_test_shaders::support;

use std::time::Duration;

use bytemuck::NoUninit;
use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferDesc, BufferImageCopy, BufferIndex, BufferUsage,
    ColorAttachment, ColorTarget, CommandEncoder, CompareOp, ComputePipelineDesc, CullMode,
    DepthAttachment, DepthBias, DepthState, DrawArguments, DrawIndexedArguments, Error, Extent3d,
    FRAMES_IN_FLIGHT, Filter, Format, FrontFace, Graphics, GraphicsPipeline, GraphicsPipelineDesc,
    HeapIndices, Image, ImageAccess, ImageBarrier, ImageDesc, ImageDimension, ImageRange,
    ImageUsage, ImageView, IndexBuffer, IndexFormat, Indices, IndirectDraws, LoadOp, Memory,
    PushData, Rect, RenderPassDesc, Result, Rhi, SampleCount, Sampler, SamplerDesc, SamplerIndex,
    Shader, ShaderStages, StoreOp, Topology, Viewport,
};
use dirk_rhi_test_shaders::{
    ColorFs, Count, CountCs, CoveringVs, DepthCompare, DepthCompareFs, Fill, FillFs, FillVs, Paint,
    PaintCs, Sliced, SlicedVs, Stored, StoredFs, Textured, TexturedFs, TexturedVs, Vertex,
};
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
fn graphics_pipeline<V: Shader<Push = P>, F: Shader<Push = P>, P: PushData>(
    rhi: &Rhi,
    color_targets: &[ColorTarget],
    depth: Option<DepthState>,
    samples: SampleCount,
) -> Result<GraphicsPipeline<P>> {
    let shaders = (rhi.create_shader::<V>()?, rhi.create_shader::<F>()?);
    pipeline(rhi, shaders, color_targets, depth, samples)
}

/// A pipeline rendering to `color_targets` with a vertex and a fragment
/// shader.
fn pipeline<P: PushData>(
    rhi: &Rhi,
    (vertex, fragment): (dirk_rhi::ShaderModule<P>, dirk_rhi::ShaderModule<P>),
    color_targets: &[ColorTarget],
    depth: Option<DepthState>,
    samples: SampleCount,
) -> Result<GraphicsPipeline<P>> {
    rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
        label: fragment.entry_point(),
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
#[expect(clippy::panic, reason = "Test helpers fail the test by panicking.")]
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
            // Empty draws record nothing.
            pass.draw(&pipeline, &red, 3..3, 0..1)?;
            pass.draw(&pipeline, &red, 0..3, 1..1)?;
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

/// An upload buffer of `usage` holding `data`.
fn upload(rhi: &Rhi, label: &str, usage: BufferUsage, data: &[u8]) -> Result<Buffer> {
    let mut buffer = rhi.create_buffer(&BufferDesc {
        label,
        size: data.len() as u64,
        usage,
        memory: Memory::Upload,
    })?;
    // SAFETY: the buffer was never submitted.
    unsafe { buffer.write(0, data)? };
    Ok(buffer)
}

#[test]
fn indirect_draws_read_their_arguments_from_a_buffer() -> Result<()> {
    with_rhi(|rhi| {
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        // No instances, then the covering triangle.
        let draws = [
            DrawArguments {
                vertex_count: 3,
                instance_count: 0,
                ..DrawArguments::default()
            },
            DrawArguments {
                vertex_count: 3,
                instance_count: 1,
                ..DrawArguments::default()
            },
        ];
        let arguments = upload(
            rhi,
            "draws",
            BufferUsage::INDIRECT,
            bytemuck::cast_slice(&draws),
        )?;
        let indexed = DrawIndexedArguments {
            index_count: 3,
            instance_count: 1,
            first_index: 1,
            ..DrawIndexedArguments::default()
        };
        let indexed_arguments = upload(
            rhi,
            "indexed draws",
            BufferUsage::INDIRECT,
            bytemuck::bytes_of(&indexed),
        )?;
        let indices = upload(
            rhi,
            "indices",
            BufferUsage::INDEX,
            bytemuck::cast_slice(&[7_u16, 7, 0, 1, 2]),
        )?;
        let fill = |color| Fill { color, depth: 0.5 };
        let mut render_draws = |offset, count, color| {
            render(rhi, |encoder, target| {
                let mut pass = encoder.render_pass(&RenderPassDesc {
                    label: "indirect",
                    colors: &[cleared(target)],
                    depth: None,
                })?;
                let draws = IndirectDraws {
                    buffer: &arguments,
                    offset,
                    count,
                };
                pass.draw_indirect(&pipeline, &fill(color), draws)
            })
        };
        let black = [0, 0, 0, 255];
        assert!(render_draws(0, 1, RED)?.iter().all(|pixel| *pixel == black));
        assert!(
            render_draws(0, 2, RED)?
                .iter()
                .all(|pixel| *pixel == unorm(RED))
        );
        assert!(
            render_draws(16, 1, GREEN)?
                .iter()
                .all(|pixel| *pixel == unorm(GREEN))
        );

        let pixels = render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "indexed indirect",
                colors: &[cleared(target)],
                depth: None,
            })?;
            let indices = IndexBuffer {
                buffer: &indices,
                format: IndexFormat::U16,
                // Aligned to the index size only.
                offset: 2,
            };
            let draws = IndirectDraws {
                buffer: &indexed_arguments,
                offset: 0,
                count: 1,
            };
            // SAFETY: the draw reads indices 1..4 of the four from offset 2.
            unsafe { pass.draw_indexed_indirect(&pipeline, &fill(BLUE), indices, draws) }
        })?;
        assert!(pixels.iter().all(|pixel| *pixel == unorm(BLUE)));
        Ok(())
    })
}

#[test]
fn indirect_draws_are_validated() -> Result<()> {
    with_rhi(|rhi| {
        let pipeline = graphics_pipeline::<FillVs, FillFs, Fill>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let draws = [DrawArguments::default(); 2];
        let arguments = upload(
            rhi,
            "draws",
            BufferUsage::INDIRECT,
            bytemuck::cast_slice(&draws),
        )?;
        let storage = upload(
            rhi,
            "storage",
            BufferUsage::STORAGE,
            bytemuck::cast_slice(&draws),
        )?;
        let fill = Fill {
            color: RED,
            depth: 0.5,
        };
        render(rhi, |encoder, target| {
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "invalid indirect",
                colors: &[cleared(target)],
                depth: None,
            })?;
            let mut draw = |buffer, offset, count| {
                pass.draw_indirect(
                    &pipeline,
                    &fill,
                    IndirectDraws {
                        buffer,
                        offset,
                        count,
                    },
                )
            };
            assert_invalid(draw(&storage, 0, 1), "INDIRECT");
            assert_invalid(draw(&arguments, 2, 1), "misaligned");
            assert_invalid(draw(&arguments, 16, 2), "exceed");
            Ok(())
        })?;
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
        let vertex = rhi.create_shader::<FillVs>()?;
        let fragment = rhi.create_shader::<FillFs>()?;
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
        let vertex = rhi.create_shader::<FillVs>()?;
        let fragment = rhi.create_shader::<FillFs>()?;
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
        assert_invalid(pass.draw(&pipeline, &fill, 3..3, 0..1), "not this pass's");
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
            pass.draw_indexed(&rgba8, &fill, index(1, 0..0), 0..1),
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

#[test]
fn render_passes_attach_each_image_once() -> Result<()> {
    with_rhi(|rhi| {
        let target = attachment(
            rhi,
            Format::Rgba8Unorm,
            ImageUsage::COLOR_ATTACHMENT,
            SampleCount::X1,
        )?;
        let multisampled = || {
            rhi.create_image(&ImageDesc {
                samples: SampleCount::X4,
                ..desc_of(&target)
            })
        };
        let multisampled = [multisampled()?, multisampled()?];
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("attached twice")?;
        assert_invalid(
            encoder
                .render_pass(&RenderPassDesc {
                    label: "attached twice",
                    colors: &[cleared(&target), cleared(&target)],
                    depth: None,
                })
                .map(|_| ()),
            "attached twice",
        );
        let resolved = |image| ColorAttachment {
            resolve: Some(&target),
            ..cleared(image)
        };
        assert_invalid(
            encoder
                .render_pass(&RenderPassDesc {
                    label: "resolved twice",
                    colors: &[resolved(&multisampled[0]), resolved(&multisampled[1])],
                    depth: None,
                })
                .map(|_| ()),
            "attached twice",
        );
        frame.finish();
        Ok(())
    })
}

#[test]
fn viewports_are_validated() -> Result<()> {
    with_rhi(|rhi| {
        let usage = ImageUsage::COLOR_ATTACHMENT;
        let target = attachment(rhi, Format::Rgba8Unorm, usage, SampleCount::X1)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("viewports")?;
        let mut pass = encoder.render_pass(&RenderPassDesc {
            label: "viewports",
            colors: &[cleared(&target)],
            depth: None,
        })?;
        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        pass.set_viewport(viewport)?;
        assert_invalid(
            pass.set_viewport(Viewport {
                width: 0.0,
                ..viewport
            }),
            "not empty",
        );
        // Larger and further out than any device allows.
        for viewport in [
            Viewport {
                width: 1e9,
                ..viewport
            },
            Viewport {
                y: -1e9,
                ..viewport
            },
        ] {
            assert_invalid(pass.set_viewport(viewport), "viewport limits");
        }
        pass.end();
        frame.finish();
        Ok(())
    })
}

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

        // A pass leaked last ends as its encoder is submitted.
        let usage = ImageUsage::COLOR_ATTACHMENT;
        let target = attachment(rhi, Format::Rgba8Unorm, usage, SampleCount::X1)?;
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("leaked last")?;
        std::mem::forget(encoder.render_pass(&RenderPassDesc {
            label: "leaked last",
            colors: &[cleared(&target)],
            depth: None,
        })?);
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, Duration::from_secs(10))
    })
}

/// The description `image` was created with.
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
        // The fragment shader reads no push data.
        let shaders = (
            rhi.create_shader::<InstancedVs>()?,
            rhi.create_shader::<ColorFs>()?.with_push(),
        );
        let pipeline = pipeline(
            rhi,
            shaders,
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

/// A 2x2 texture, red and green above blue and white, uploaded and drawn
/// over the target by a quad pulling its vertices through the heap.
struct TexturedQuad {
    texels: [[u8; 4]; 4],
    texture: Image,
    staging: Buffer,
    vertices: Buffer,
    pipeline: GraphicsPipeline<Textured>,
}

impl TexturedQuad {
    fn new(rhi: &Rhi) -> Result<Self> {
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
        let pipeline = graphics_pipeline::<TexturedVs, TexturedFs, Textured>(
            rhi,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        Ok(Self {
            texels,
            texture,
            staging,
            vertices,
            pipeline,
        })
    }

    /// A sampler reading the texture's texels as they are.
    fn sampler(rhi: &Rhi) -> Result<Sampler> {
        rhi.create_sampler(&SamplerDesc {
            label: "nearest",
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_filter: Filter::Nearest,
            ..SamplerDesc::default()
        })
    }

    /// Push data drawing the quad with `sampler`.
    #[expect(
        clippy::expect_used,
        reason = "Test helpers fail the test by panicking."
    )]
    fn push(&self, sampler: SamplerIndex) -> Textured {
        Textured {
            vertices: self
                .vertices
                .index()
                .expect("storage buffers have an index"),
            texture: self
                .texture
                .view()
                .and_then(ImageView::index)
                .expect("sampled images have an index"),
            sampler,
        }
    }

    /// Uploads the texture and draws the quad over `target` with `push`.
    fn draw(
        &self,
        encoder: &mut CommandEncoder<'_>,
        target: &Image,
        push: &Textured,
    ) -> Result<()> {
        encoder.copy_buffer_to_image(
            &self.staging,
            &self.texture,
            &[BufferImageCopy::packed(
                Format::Rgba8Unorm,
                0,
                self.texture.info().mip_region(0),
            )],
        )?;
        let mut pass = encoder.render_pass(&RenderPassDesc {
            label: "textured quad",
            colors: &[cleared(target)],
            depth: None,
        })?;
        pass.draw(&self.pipeline, push, 0..6, 0..1)
    }

    /// Checks that `pixels` show each texel over a quarter of the target.
    fn assert_drawn(&self, pixels: &[[u8; 4]]) {
        for (index, pixel) in pixels.iter().enumerate() {
            let (x, y) = (index % SIZE as usize, index / SIZE as usize);
            let texel =
                usize::from(x >= SIZE as usize / 2) + 2 * usize::from(y >= SIZE as usize / 2);
            assert_eq!(*pixel, self.texels[texel], "pixel ({x}, {y})");
        }
    }
}

#[test]
fn draws_pull_vertices_and_sample_textures_through_the_heap() -> Result<()> {
    with_rhi(|rhi| {
        let quad = TexturedQuad::new(rhi)?;
        let sampler = TexturedQuad::sampler(rhi)?;
        let push = quad.push(sampler.index());
        let pixels = render(rhi, |encoder, target| quad.draw(encoder, target, &push))?;
        quad.assert_drawn(&pixels);
        Ok(())
    })
}

/// A depth-only pipeline writing the depth of fills to targets of `format`.
fn depth_only_pipeline(rhi: &Rhi, format: Format) -> Result<GraphicsPipeline<Fill>> {
    let vertex = rhi.create_shader::<FillVs>()?;
    rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
        label: "depth only",
        vertex: &vertex,
        fragment: None,
        topology: Topology::default(),
        cull_mode: CullMode::default(),
        front_face: FrontFace::default(),
        color_targets: &[],
        depth: Some(DepthState {
            format,
            write: true,
            compare: CompareOp::Always,
            bias: DepthBias::default(),
        }),
        samples: SampleCount::X1,
    })
}

/// Renders the 2x2 depth texture `shadow_map` with `depth_only`: `near` on
/// its diagonal from the top left, `far` off it. The pass leaves it at rest,
/// sampled.
fn render_shadow_map(
    encoder: &mut CommandEncoder<'_>,
    shadow_map: &Image,
    depth_only: &GraphicsPipeline<Fill>,
    (near, far): (f32, f32),
) -> Result<()> {
    let mut pass = encoder.render_pass(&RenderPassDesc {
        label: "shadow map",
        colors: &[],
        depth: Some(DepthAttachment {
            image: shadow_map,
            load: LoadOp::Clear(far),
            store: StoreOp::Store,
        }),
    })?;
    for texel in 0..2 {
        pass.set_scissor(Rect {
            x: texel,
            y: texel,
            width: 1,
            height: 1,
        })?;
        let fill = Fill {
            color: RED,
            depth: near,
        };
        pass.draw(depth_only, &fill, 0..3, 0..1)?;
    }
    pass.end();
    Ok(())
}

#[test]
fn comparison_samplers_filter_depth_comparisons() -> Result<()> {
    with_rhi(|rhi| {
        let (near, far) = (0.25, 0.75);
        let shadow_map = rhi.create_image(&ImageDesc {
            label: "shadow map",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(2, 2),
            format: Format::Depth32Float,
            usage: ImageUsage::SAMPLED | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        })?;
        let view = shadow_map.view().expect("sampled images have a view");
        assert!(view.index().is_none(), "depth views are not color textures");
        let depth_only = depth_only_pipeline(rhi, Format::Depth32Float)?;
        // The vertex shader reads no push data.
        let shaders = (
            rhi.create_shader::<CoveringVs>()?.with_push(),
            rhi.create_shader::<DepthCompareFs>()?,
        );
        let pipeline = pipeline(
            rhi,
            shaders,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let mut compare = |filter| {
            let sampler = rhi.create_sampler(&SamplerDesc {
                label: "shadow",
                mag_filter: filter,
                min_filter: filter,
                compare: Some(CompareOp::Less),
                ..SamplerDesc::default()
            })?;
            let push = DepthCompare {
                texture: view
                    .depth_index()
                    .expect("sampled depth images have an index"),
                sampler: sampler.index(),
                reference: 0.5,
            };
            render(rhi, |encoder, target| {
                render_shadow_map(encoder, &shadow_map, &depth_only, (near, far))?;
                let mut pass = encoder.render_pass(&RenderPassDesc {
                    label: "depth comparison",
                    colors: &[cleared(target)],
                    depth: None,
                })?;
                pass.draw(&pipeline, &push, 0..3, 0..1)
            })
        };

        // Pixels pass where the reference is nearer than their texel: off
        // the diagonal.
        let nearest = compare(Filter::Nearest)?;
        for (index, pixel) in nearest.iter().enumerate() {
            let (x, y) = (index % SIZE as usize, index / SIZE as usize);
            let diagonal = (x < SIZE as usize / 2) == (y < SIZE as usize / 2);
            assert_eq!(pixel[0], if diagonal { 0 } else { 255 }, "pixel ({x}, {y})");
        }
        // Linear filtering blends the comparisons of neighbouring texels.
        let linear = compare(Filter::Linear)?;
        assert!(
            (1..255).contains(&linear[3 * SIZE as usize + 3][0]),
            "pixel (3, 3) blends passing and failing texels"
        );
        Ok(())
    })
}

#[test]
fn shaders_read_frame_allocations_through_the_heap() -> Result<()> {
    with_rhi(|rhi| {
        // The fragment shader reads no push data.
        let shaders = (
            rhi.create_shader::<SlicedVs>()?,
            rhi.create_shader::<ColorFs>()?.with_push(),
        );
        let pipeline = pipeline(
            rhi,
            shaders,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        // Two triangles covering the target.
        let quad = [(0, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 1)].map(|(x, y): (u8, u8)| {
            let corner = |at| f32::from(at) * 2.0 - 1.0;
            Vertex {
                position: [corner(x), corner(y)],
                uv: [0.0; 2],
            }
        });
        let colors = [RED, GREEN].map(|color| u32::from_le_bytes(unorm(color)));

        let pixels = render(rhi, |encoder, target| {
            let frame = encoder.frame();
            // An earlier allocation puts the slices past their buffer's start.
            frame.alloc(&[0_u8; 3])?;
            let (vertices, colors) = (frame.alloc(&quad)?, frame.alloc(&colors)?);
            let push = Sliced {
                vertices: vertices.index(),
                colors: colors.index(),
            };
            assert!(push.vertices.first() > 0);
            assert_eq!(u64::from(push.colors.first()) * 4, colors.offset());
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "sliced quad",
                colors: &[cleared(target)],
                depth: None,
            })?;
            // The second instance reads the second color.
            pass.draw(&pipeline, &push, 0..6, 1..2)
        })?;
        assert!(pixels.iter().all(|&pixel| pixel == unorm(GREEN)));
        Ok(())
    })
}

#[test]
fn read_only_draws_read_storage_that_dispatches_wrote() -> Result<()> {
    with_rhi(|rhi| {
        // Pipelines writing storage declare it to Metal encoders;
        // read-only draws reach it through the heaps alone.
        let (count, paint) = (
            rhi.create_shader::<CountCs>()?,
            rhi.create_shader::<PaintCs>()?,
        );
        assert!(count.writes_storage() && paint.writes_storage());
        let shaders = (
            rhi.create_shader::<CoveringVs>()?.with_push(),
            rhi.create_shader::<StoredFs>()?,
        );
        assert!(!shaders.0.writes_storage() && !shaders.1.writes_storage());
        let stored = pipeline(
            rhi,
            shaders,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let count = rhi.create_compute_pipeline(&ComputePipelineDesc {
            label: "count",
            shader: &count,
        })?;
        let paint = rhi.create_compute_pipeline(&ComputePipelineDesc {
            label: "paint",
            shader: &paint,
        })?;
        let numbers = upload(rhi, "numbers", BufferUsage::STORAGE, &[0; 8])?;
        let image = rhi.create_image(&ImageDesc {
            label: "painted",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(1, 1),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::STORAGE,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        })?;
        let image_index = image
            .view()
            .and_then(ImageView::storage_index)
            .expect("storage images have an index");
        let numbers_index = numbers.index().expect("storage buffers have an index");

        let pixels = render(rhi, |encoder, target| {
            let push = Count {
                output: numbers_index,
                base: 0,
            };
            encoder.dispatch(&count, &push, [1, 1, 1])?;
            let push = Paint {
                image: image_index,
                color: RED,
            };
            encoder.dispatch(&paint, &push, [1, 1, 1])?;
            let written = ShaderStages::COMPUTE;
            // SAFETY: the dispatches wrote both last; the image stays in its
            // resting access.
            unsafe {
                encoder.barrier(
                    &[BufferBarrier {
                        buffer: &numbers,
                        before: BufferAccess::ShaderWrite(written),
                        after: BufferAccess::ShaderRead(ShaderStages::FRAGMENT),
                        transfer: None,
                    }],
                    &[ImageBarrier {
                        image: &image,
                        range: ImageRange::ALL,
                        before: image.info().rest,
                        after: image.info().rest,
                        transfer: None,
                    }],
                )?;
            }
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "stored",
                colors: &[cleared(target)],
                depth: None,
            })?;
            let push = Stored {
                image: image_index,
                numbers: numbers_index,
            };
            pass.draw(&stored, &push, 0..3, 0..1)
        })?;
        // Red from the image, blue from the buffer's second number.
        assert!(pixels.iter().all(|&pixel| pixel == [255, 0, 255, 255]));
        Ok(())
    })
}

#[test]
fn draws_sample_storage_images_that_dispatches_wrote_in_place() -> Result<()> {
    with_rhi(|rhi| {
        let paint = rhi.create_compute_pipeline(&ComputePipelineDesc {
            label: "paint",
            shader: &rhi.create_shader::<PaintCs>()?,
        })?;
        let shaders = (
            rhi.create_shader::<CoveringVs>()?.with_push(),
            rhi.create_shader::<TexturedFs>()?,
        );
        let sample = pipeline(
            rhi,
            shaders,
            &[ColorTarget::new(Format::Rgba8Unorm)],
            None,
            SampleCount::X1,
        )?;
        let image = rhi.create_image(&ImageDesc {
            label: "painted",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(SIZE, SIZE),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        })?;
        // Dispatches write it in place, and draws sample it in the same
        // layout, which both its descriptors name.
        let rest = image.info().rest;
        assert_eq!(rest, ImageAccess::StorageWrite(ShaderStages::all()));
        let view = image.view().expect("sampled storage images have a view");
        let sampler = rhi.create_sampler(&SamplerDesc::default())?;
        let push = Textured {
            // Unread: the covering vertex shader pulls no vertices.
            vertices: BufferIndex::new(0),
            texture: view.index().expect("sampled images have an index"),
            sampler: sampler.index(),
        };
        let pixels = render(rhi, |encoder, target| {
            let paint_push = Paint {
                image: view.storage_index().expect("storage images have an index"),
                color: RED,
            };
            encoder.dispatch(&paint, &paint_push, [1, 1, 1])?;
            // SAFETY: the dispatch wrote the image last; it stays in its
            // resting access.
            unsafe {
                encoder.barrier(
                    &[],
                    &[ImageBarrier {
                        image: &image,
                        range: ImageRange::ALL,
                        before: rest,
                        after: rest,
                        transfer: None,
                    }],
                )?;
            }
            let mut pass = encoder.render_pass(&RenderPassDesc {
                label: "sample",
                colors: &[cleared(target)],
                depth: None,
            })?;
            pass.draw(&sample, &push, 0..3, 0..1)
        })?;
        assert!(pixels.iter().all(|&pixel| pixel == unorm(RED)));
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

/// Push data reading a sampler through the heap, as frame allocations hold
/// it.
#[derive(Clone, Copy, NoUninit, HeapIndices)]
#[repr(C)]
struct Material {
    sampler: SamplerIndex,
}

/// Checks that the slot of `sampler`, dropped before the last frame, which
/// read it, is reused only once that frame completed.
fn assert_freed_after_the_last_frame(rhi: &mut Rhi, sampler: SamplerIndex) -> Result<()> {
    let frame = rhi.begin_frame()?;
    let other = frame.rhi().create_sampler(&SamplerDesc::default())?;
    assert_ne!(other.index(), sampler, "the slot is reused while read");
    frame.finish();
    rhi.begin_frame()?.finish();
    let next = rhi.create_sampler(&SamplerDesc::default())?;
    assert_eq!(
        next.index(),
        sampler,
        "the slot is reused once its work completed"
    );
    Ok(())
}

#[test]
fn draws_keep_the_resources_they_index_until_they_complete() -> Result<()> {
    with_rhi(|rhi| {
        let quad = TexturedQuad::new(rhi)?;
        let sampler = TexturedQuad::sampler(rhi)?;
        let push = quad.push(sampler.index());
        // Dropped before the frame drawing with it.
        drop(sampler);
        let pixels = render(rhi, |encoder, target| quad.draw(encoder, target, &push))?;
        quad.assert_drawn(&pixels);
        assert_freed_after_the_last_frame(rhi, push.sampler)
    })
}

#[test]
fn frame_allocations_keep_the_resources_they_index_until_they_complete() -> Result<()> {
    with_rhi(|rhi| {
        let sampler = rhi.create_sampler(&SamplerDesc::default())?.index();
        let frame = rhi.begin_frame()?;
        frame.alloc(&[Material { sampler }])?;
        frame.finish();
        assert_freed_after_the_last_frame(rhi, sampler)
    })
}

#[test]
fn freed_slots_read_again_are_reused_once_that_work_completes() -> Result<()> {
    with_rhi(|rhi| {
        let sampler = rhi.create_sampler(&SamplerDesc::default())?.index();
        for _ in 0..=FRAMES_IN_FLIGHT {
            rhi.begin_frame()?.finish();
        }
        // The slot is free now; a stale index reads it again.
        let frame = rhi.begin_frame()?;
        frame.alloc(&[Material { sampler }])?;
        frame.finish();
        assert_freed_after_the_last_frame(rhi, sampler)
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
        // Shaders write it in place, and sample it in the same layout.
        assert_eq!(
            flat.info().rest,
            ImageAccess::StorageWrite(ShaderStages::all())
        );
        let target = image(ImageDimension::D2, 1, ImageUsage::COLOR_ATTACHMENT)?;
        assert_eq!(indices(&target), (false, false));
        // Cubes are only sampled through the heap.
        let cube = image(ImageDimension::Cube, 6, both)?;
        assert_eq!(indices(&cube), (true, false));
        Ok(())
    })
}
