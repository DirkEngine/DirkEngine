//! Headless Vulkan regressions. Run explicitly with a Vulkan 1.3 driver and
//! `VK_LAYER_KHRONOS_validation` installed:
//! `cargo nextest run -p dirk_rhi --run-ignored only`.
#![cfg(not(target_vendor = "apple"))]
#![allow(
    unsafe_code,
    reason = "controlled shaders, recording, and loader access in native tests"
)]

use dirk_rhi::*;
use std::ffi::CStr;

struct TestDevice {
    rhi: Rhi,
}

impl TestDevice {
    fn new() -> Result<Self> {
        // SAFETY: no borrowed loader symbols escape this scope.
        let entry = unsafe { ash::Entry::load() }.expect("Vulkan loader must be installed");
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("Vulkan layers can be enumerated");
        assert!(
            layers.iter().any(|layer| unsafe {
                CStr::from_ptr(layer.layer_name.as_ptr()) == c"VK_LAYER_KHRONOS_validation"
            }),
            "native regression tests require the Khronos validation layer"
        );
        Ok(Self {
            rhi: Rhi::new(&RhiCreateInfo {
                application_name: "native validation test",
                application_version: (0, 0, 1),
                engine_name: "DirkEngine",
                engine_version: (0, 0, 1),
                validation: true,
                compatible_surface: None,
            })?,
        })
    }

    fn check(&mut self) -> Result<()> {
        self.rhi.flush()?;
        assert_eq!(
            self.rhi.validation_error_count(),
            0,
            "native validation reported an error"
        );
        Ok(())
    }

    fn pipeline(
        &self,
        color_targets: &[ColorTargetState],
        depth: Option<DepthState>,
    ) -> Result<GraphicsPipeline> {
        let code = Self::vertex_words();
        // SAFETY: this shader only writes a constant position and has no inputs or bindings.
        let vertex = unsafe {
            self.rhi.create_shader(&ShaderDesc {
                label: "constant position",
                stage: ShaderStage::Vertex,
                entry: "main",
                source: ShaderSource::SpirV(&code),
            })?
        };
        let layout = self.rhi.create_pipeline_layout(&PipelineLayoutDesc {
            label: "empty",
            bind_group_layouts: &[],
        })?;
        self.rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: "validation pipeline",
            layout: &layout,
            vertex: &vertex,
            fragment: None,
            vertex_buffers: &[],
            raster: RasterState::default(),
            color_targets,
            depth,
            depth_bias: DepthBiasState::default(),
            primitive_restart: None,
            alpha_to_coverage: false,
            samples: SampleCount::One,
        })
    }

    fn record_depth_pass(
        &self,
        pipeline: &GraphicsPipeline,
        image: &Image,
        view: &ImageView,
    ) -> Result<()> {
        let mut encoder = self.rhi.create_encoder::<Graphics>("stencil")?;
        // SAFETY: a fresh image is transitioned before use; the constant shader has
        // no input/resource accesses. These commands are discarded without submission.
        unsafe {
            encoder.barrier(&DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[ImageBarrier {
                    image,
                    old_state: ImageState::Undefined,
                    new_state: ImageState::DepthStencilAttachment,
                    aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    queue_transfer: None,
                }],
            })?;
            let mut pass = encoder.begin_render_pass(&RenderingInfo {
                label: "stencil",
                width: 16,
                height: 16,
                layer_count: 1,
                color_attachments: &[],
                depth_attachment: Some(DepthAttachment {
                    view,
                    depth_load: LoadOp::Clear(1.0),
                    depth_store: StoreOp::Store,
                    stencil_load: LoadOp::Clear(0),
                    stencil_store: StoreOp::Store,
                }),
            })?;
            pass.bind_graphics_pipeline(pipeline)?;
            pass.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 16.0,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            pass.set_scissor(Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            })?;
            pass.set_stencil_reference(1, 1)?;
            pass.set_blend_constants(Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            })?;
            pass.draw(3, 1, 0, 0)?;
        }
        drop(encoder.finish()?);
        Ok(())
    }

    // SPIR-V 1.0 for `void main() { gl_Position = vec4(0, 0, 0, 1); }`.
    fn vertex_words() -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 12, 0];
        for (opcode, operands) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![0, 4, 0x6e69_616d, 0, 9]),
            (71, vec![9, 11, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (22, vec![3, 32]),
            (23, vec![5, 3, 4]),
            (32, vec![6, 3, 5]),
            (43, vec![3, 7, 0]),
            (43, vec![3, 8, 0x3f80_0000]),
            (44, vec![5, 10, 7, 7, 7, 8]),
            (59, vec![6, 9, 3]),
            (54, vec![1, 4, 0, 2]),
            (248, vec![11]),
            (62, vec![9, 10]),
            (253, vec![]),
            (56, vec![]),
        ] {
            words.push(
                ((u32::try_from(operands.len()).expect("short instruction") + 1) << 16) | opcode,
            );
            words.extend(operands);
        }
        words
    }
}

#[test]
#[ignore = "requires Vulkan 1.3 and Khronos validation"]
fn headless_device_and_cube_array_view_are_valid() -> Result<()> {
    let mut test = TestDevice::new()?;
    {
        let image = test.rhi.create_image(&ImageDesc {
            label: "cube array",
            dimension: ImageDimension::Cube,
            extent: Extent3d::new_2d(16, 16),
            format: TextureFormat::Rgba8Unorm,
            usage: ImageUsages::SAMPLED,
            mip_levels: 1,
            array_layers: 12,
            samples: SampleCount::One,
        })?;
        match test.rhi.view(&image) {
            Ok(_)
            | Err(Error::Unsupported(UnsupportedOperation::Capability("cube array image views"))) =>
                {}
            Err(error) => return Err(error),
        }
    }
    test.check()
}

#[test]
#[ignore = "requires Vulkan 1.3 and Khronos validation"]
fn uniform_binding_limit_applies_to_resolved_ranges() -> Result<()> {
    let mut test = TestDevice::new()?;
    {
        let limits = test.rhi.capabilities().limits;
        let max_range = limits.max_uniform_buffer_binding_size;
        assert!(
            limits.max_buffer_size > max_range,
            "test requires an allocation larger than a uniform binding"
        );
        let buffer = test.rhi.create_buffer(&BufferDesc {
            label: "oversized uniform allocation",
            size: max_range + 1,
            usage: BufferUsages::UNIFORM,
            memory: MemoryDomain::Upload,
        })?;
        for dynamic_offset in [false, true] {
            let layout = test.rhi.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "uniform layout",
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::UniformBuffer { dynamic_offset },
                }],
            })?;
            for size in [max_range, max_range + 1, u64::MAX] {
                let result = test.rhi.create_bind_group(&BindGroupDesc {
                    label: "uniform group",
                    layout: &layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &buffer,
                            offset: 0,
                            size,
                        },
                    }],
                });
                if size == max_range {
                    drop(result?);
                } else {
                    assert!(
                        matches!(result, Err(Error::InvalidResource(ref error)) if error.kind() == InvalidResourceKind::OutOfRange)
                    );
                }
            }
        }
    }
    test.check()
}

#[test]
#[ignore = "requires Vulkan 1.3 and Khronos validation"]
fn independent_color_write_masks_are_enabled_or_rejected() -> Result<()> {
    let mut test = TestDevice::new()?;
    let targets = [
        ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: ColorWrites::ALL,
        },
        ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: ColorWrites::RED,
        },
    ];
    match test.pipeline(&targets, None) {
        Ok(_)
        | Err(Error::Unsupported(UnsupportedOperation::Capability(
            "independent color attachment blending",
        ))) => {}
        Err(error) => return Err(error),
    }
    test.check()
}

#[test]
#[ignore = "requires Vulkan 1.3 and Khronos validation"]
fn depth_stencil_pipeline_matches_rendering_attachment() -> Result<()> {
    let mut test = TestDevice::new()?;
    {
        let format = test
            .rhi
            .supported_depth_formats()
            .iter()
            .copied()
            .find(|format| format.aspects().contains(ImageAspects::STENCIL))
            .expect("test requires a depth/stencil format");
        let face = StencilFaceState {
            compare: CompareOp::Always,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Replace,
        };
        let depth = DepthState {
            format,
            write_enabled: true,
            compare: CompareOp::Always,
            stencil: Some(StencilState {
                front: face,
                back: face,
                read_mask: 255,
                write_mask: 255,
            }),
        };
        let image = test.rhi.create_image(&ImageDesc {
            label: "stencil target",
            dimension: ImageDimension::TwoD,
            extent: Extent3d::new_2d(16, 16),
            format,
            usage: ImageUsages::DEPTH_STENCIL_ATTACHMENT,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::One,
        })?;
        let view = test.rhi.view(&image)?;
        let depth_only = test.rhi.create_image_view(&ImageViewDesc {
            label: "depth only",
            image: &image,
            view_type: ImageViewType::TwoD,
            aspects: ImageAspects::DEPTH,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        })?;
        let mut rejected = test.rhi.create_encoder::<Graphics>("partial attachment")?;
        // SAFETY: validation rejects the incompatible aspect selection before recording.
        let result = unsafe {
            rejected.begin_render_pass(&RenderingInfo {
                label: "partial attachment",
                width: 16,
                height: 16,
                layer_count: 1,
                color_attachments: &[],
                depth_attachment: Some(DepthAttachment {
                    view: &depth_only,
                    depth_load: LoadOp::Clear(1.0),
                    depth_store: StoreOp::Store,
                    stencil_load: LoadOp::Clear(0),
                    stencil_store: StoreOp::Store,
                }),
            })
        };
        assert!(
            matches!(result, Err(Error::InvalidResource(ref error)) if error.kind() == InvalidResourceKind::Mismatch)
        );
        drop(result);
        drop(rejected.finish()?);
        let invalid_stencil = test.pipeline(
            &[],
            Some(DepthState {
                format: TextureFormat::Depth32Float,
                ..depth
            }),
        );
        assert!(
            matches!(invalid_stencil, Err(Error::InvalidResource(ref error)) if error.kind() == InvalidResourceKind::Mismatch)
        );
        // Stencil format must match the attachment even when stencil testing is disabled.
        for stencil in [depth.stencil, None] {
            let pipeline = test.pipeline(&[], Some(DepthState { stencil, ..depth }))?;
            test.record_depth_pass(&pipeline, &image, &view)?;
        }
    }
    test.check()
}
