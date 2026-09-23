//! Typed graphics pipelines with reflected shader interface validation.
use std::marker::PhantomData;

use dirk_rhi::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BlendState, ColorTargetState, ColorWrites, CullMode,
    DepthBiasState, DepthState, FrontFace, GraphicsPipelineDesc, IndexFormat, PipelineLayoutDesc,
    PrimitiveTopology, RasterState, SampleCount,
};
use tracing::debug;

use crate::{
    binding::{DescriptorSet, SetLayout},
    buffer::{VertexBuffer, VertexInput},
    shader::{FragmentShader, Shader as _, VertexShader},
};
use anyhow::Result;
use dirk_rhi::{
    BindGroup, GraphicsPipeline as RhiGraphicsPipeline, PipelineLayout, RenderPass, Rhi,
};
/// Output compatibility chosen by the renderer when creating a pipeline.
#[derive(Clone, Copy)]
pub struct PipelineSettings {
    /// Color output format.
    pub color_format: dirk_rhi::TextureFormat,
    /// Depth/stencil behavior, if present.
    pub depth: Option<DepthState>,
    /// Raster sample count.
    pub samples: SampleCount,
}

/// Host-side contract for a graphics pipeline's resource types.
pub trait GraphicsPipelineSpec {
    /// Vertex-stage shader and reflected metadata.
    type VertexShader: VertexShader;
    /// Fragment-stage shader and reflected metadata.
    type FragmentShader: FragmentShader;
    /// Host vertex record.
    type Input: VertexInput;
    /// Tuple of bind-group layout types in set order.
    type DescriptorSets: DescriptorSetInput;

    /// Diagnostic label.
    const NAME: &'static str;

    /// Rasterization settings.
    #[must_use]
    fn raster() -> RasterState {
        RasterState {
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::CounterClockwise,
            cull_mode: CullMode::Back,
        }
    }

    /// Optional color blending.
    #[must_use]
    fn blend() -> Option<BlendState> {
        None
    }

    /// Depth bias settings.
    #[must_use]
    fn depth_bias() -> DepthBiasState {
        DepthBiasState::default()
    }

    /// Index format permitting strip restart, if enabled.
    #[must_use]
    fn primitive_restart() -> Option<IndexFormat> {
        None
    }

    /// Whether alpha controls sample coverage.
    #[must_use]
    fn alpha_to_coverage() -> bool {
        false
    }

    /// Checks reflected shader interfaces against the host types.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    fn validate() -> Result<()>
    where
        Self: Sized,
    {
        let reflected =
            merge_shader_set_layouts::<Self::VertexShader, Self::FragmentShader>(Self::NAME)?;
        validate_pipeline_descriptor_layout::<Self>(Self::NAME, &reflected)?;
        validate_pipeline_vertex_input::<Self>(Self::NAME)
    }
}

/// Bind-group tuple metadata and typed references.
pub trait DescriptorSetInput {
    /// Number of descriptor sets.
    const SET_COUNT: usize;
    /// Bindings for each set.
    const BINDINGS: &'static [&'static [BindGroupLayoutEntry]];

    /// Borrowed typed sets used by a draw.
    type Refs<'a>
    where
        Self: 'a;

    /// Borrows the underlying groups without allocating a temporary vector.
    fn with_groups<'a, T>(
        sets: &'a Self::Refs<'a>,
        use_groups: impl FnOnce(&[&BindGroup]) -> T,
    ) -> T;
}

macro_rules! impl_descriptor_set_input_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> DescriptorSetInput for ($($name,)+)
        where
            $($name: SetLayout + 'static),+
        {
            const SET_COUNT: usize = [$(stringify!($name)),+].len();
            const BINDINGS: &'static [&'static [BindGroupLayoutEntry]] =
                &[$($name::BINDINGS),+];

            type Refs<'a> = ($(&'a DescriptorSet<$name>,)+) where Self: 'a;

            #[allow(non_snake_case, reason = "tuple type names are also used as destructured bindings")]
            fn with_groups<'a, T>(sets: &'a Self::Refs<'a>, use_groups: impl FnOnce(&[&BindGroup]) -> T) -> T {
                let ($($name,)+) = sets;
                use_groups(&[$($name.group()),+])
            }
        }
    };
}

impl_descriptor_set_input_for_tuple!(A);
impl_descriptor_set_input_for_tuple!(A, B);
impl_descriptor_set_input_for_tuple!(A, B, C);
impl_descriptor_set_input_for_tuple!(A, B, C, D);
impl_descriptor_set_input_for_tuple!(A, B, C, D, E);
impl_descriptor_set_input_for_tuple!(A, B, C, D, E, F);
impl_descriptor_set_input_for_tuple!(A, B, C, D, E, F, G);
impl_descriptor_set_input_for_tuple!(A, B, C, D, E, F, G, H);

/// Native pipeline and layout tagged with the host input and binding contract.
pub struct GraphicsPipeline<Spec: GraphicsPipelineSpec> {
    layout: PipelineLayout,
    pipeline: RhiGraphicsPipeline,
    _spec: PhantomData<Spec>,
}

/// Typed rendering context for a bound graphics pipeline.
pub struct GraphicsPipelineRenderingContext<'cmd, 'pass, Spec: GraphicsPipelineSpec> {
    command: &'cmd mut RenderPass<'pass>,
    layout: &'cmd PipelineLayout,
    _spec: PhantomData<Spec>,
}

impl<Spec: GraphicsPipelineSpec> GraphicsPipeline<Spec> {
    /// Creates a validated pipeline for explicit attachment formats and samples.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn build(device: &Rhi, settings: PipelineSettings) -> Result<Self> {
        Spec::validate()?;
        let reflected =
            merge_shader_set_layouts::<Spec::VertexShader, Spec::FragmentShader>(Spec::NAME)?;
        let bind_group_layouts = reflected
            .iter()
            .map(|entries| {
                Ok(device.create_bind_group_layout(&BindGroupLayoutDesc {
                    label: Spec::NAME,
                    entries,
                })?)
            })
            .collect::<Result<Vec<_>>>()?;
        let layout_refs = bind_group_layouts.iter().collect::<Vec<_>>();
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Spec::NAME,
            bind_group_layouts: &layout_refs,
        })?;
        let vertex = Spec::VertexShader::create(device)?;
        let fragment = Spec::FragmentShader::create(device)?;
        let vertex_layout = Spec::Input::layout();
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: Spec::NAME,
            layout: &layout,
            vertex: &vertex,
            fragment: Some(&fragment),
            vertex_buffers: &[vertex_layout],
            raster: Spec::raster(),
            color_targets: &[ColorTargetState {
                format: settings.color_format,
                blend: Spec::blend(),
                write_mask: ColorWrites::RED
                    | ColorWrites::GREEN
                    | ColorWrites::BLUE
                    | ColorWrites::ALPHA,
            }],
            depth: settings.depth,
            depth_bias: Spec::depth_bias(),
            primitive_restart: Spec::primitive_restart(),
            alpha_to_coverage: Spec::alpha_to_coverage(),
            samples: settings.samples,
        })?;

        Ok(Self {
            layout,
            pipeline,
            _spec: PhantomData,
        })
    }

    /// Binds the pipeline and borrows its layout for subsequent typed bindings.
    ///
    /// # Safety
    /// The pipeline must remain alive or retired in this cycle until GPU completion.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn bind<'cmd, 'pass>(
        &'cmd self,
        command: &'cmd mut RenderPass<'pass>,
    ) -> Result<GraphicsPipelineRenderingContext<'cmd, 'pass, Spec>> {
        unsafe {
            command.bind_graphics_pipeline(&self.pipeline)?;
            Ok(GraphicsPipelineRenderingContext {
                command,
                layout: &self.layout,
                _spec: PhantomData,
            })
        }
    }
}

impl<'pass, Spec: GraphicsPipelineSpec> GraphicsPipelineRenderingContext<'_, 'pass, Spec> {
    /// Binds a tuple matching the pipeline descriptor layout.
    ///
    /// # Safety
    /// All resources referenced by these non-owning groups must remain valid and have appropriate access dependencies.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn bind_descriptor_sets<'a>(
        &mut self,
        sets: &'a <Spec::DescriptorSets as DescriptorSetInput>::Refs<'a>,
    ) -> Result<()> {
        unsafe {
            Spec::DescriptorSets::with_groups(sets, |groups| {
                self.command
                    .bind_groups(self.layout, 0, groups, &[])
                    .map_err(Into::into)
            })
        }
    }

    /// Binds vertices with the pipeline input record type.
    ///
    /// # Safety
    /// The buffer must remain valid through GPU use and have synchronized vertex access.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn bind_vertex_buffer(
        &mut self,
        vertex_buffer: &VertexBuffer<Spec::Input>,
    ) -> Result<()> {
        unsafe {
            self.command
                .bind_vertex_buffer(0, vertex_buffer.buffer(), 0)
                .map_err(Into::into)
        }
    }

    /// Borrows the recording scope for draw and dynamic-state commands.
    pub fn command(&mut self) -> &mut RenderPass<'pass> {
        self.command
    }
}

fn merge_shader_set_layouts<V: VertexShader, F: FragmentShader>(
    pipeline: &'static str,
) -> Result<Vec<Vec<BindGroupLayoutEntry>>> {
    let max_sets = V::SET_LAYOUTS.len().max(F::SET_LAYOUTS.len());
    (0..max_sets)
        .map(|set| {
            merge_descriptor_set_layout(
                pipeline,
                set,
                V::SET_LAYOUTS.get(set).copied().unwrap_or_default(),
                F::SET_LAYOUTS.get(set).copied().unwrap_or_default(),
            )
        })
        .collect()
}

fn merge_descriptor_set_layout(
    pipeline: &'static str,
    set: usize,
    vertex: &[BindGroupLayoutEntry],
    fragment: &[BindGroupLayoutEntry],
) -> Result<Vec<BindGroupLayoutEntry>> {
    let mut merged = Vec::new();
    for binding in vertex.iter().chain(fragment).copied() {
        if let Some(existing) = merged
            .iter_mut()
            .find(|entry: &&mut BindGroupLayoutEntry| entry.binding == binding.binding)
        {
            if existing.ty != binding.ty {
                debug!(
                    pipeline,
                    set,
                    ?existing,
                    ?binding,
                    "pipeline binding mismatch"
                );
                anyhow::bail!("pipeline {pipeline}: descriptor layout mismatch in set {set}");
            }
            existing.visibility |= binding.visibility;
        } else {
            merged.push(binding);
        }
    }
    merged.sort_by_key(|entry| entry.binding);
    Ok(merged)
}

fn validate_pipeline_descriptor_layout<S: GraphicsPipelineSpec>(
    pipeline: &'static str,
    reflected: &[Vec<BindGroupLayoutEntry>],
) -> Result<()> {
    let max_sets = reflected.len().max(S::DescriptorSets::SET_COUNT);
    for set in 0..max_sets {
        let actual = reflected.get(set).map(Vec::as_slice).unwrap_or_default();
        let expected = S::DescriptorSets::BINDINGS
            .get(set)
            .copied()
            .unwrap_or_default();
        if expected != actual {
            debug!(
                pipeline,
                set,
                ?expected,
                ?actual,
                "pipeline descriptor layout mismatch"
            );
            anyhow::bail!("pipeline {pipeline}: descriptor layout mismatch in set {set}");
        }
    }
    Ok(())
}

fn validate_pipeline_vertex_input<S: GraphicsPipelineSpec>(pipeline: &'static str) -> Result<()> {
    let expected = S::Input::layout();
    let actual = S::VertexShader::INPUT_LAYOUTS;
    let matches = actual.len() == 1
        && actual[0].stride == expected.stride
        && actual[0].step_mode == expected.step_mode
        && actual[0].attributes == expected.attributes;
    if matches {
        Ok(())
    } else {
        debug!(pipeline, "pipeline vertex input mismatch");
        Err(anyhow::anyhow!(
            "pipeline {pipeline}: vertex input layout mismatch"
        ))
    }
}
