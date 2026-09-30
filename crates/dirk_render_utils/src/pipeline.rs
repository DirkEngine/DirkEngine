//! Typed graphics pipelines with reflected shader interface validation.
//!
//! The merge and validation helpers are public so renderers building pipelines
//! outside [`GraphicsPipeline`] share one implementation of these checks.
use std::marker::PhantomData;

use dirk_rhi::{
    BindGroupLayoutDesc, BindGroupLayoutEntry, BlendState, ColorTargetState, ColorWrites, CullMode,
    DepthBiasState, DepthState, FrontFace, GraphicsPipelineDesc, IndexFormat, PipelineLayoutDesc,
    PrimitiveTopology, RasterState, SampleCount, VertexBufferLayout,
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
        validate_descriptor_layouts(
            Self::NAME,
            <Self::DescriptorSets as DescriptorSetInput>::BINDINGS,
            &reflected,
        )?;
        validate_vertex_input(
            Self::NAME,
            &Self::Input::layout(),
            Self::VertexShader::INPUT_LAYOUTS,
        )
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

/// Merges the reflected vertex and fragment layouts of every descriptor set.
///
/// # Errors
/// Rejects a binding reflected with different types by the two stages.
pub fn merge_shader_set_layouts<V: VertexShader, F: FragmentShader>(
    pipeline: &str,
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

/// Merges one set's per-stage bindings, sorted by binding with unioned visibility.
///
/// # Errors
/// Rejects a binding reflected with different types by the two stages.
pub fn merge_descriptor_set_layout(
    pipeline: &str,
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

/// Checks host set declarations against merged reflected layouts, set by set.
///
/// `expected` bindings must be declared in binding order with the merged visibility.
///
/// # Errors
/// Rejects any set whose declared bindings differ from the reflected ones.
pub fn validate_descriptor_layouts(
    pipeline: &str,
    expected: &[&[BindGroupLayoutEntry]],
    reflected: &[Vec<BindGroupLayoutEntry>],
) -> Result<()> {
    for set in 0..reflected.len().max(expected.len()) {
        let actual = reflected.get(set).map(Vec::as_slice).unwrap_or_default();
        let expected = expected.get(set).copied().unwrap_or_default();
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

/// Checks a host vertex record against the vertex shader's reflected inputs.
///
/// # Errors
/// Rejects anything other than exactly one reflected buffer matching `expected`.
pub fn validate_vertex_input(
    pipeline: &str,
    expected: &VertexBufferLayout<'_>,
    reflected: &[VertexBufferLayout<'_>],
) -> Result<()> {
    let matches = matches!(reflected, [actual] if actual.stride == expected.stride
        && actual.step_mode == expected.step_mode
        && actual.attributes == expected.attributes);
    if matches {
        Ok(())
    } else {
        debug!(
            pipeline,
            ?expected,
            ?reflected,
            "pipeline vertex input mismatch"
        );
        anyhow::bail!("pipeline {pipeline}: vertex input layout mismatch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{Shader, ShaderCode};
    use dirk_rhi::{BindingType, ShaderStage, ShaderStages, VertexAttribute, VertexFormat};

    const UNIFORM: BindingType = BindingType::UniformBuffer {
        dynamic_offset: false,
    };

    const fn entry(
        binding: u32,
        ty: BindingType,
        visibility: ShaderStages,
    ) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            ty,
            visibility,
        }
    }

    const CAMERA_VERTEX: &[BindGroupLayoutEntry] = &[entry(0, UNIFORM, ShaderStages::VERTEX)];
    const MATERIAL_FRAGMENT: &[BindGroupLayoutEntry] = &[
        entry(1, BindingType::SampledImage, ShaderStages::FRAGMENT),
        entry(0, UNIFORM, ShaderStages::FRAGMENT),
    ];
    const MERGED: &[BindGroupLayoutEntry] = &[
        entry(
            0,
            UNIFORM,
            ShaderStages::VERTEX.union(ShaderStages::FRAGMENT),
        ),
        entry(1, BindingType::SampledImage, ShaderStages::FRAGMENT),
    ];
    const POSITION: &[VertexAttribute] = &[VertexAttribute {
        location: 0,
        format: VertexFormat::Float32x3,
        offset: 0,
    }];

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::NoUninit)]
    struct Position([f32; 3]);
    impl VertexInput for Position {
        const ATTRIBUTES: &'static [VertexAttribute] = POSITION;
    }

    struct Set;
    impl SetLayout for Set {
        const BINDINGS: &'static [BindGroupLayoutEntry] = MERGED;
    }

    const NO_CODE: ShaderCode = ShaderCode {
        #[cfg(not(target_vendor = "apple"))]
        spirv: &[],
        #[cfg(target_vendor = "apple")]
        msl: "",
    };

    struct Vertex;
    // SAFETY: never created; only the reflected metadata is inspected.
    unsafe impl Shader for Vertex {
        const CODE: ShaderCode = NO_CODE;
        const ENTRYPOINT: &'static str = "main";
        const STAGE: ShaderStage = ShaderStage::Vertex;
        const SET_LAYOUTS: &'static [&'static [BindGroupLayoutEntry]] = &[CAMERA_VERTEX];
    }
    impl VertexShader for Vertex {
        const INPUT_LAYOUTS: &'static [VertexBufferLayout<'static>] = &[VertexBufferLayout {
            stride: 12,
            step_mode: dirk_rhi::VertexStepMode::Vertex,
            attributes: POSITION,
        }];
    }

    struct Fragment;
    // SAFETY: never created; only the reflected metadata is inspected.
    unsafe impl Shader for Fragment {
        const CODE: ShaderCode = NO_CODE;
        const ENTRYPOINT: &'static str = "main";
        const STAGE: ShaderStage = ShaderStage::Fragment;
        const SET_LAYOUTS: &'static [&'static [BindGroupLayoutEntry]] = &[MATERIAL_FRAGMENT];
    }
    impl FragmentShader for Fragment {}

    struct Spec;
    impl GraphicsPipelineSpec for Spec {
        type VertexShader = Vertex;
        type FragmentShader = Fragment;
        type Input = Position;
        type DescriptorSets = (Set,);
        const NAME: &'static str = "test pipeline";
    }

    #[test]
    fn merged_layouts_union_visibility_in_binding_order() -> Result<()> {
        assert_eq!(
            merge_shader_set_layouts::<Vertex, Fragment>("merge")?,
            [MERGED]
        );
        let conflicting = [entry(0, BindingType::SampledImage, ShaderStages::FRAGMENT)];
        let error = merge_descriptor_set_layout("merge", 0, CAMERA_VERTEX, &conflicting)
            .expect_err("conflicting binding types");
        assert!(
            error
                .to_string()
                .contains("descriptor layout mismatch in set 0")
        );
        Ok(())
    }

    #[test]
    fn descriptor_layouts_must_match_every_reflected_set() {
        let reflected = [MERGED.to_vec()];
        assert!(validate_descriptor_layouts("sets", &[MERGED], &reflected).is_ok());
        for expected in [&[][..], &[CAMERA_VERTEX], &[MERGED, CAMERA_VERTEX]] {
            let error = validate_descriptor_layouts("sets", expected, &reflected)
                .expect_err("mismatched descriptor sets");
            assert!(error.to_string().contains("descriptor layout mismatch"));
        }
    }

    #[test]
    fn vertex_input_must_match_the_single_reflected_buffer() {
        let expected = Position::layout();
        assert!(validate_vertex_input("input", &expected, Vertex::INPUT_LAYOUTS).is_ok());
        let wider = VertexBufferLayout {
            stride: 16,
            ..expected
        };
        for reflected in [&[][..], &[wider], &[expected, expected]] {
            let error = validate_vertex_input("input", &expected, reflected)
                .expect_err("mismatched vertex input");
            assert!(error.to_string().contains("vertex input layout mismatch"));
        }
    }

    #[test]
    fn specs_validate_reflected_interfaces() {
        assert!(Spec::validate().is_ok());
    }
}
