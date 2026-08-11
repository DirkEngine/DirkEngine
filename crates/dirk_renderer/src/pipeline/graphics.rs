use std::marker::PhantomData;

use dirk_rhi::{
    Backend as _, BindGroupLayoutDesc, BindGroupLayoutEntry, CommandBuffer as _, CompareOp,
    CullMode, DepthState, FrontFace, GraphicsPipelineDesc, PipelineLayoutDesc, PrimitiveTopology,
    RasterState,
};
use tracing::debug;

use crate::{
    Error, Result,
    resources::{
        ActiveBindGroup, ActiveGraphicsPipeline, ActivePipelineLayout,
        buffer::VertexBuffer,
        command_pool::CommandBuffer,
        descriptors::{DescriptorSet, layouts::SetLayout},
        device::RenderDevice,
    },
    shaders::metadata::{FragmentShader, Shader as _, VertexInput, VertexShader},
};

/// Host-side contract for a graphics pipeline's resource types.
pub trait GraphicsPipelineSpec {
    type VertexShader: VertexShader;
    type FragmentShader: FragmentShader;
    type Input: VertexInput;
    type DescriptorSets: DescriptorSetInput;

    const NAME: &'static str;

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
    const SET_COUNT: usize;
    const BINDINGS: &'static [&'static [BindGroupLayoutEntry]];

    type Refs<'a>
    where
        Self: 'a;

    fn groups<'a>(sets: &'a Self::Refs<'a>) -> Vec<&'a ActiveBindGroup>;
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

            #[allow(non_snake_case)]
            fn groups<'a>(sets: &'a Self::Refs<'a>) -> Vec<&'a ActiveBindGroup> {
                let ($($name,)+) = sets;
                vec![$($name.group()),+]
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

pub struct GraphicsPipeline<Spec: GraphicsPipelineSpec> {
    layout: ActivePipelineLayout,
    pipeline: ActiveGraphicsPipeline,
    _spec: PhantomData<Spec>,
}

/// Typed rendering context for a bound graphics pipeline.
pub struct GraphicsPipelineRenderingContext<'cmd, Spec: GraphicsPipelineSpec> {
    command: &'cmd mut CommandBuffer,
    layout: &'cmd ActivePipelineLayout,
    _spec: PhantomData<Spec>,
}

impl<Spec: GraphicsPipelineSpec> GraphicsPipeline<Spec> {
    pub fn build(device: &RenderDevice) -> Result<Self> {
        Spec::validate()?;
        let reflected =
            merge_shader_set_layouts::<Spec::VertexShader, Spec::FragmentShader>(Spec::NAME)?;
        let bind_group_layouts = reflected
            .iter()
            .map(|entries| {
                Ok(device.rhi.create_bind_group_layout(&BindGroupLayoutDesc {
                    label: Spec::NAME,
                    entries,
                })?)
            })
            .collect::<Result<Vec<_>>>()?;
        let layout_refs = bind_group_layouts.iter().collect::<Vec<_>>();
        let layout = device.rhi.create_pipeline_layout(&PipelineLayoutDesc {
            label: Spec::NAME,
            bind_group_layouts: &layout_refs,
        })?;
        let vertex = Spec::VertexShader::create(device)?;
        let fragment = Spec::FragmentShader::create(device)?;
        let vertex_layout = Spec::Input::layout();
        let pipeline = device.rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            label: Spec::NAME,
            layout: &layout,
            vertex: &vertex,
            fragment: &fragment,
            vertex_buffers: &[vertex_layout],
            raster: RasterState {
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::CounterClockwise,
                cull_mode: CullMode::Back,
            },
            color_formats: &[device.properties.surface_format],
            depth: Some(DepthState {
                format: device.properties.depth_format,
                write_enabled: true,
                compare: CompareOp::Less,
            }),
            samples: device.properties.msaa_samples,
        })?;

        Ok(Self {
            layout,
            pipeline,
            _spec: PhantomData,
        })
    }

    pub fn bind<'cmd>(
        &'cmd self,
        command: &'cmd mut CommandBuffer,
    ) -> GraphicsPipelineRenderingContext<'cmd, Spec> {
        command.rhi_mut().bind_graphics_pipeline(&self.pipeline);
        GraphicsPipelineRenderingContext {
            command,
            layout: &self.layout,
            _spec: PhantomData,
        }
    }
}

impl<Spec: GraphicsPipelineSpec> GraphicsPipelineRenderingContext<'_, Spec> {
    pub fn bind_descriptor_sets<'a>(
        &mut self,
        sets: &'a <Spec::DescriptorSets as DescriptorSetInput>::Refs<'a>,
    ) {
        let groups = Spec::DescriptorSets::groups(sets);
        self.command.rhi_mut().bind_groups(self.layout, 0, &groups);
    }

    pub fn bind_vertex_buffer(&mut self, vertex_buffer: &VertexBuffer<Spec::Input>) {
        self.command
            .rhi_mut()
            .bind_vertex_buffer(0, vertex_buffer.buffer(), 0);
    }

    pub fn command(&mut self) -> &mut CommandBuffer {
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
                return Err(Error::PipelineDescriptorLayoutMismatch { pipeline, set });
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
            return Err(Error::PipelineDescriptorLayoutMismatch { pipeline, set });
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
        Err(Error::PipelineVertexInputMismatch { pipeline })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::MainPipelineSpec;

    #[test]
    fn main_pipeline_contract_matches_reflection() {
        let reflected = merge_shader_set_layouts::<
            <MainPipelineSpec as GraphicsPipelineSpec>::VertexShader,
            <MainPipelineSpec as GraphicsPipelineSpec>::FragmentShader,
        >(MainPipelineSpec::NAME)
        .unwrap();
        validate_pipeline_descriptor_layout::<MainPipelineSpec>(MainPipelineSpec::NAME, &reflected)
            .unwrap();
        validate_pipeline_vertex_input::<MainPipelineSpec>(MainPipelineSpec::NAME).unwrap();
    }
}
