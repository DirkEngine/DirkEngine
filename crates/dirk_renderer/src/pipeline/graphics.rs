use std::marker::PhantomData;

use ash::vk;
use tracing::debug;

use crate::{
    Error, Result,
    resources::{
        buffer::VertexBuffer,
        command_pool::CommandBuffer,
        descriptors::{DescriptorSet, layouts::SetLayout},
        device::{Garbage, RenderDevice},
    },
    shaders::metadata::{FragmentShader, VertexInput, VertexShader},
};

/// Host-side contract for a graphics pipeline's resource types.
pub trait GraphicsPipelineSpec {
    /// Vertex shader used by this pipeline.
    type VertexShader: VertexShader;
    /// Fragment shader used by this pipeline.
    type FragmentShader: FragmentShader;
    /// Vertex input type accepted by this pipeline.
    type Input: VertexInput;
    /// Descriptor set input tuple accepted by this pipeline.
    type DescriptorSets: DescriptorSetInput;

    /// Human-readable pipeline name used in diagnostics.
    const NAME: &'static str;

    /// Validates host-side inputs against reflected shader metadata.
    fn validate() -> Result<()>
    where
        Self: Sized,
    {
        let reflected_layouts =
            merge_shader_set_layouts::<Self::VertexShader, Self::FragmentShader>(Self::NAME)?;
        validate_pipeline_descriptor_layout::<Self>(Self::NAME, &reflected_layouts)?;
        validate_pipeline_vertex_input::<Self>(Self::NAME)
    }
}

/// Descriptor set tuple metadata and typed references.
pub trait DescriptorSetInput {
    /// Number of descriptor sets in the tuple.
    const SET_COUNT: usize;
    /// Descriptor bindings, ordered by Vulkan set index.
    const BINDINGS: &'static [&'static [vk::DescriptorSetLayoutBinding<'static>]];

    /// Borrowed descriptor sets in tuple order.
    type Refs<'a>
    where
        Self: 'a;

    /// Returns raw Vulkan descriptor set handles in tuple order.
    fn raw_sets(sets: &Self::Refs<'_>) -> Vec<vk::DescriptorSet>;
}

macro_rules! impl_descriptor_set_input_for_tuple {
    ($($name:ident),+ $(,)?) => {
        impl<$($name),+> DescriptorSetInput for ($($name,)+)
        where
            $($name: SetLayout + 'static),+
        {
            const SET_COUNT: usize = [$(stringify!($name)),+].len();
            const BINDINGS: &'static [&'static [vk::DescriptorSetLayoutBinding<'static>]] =
                &[$($name::BINDINGS),+];

            type Refs<'a>
                = ($(&'a DescriptorSet<$name>,)+)
            where
                Self: 'a;

            #[allow(non_snake_case)]
            fn raw_sets(sets: &Self::Refs<'_>) -> Vec<vk::DescriptorSet> {
                let ($($name,)+) = sets;
                vec![$($name.raw()),+]
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

pub struct GraphicsPipeline<Spec>
where
    Spec: GraphicsPipelineSpec,
{
    device: RenderDevice,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    _spec: PhantomData<Spec>,
}

/// Typed rendering context for a bound graphics pipeline.
pub struct GraphicsPipelineRenderingContext<'cmd, Spec>
where
    Spec: GraphicsPipelineSpec,
{
    cmd: &'cmd CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    _spec: PhantomData<Spec>,
}

impl<Spec> GraphicsPipeline<Spec>
where
    Spec: GraphicsPipelineSpec,
{
    pub fn build(device: &RenderDevice) -> Result<Self> {
        let mut device = device.clone();

        Spec::validate()?;

        let reflected_layouts =
            merge_shader_set_layouts::<Spec::VertexShader, Spec::FragmentShader>(Spec::NAME)?;
        let set_layouts = Self::create_pipeline_set_layouts(&mut device, &reflected_layouts)?;

        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { device.device.create_pipeline_layout(&layout_info, None)? };

        let shader_stages = [
            Spec::VertexShader::shader_create_info(&mut device)?,
            Spec::FragmentShader::shader_create_info(&mut device)?,
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(Spec::VertexShader::INPUT_BINDINGS)
            .vertex_attribute_descriptions(Spec::VertexShader::INPUT_ATTRIBUTES);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false) // enabling this disables output to frame buffer
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(device.properties.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_test_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(
                &device.properties.surface_format.format,
            ))
            .depth_attachment_format(device.properties.depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_test_info)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .push_next(&mut pipeline_rendering_info);

        let pipeline = unsafe {
            device
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, err)| err)?[0]
        };

        Ok(Self {
            device,
            pipeline,
            pipeline_layout,
            _spec: PhantomData,
        })
    }
    pub fn bind<'cmd>(
        &self,
        cmd: &'cmd CommandBuffer,
    ) -> GraphicsPipelineRenderingContext<'cmd, Spec> {
        cmd.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        GraphicsPipelineRenderingContext {
            cmd,
            pipeline_layout: self.pipeline_layout,
            _spec: PhantomData,
        }
    }

    fn create_pipeline_set_layouts(
        device: &mut RenderDevice,
        reflected_layouts: &[Vec<vk::DescriptorSetLayoutBinding<'static>>],
    ) -> Result<Vec<vk::DescriptorSetLayout>> {
        reflected_layouts
            .iter()
            .map(|bindings| {
                let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
                let layout = unsafe { device.device.create_descriptor_set_layout(&info, None)? };
                device.destroy(Garbage::DescriptorSetLayout(layout));
                Ok(layout)
            })
            .collect()
    }
}

impl<Spec> GraphicsPipelineRenderingContext<'_, Spec>
where
    Spec: GraphicsPipelineSpec,
{
    pub fn bind_descriptor_sets(
        &self,
        sets: &<Spec::DescriptorSets as DescriptorSetInput>::Refs<'_>,
    ) {
        let descriptor_sets = Spec::DescriptorSets::raw_sets(sets);
        self.cmd.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &descriptor_sets,
            &[],
        );
    }

    pub fn bind_vertex_buffer(&self, vertex_buffer: &VertexBuffer<Spec::Input>) {
        self.cmd
            .bind_vertex_buffers(0, &[vertex_buffer.buffer()], &[0]);
    }
}

impl<Spec> Drop for GraphicsPipeline<Spec>
where
    Spec: GraphicsPipelineSpec,
{
    fn drop(&mut self) {
        self.device
            .destroy(Garbage::PipelineLayout(self.pipeline_layout));
        self.device.destroy(Garbage::Pipeline(self.pipeline));
    }
}

fn merge_shader_set_layouts<V, F>(
    pipeline: &'static str,
) -> Result<Vec<Vec<vk::DescriptorSetLayoutBinding<'static>>>>
where
    V: VertexShader,
    F: FragmentShader,
{
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
    vertex_bindings: &[vk::DescriptorSetLayoutBinding<'static>],
    fragment_bindings: &[vk::DescriptorSetLayoutBinding<'static>],
) -> Result<Vec<vk::DescriptorSetLayoutBinding<'static>>> {
    let mut merged = Vec::new();

    for binding in vertex_bindings.iter().chain(fragment_bindings).copied() {
        if let Some(existing) =
            merged
                .iter_mut()
                .find(|existing: &&mut vk::DescriptorSetLayoutBinding<'static>| {
                    existing.binding == binding.binding
                })
        {
            if existing.descriptor_type != binding.descriptor_type
                || existing.descriptor_count != binding.descriptor_count
            {
                debug!(
                    pipeline,
                    set,
                    expected = ?existing,
                    actual = ?binding,
                    "pipeline descriptor duplicate binding mismatch"
                );
                return Err(Error::PipelineDescriptorLayoutMismatch { pipeline, set });
            }
            existing.stage_flags |= binding.stage_flags;
        } else {
            merged.push(binding);
        }
    }

    merged.sort_by_key(|binding| binding.binding);
    Ok(merged)
}

fn validate_pipeline_descriptor_layout<S>(
    pipeline: &'static str,
    reflected_layouts: &[Vec<vk::DescriptorSetLayoutBinding<'static>>],
) -> Result<()>
where
    S: GraphicsPipelineSpec,
{
    let max_sets = reflected_layouts.len().max(S::DescriptorSets::SET_COUNT);
    for set in 0..max_sets {
        let reflected = reflected_layouts.get(set).map_or_default(Vec::as_slice);
        let expected = S::DescriptorSets::BINDINGS
            .get(set)
            .copied()
            .unwrap_or_default();

        if !descriptor_layouts_match(expected, reflected) {
            debug!(
                pipeline,
                set,
                expected = ?expected,
                actual = ?reflected,
                "pipeline descriptor layout mismatch"
            );
            return Err(Error::PipelineDescriptorLayoutMismatch { pipeline, set });
        }
    }

    Ok(())
}

fn validate_pipeline_vertex_input<S>(pipeline: &'static str) -> Result<()>
where
    S: GraphicsPipelineSpec,
{
    let vertex_layout = S::Input::layout();
    let expected_bindings = [vertex_layout.binding(0)];
    let expected_attributes = vertex_layout.attrs(0, 0);

    if !vertex_bindings_match(S::VertexShader::INPUT_BINDINGS, &expected_bindings)
        || !vertex_attributes_match(S::VertexShader::INPUT_ATTRIBUTES, &expected_attributes)
    {
        debug!(
            pipeline,
            expected_bindings = ?expected_bindings,
            actual_bindings = ?S::VertexShader::INPUT_BINDINGS,
            expected_attributes = ?expected_attributes,
            actual_attributes = ?S::VertexShader::INPUT_ATTRIBUTES,
            "pipeline vertex input mismatch"
        );
        return Err(Error::PipelineVertexInputMismatch { pipeline });
    }

    Ok(())
}

fn descriptor_layouts_match(
    expected: &[vk::DescriptorSetLayoutBinding<'static>],
    actual: &[vk::DescriptorSetLayoutBinding<'static>],
) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| descriptor_bindings_match(expected, actual))
}

fn descriptor_bindings_match(
    expected: &vk::DescriptorSetLayoutBinding<'static>,
    actual: &vk::DescriptorSetLayoutBinding<'static>,
) -> bool {
    expected.binding == actual.binding
        && expected.descriptor_type == actual.descriptor_type
        && expected.descriptor_count == actual.descriptor_count
        && expected.stage_flags == actual.stage_flags
}

fn vertex_bindings_match(
    actual: &[vk::VertexInputBindingDescription],
    expected: &[vk::VertexInputBindingDescription],
) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual.binding == expected.binding
                && actual.stride == expected.stride
                && actual.input_rate == expected.input_rate
        })
}

fn vertex_attributes_match(
    actual: &[vk::VertexInputAttributeDescription],
    expected: &[vk::VertexInputAttributeDescription],
) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(actual, expected)| {
            actual.location == expected.location
                && actual.binding == expected.binding
                && actual.format == expected.format
                && actual.offset == expected.offset
        })
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use crate::{
        pipeline::MainPipelineSpec,
        resources::descriptors::sets::{MaterialSet, ObjectSet, SceneSet},
    };

    use super::*;

    const EMPTY: &[vk::DescriptorSetLayoutBinding<'static>] = &[];
    const TEST_VERTEX_BINDINGS: &[vk::DescriptorSetLayoutBinding<'static>] = &[binding(
        0,
        vk::DescriptorType::UNIFORM_BUFFER,
        1,
        vk::ShaderStageFlags::VERTEX,
    )];
    const TEST_FRAGMENT_BINDINGS: &[vk::DescriptorSetLayoutBinding<'static>] = &[binding(
        0,
        vk::DescriptorType::UNIFORM_BUFFER,
        1,
        vk::ShaderStageFlags::FRAGMENT,
    )];
    const TEST_FRAGMENT_CONFLICT_TYPE_BINDINGS: &[vk::DescriptorSetLayoutBinding<'static>] =
        &[binding(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            1,
            vk::ShaderStageFlags::FRAGMENT,
        )];
    const TEST_FRAGMENT_CONFLICT_COUNT_BINDINGS: &[vk::DescriptorSetLayoutBinding<'static>] =
        &[binding(
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
            2,
            vk::ShaderStageFlags::FRAGMENT,
        )];

    const fn binding(
        binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayoutBinding<'static> {
        vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type,
            descriptor_count,
            stage_flags,
            p_immutable_samplers: ::core::ptr::null(),
            _marker: PhantomData,
        }
    }

    #[test]
    fn main_pipeline_spec_matches_reflected_descriptor_layouts() {
        let reflected = merge_shader_set_layouts::<
            <MainPipelineSpec as GraphicsPipelineSpec>::VertexShader,
            <MainPipelineSpec as GraphicsPipelineSpec>::FragmentShader,
        >(MainPipelineSpec::NAME)
        .expect("main shader layouts should merge");

        validate_pipeline_descriptor_layout::<MainPipelineSpec>(MainPipelineSpec::NAME, &reflected)
            .expect("main pipeline descriptor layouts should match");
    }

    #[test]
    fn main_pipeline_spec_matches_reflected_vertex_input() {
        validate_pipeline_vertex_input::<MainPipelineSpec>(MainPipelineSpec::NAME)
            .expect("main pipeline vertex input should match");
    }

    #[test]
    fn descriptor_layout_merge_combines_stage_flags_for_same_binding() {
        let merged =
            merge_descriptor_set_layout("test", 0, TEST_VERTEX_BINDINGS, TEST_FRAGMENT_BINDINGS)
                .expect("compatible bindings should merge");

        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
    }

    #[test]
    fn descriptor_layout_merge_rejects_conflicting_descriptor_type() {
        let error = merge_descriptor_set_layout(
            "test",
            0,
            TEST_VERTEX_BINDINGS,
            TEST_FRAGMENT_CONFLICT_TYPE_BINDINGS,
        )
        .expect_err("conflicting descriptor type should fail");

        assert!(matches!(
            error,
            Error::PipelineDescriptorLayoutMismatch {
                pipeline: "test",
                set: 0
            }
        ));
    }

    #[test]
    fn descriptor_layout_merge_rejects_conflicting_descriptor_count() {
        let error = merge_descriptor_set_layout(
            "test",
            0,
            TEST_VERTEX_BINDINGS,
            TEST_FRAGMENT_CONFLICT_COUNT_BINDINGS,
        )
        .expect_err("conflicting descriptor count should fail");

        assert!(matches!(
            error,
            Error::PipelineDescriptorLayoutMismatch {
                pipeline: "test",
                set: 0
            }
        ));
    }

    #[test]
    fn descriptor_set_input_bindings_returns_three_ordered_sets() {
        assert_eq!(
            <(SceneSet, ObjectSet, MaterialSet) as DescriptorSetInput>::SET_COUNT,
            3
        );

        let bindings = <(SceneSet, ObjectSet, MaterialSet) as DescriptorSetInput>::BINDINGS;
        assert_eq!(bindings.len(), 3);
        assert!(descriptor_layouts_match(SceneSet::BINDINGS, bindings[0]));
        assert!(descriptor_layouts_match(ObjectSet::BINDINGS, bindings[1]));
        assert!(descriptor_layouts_match(MaterialSet::BINDINGS, bindings[2]));
    }

    #[test]
    fn empty_reflected_sets_are_compared_explicitly() {
        let reflected = vec![Vec::new()];

        validate_pipeline_descriptor_layout::<MainPipelineSpec>("test", &reflected)
            .expect_err("empty set zero must not match the main scene set");
        assert!(descriptor_layouts_match(EMPTY, &[]));
    }

    #[test]
    fn render_api_compile_time_patterns_accept_main_types() {
        fn accepts_main_pipeline(_: &GraphicsPipeline<MainPipelineSpec>) {}
        fn accepts_any<T>(_: T) {}

        let _ = accepts_main_pipeline;
        accepts_any(GraphicsPipelineRenderingContext::<MainPipelineSpec>::bind_vertex_buffer);
    }
}
