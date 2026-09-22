use crate::{
    resources::descriptors::sets::{MaterialSet, ObjectSet, SceneSet},
    shaders::{MainFS, MainVS},
    utils::Vertex,
};

use super::graphics::GraphicsPipelineSpec;

/// Main model rendering pipeline.
pub struct MainPipelineSpec;

impl GraphicsPipelineSpec for MainPipelineSpec {
    type VertexShader = MainVS;
    type FragmentShader = MainFS;
    type Input = Vertex;
    type DescriptorSets = (SceneSet, ObjectSet, MaterialSet);

    const NAME: &'static str = "main";
}

#[cfg(test)]
mod test {
    use crate::pipeline::{MainPipelineSpec, graphics::GraphicsPipelineSpec};

    #[test]
    fn validate_main_pipeline_spec() {
        MainPipelineSpec::validate().expect("main pipeline spec should match shader reflection");
    }
}

impl MainPipelineSpec {
    pub fn settings(properties: crate::RendererProperties) -> super::graphics::PipelineSettings {
        super::graphics::PipelineSettings {
            color_format: properties.surface_format,
            depth: Some(dirk_rhi::DepthState {
                format: properties.depth_format,
                write_enabled: true,
                compare: dirk_rhi::CompareOp::Less,
                stencil: None,
            }),
            samples: properties.msaa_samples,
        }
    }
}
