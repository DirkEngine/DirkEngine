use dirk_rhi::{BindGroupLayoutEntry, BindingType, ShaderStages};

use crate::resources::descriptors::layouts::SetLayout;

pub struct SceneSet;

impl SetLayout for SceneSet {
    const BINDINGS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        ty: BindingType::UniformBuffer {
            dynamic_offset: false,
        },
        visibility: ShaderStages::VERTEX,
    }];
}

pub struct ObjectSet;

impl SetLayout for ObjectSet {
    const BINDINGS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        ty: BindingType::UniformBuffer {
            dynamic_offset: false,
        },
        visibility: ShaderStages::VERTEX,
    }];
}

pub struct MaterialSet;

impl SetLayout for MaterialSet {
    const BINDINGS: &'static [BindGroupLayoutEntry] = &[BindGroupLayoutEntry {
        binding: 0,
        ty: BindingType::SampledImage,
        visibility: ShaderStages::FRAGMENT,
    }];
}
