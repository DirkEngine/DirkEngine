use std::marker::PhantomData;

use crate::{
    CommandEncoder, Error, Graphics, PushData, Result, Rhi, ShaderModule, ShaderStage, native,
};

/// Describes a compute pipeline whose shader reads push data `P`.
#[derive(Clone, Copy)]
pub struct ComputePipelineDesc<'a, P: PushData> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// The compute shader, which declares its workgroup size.
    pub shader: &'a ShaderModule<P>,
}

/// A compute pipeline whose shader reads push data `P`. Destroyed when
/// dropped, once the GPU is done with it.
///
/// Shaders reach resources through the bindless heap, like graphics
/// shaders: storage buffers and images they write are described to
/// barriers and the render graph as
/// [`BufferAccess::ShaderWrite`](crate::BufferAccess::ShaderWrite) or
/// [`ImageAccess::StorageWrite`](crate::ImageAccess::StorageWrite) with
/// [`ShaderStages::COMPUTE`](crate::ShaderStages::COMPUTE).
pub struct ComputePipeline<P: PushData> {
    pub(crate) raw: native::ComputePipeline,
    workgroup_size: [u32; 3],
    label: Box<str>,
    push: PhantomData<fn(&P)>,
}

impl Rhi {
    /// Creates a compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the shader is not a compute shader or
    /// belongs to another device, or its workgroup size exceeds the device's
    /// limits, or a native error if the pipeline cannot be created.
    pub fn create_compute_pipeline<P: PushData>(
        &self,
        desc: &ComputePipelineDesc<'_, P>,
    ) -> Result<ComputePipeline<P>> {
        let invalid = |reason: String| Err(Error::invalid("compute pipeline", desc.label, reason));
        let shader = desc.shader;
        if shader.stage() != ShaderStage::Compute {
            return invalid(format!(
                "`{}` is not a compute shader",
                shader.entry_point()
            ));
        }
        if !shader.raw.is_from(&self.device) {
            return invalid("the shader belongs to another device".to_owned());
        }
        let size = shader.workgroup_size();
        let limits = &self.capabilities().limits;
        let invocations = size
            .iter()
            .try_fold(1_u32, |all, &size| all.checked_mul(size));
        if size.contains(&0)
            || size
                .iter()
                .zip(limits.max_compute_workgroup_size)
                .any(|(&size, limit)| size > limit)
            || invocations
                .is_none_or(|invocations| invocations > limits.max_compute_workgroup_invocations)
        {
            return invalid(format!(
                "workgroups of {size:?} are empty or exceed the device's {:?} and {} invocations",
                limits.max_compute_workgroup_size, limits.max_compute_workgroup_invocations
            ));
        }
        Ok(ComputePipeline {
            raw: native::ComputePipeline::new(&self.device, desc)?,
            workgroup_size: size,
            label: desc.label.into(),
            push: PhantomData,
        })
    }
}

impl<P: PushData> ComputePipeline<P> {
    /// The pipeline's debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Invocations per workgroup along each axis, which the shader declares.
    #[must_use]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}

impl CommandEncoder<'_, Graphics> {
    /// Runs `workgroups` workgroups of `pipeline`, whose shader reads `push`.
    /// Dispatches run outside render passes, on the graphics queue.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline belongs to another device
    /// or `workgroups` exceeds
    /// [`Limits::max_compute_workgroups`](crate::Limits::max_compute_workgroups).
    pub fn dispatch<P: PushData>(
        &mut self,
        pipeline: &ComputePipeline<P>,
        push: &P,
        workgroups: [u32; 3],
    ) -> Result<()> {
        let invalid = |reason: String| Err(Error::invalid("dispatch", pipeline.label(), reason));
        if !pipeline.raw.is_from(self.device()) {
            return invalid("the pipeline belongs to another device".to_owned());
        }
        let limit = self
            .frame
            .rhi()
            .capabilities()
            .limits
            .max_compute_workgroups;
        if workgroups
            .iter()
            .zip(limit)
            .any(|(&count, limit)| count > limit)
        {
            return invalid(format!(
                "{workgroups:?} workgroups exceed the device's {limit:?}"
            ));
        }
        self.raw.dispatch(
            self.device(),
            &pipeline.raw,
            bytemuck::bytes_of(push),
            workgroups,
        );
        Ok(())
    }
}
