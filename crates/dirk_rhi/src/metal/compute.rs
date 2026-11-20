use std::sync::Arc;

use dirk_shaders::bindless::heap::metal::PUSH_BUFFER;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use super::{Check, CommandBuffer, Device, Object, device::Garbage, heap::Usage, pass::Push};
use crate::{ComputePipelineDesc, Error, PushData, Result};

/// Identifies a compute pipeline while it is alive, to skip rebinding it.
pub(super) type ComputePipelineId = *const ProtocolObject<dyn MTLComputePipelineState>;

/// A Metal compute pipeline, the threadgroup size its dispatches use, the
/// shader's workgroup size, and the usage they need resources declared up
/// to.
pub(crate) struct ComputePipeline {
    pub(super) device: Arc<Device>,
    raw: Object<dyn MTLComputePipelineState>,
    threads: MTLSize,
    usage: Usage,
}

impl ComputePipeline {
    /// Creates a pipeline for a validated description.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline runs fewer threads per
    /// threadgroup than the workgroup size, which Metal decides per
    /// pipeline.
    pub(crate) fn new<P: PushData>(
        device: &Arc<Device>,
        desc: &ComputePipelineDesc<'_, P>,
    ) -> Result<Self> {
        let raw: Object<_> = device
            .raw
            .newComputePipelineStateWithFunction_error(&desc.shader.raw.function)
            .check("newComputePipelineStateWithFunction")?
            .into();
        let [width, height, depth] = desc.shader.workgroup_size().map(|size| size as usize);
        let limit = raw.maxTotalThreadsPerThreadgroup();
        if width * height * depth > limit {
            return Err(Error::invalid(
                "compute pipeline",
                desc.label,
                format!(
                    "workgroups of {:?} exceed the {limit} threads this pipeline runs per \
                     threadgroup",
                    desc.shader.workgroup_size()
                ),
            ));
        }
        Ok(Self {
            device: device.clone(),
            raw,
            threads: MTLSize {
                width,
                height,
                depth,
            },
            usage: Usage::of_pipeline(desc.shader.writes_storage()),
        })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.device.bury(Garbage::object(self.raw.clone()));
    }
}

impl CommandBuffer {
    /// Runs `workgroups` threadgroups of `pipeline` reading `push`, binding
    /// the pipeline unless bound to the open compute encoder.
    pub(crate) fn dispatch(
        &mut self,
        device: &Device,
        pipeline: &ComputePipeline,
        push: &[u8],
        [width, height, depth]: [u32; 3],
    ) {
        let id: ComputePipelineId = &raw const *pipeline.raw;
        let bound = self.compute_pipeline;
        let encoder = self.compute(device, pipeline.usage);
        if bound != Some(id) {
            encoder.setComputePipelineState(&pipeline.raw);
        }
        if let Some(push) = Push::new(push) {
            // SAFETY: `push` holds the bytes read during the call.
            unsafe {
                encoder.setBytes_length_atIndex(push.bytes(), push.length, PUSH_BUFFER as usize);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: width as usize,
                height: height as usize,
                depth: depth as usize,
            },
            pipeline.threads,
        );
        self.compute_pipeline = Some(id);
    }
}
