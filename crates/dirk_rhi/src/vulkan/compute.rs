use std::sync::Arc;

use ash::vk;

use super::{Check, CommandBuffer, Device, device::Garbage};
use crate::{ComputePipelineDesc, PushData, Result};

/// A Vulkan compute pipeline.
pub(crate) struct ComputePipeline {
    pub(super) device: Arc<Device>,
    raw: vk::Pipeline,
}

impl ComputePipeline {
    /// Creates a pipeline for a validated description, with the global
    /// pipeline layout.
    pub(crate) fn new<P: PushData>(
        device: &Arc<Device>,
        desc: &ComputePipelineDesc<'_, P>,
    ) -> Result<Self> {
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(desc.shader.raw.stage(vk::ShaderStageFlags::COMPUTE))
            .layout(device.heap.pipeline_layout);
        // SAFETY: the description was validated against the device, and
        // every pointer in `info` outlives the call.
        let raw = unsafe {
            device
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, result)| result)
        .check("vkCreateComputePipelines")?[0];
        device.name(raw, desc.label);
        Ok(Self {
            device: device.clone(),
            raw,
        })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.device.bury(Garbage::Pipeline(self.raw));
    }
}

impl CommandBuffer {
    /// Runs `workgroups` workgroups of `pipeline` reading `push`, binding the
    /// pipeline, and the heap on the first dispatch, unless bound.
    pub(crate) fn dispatch(
        &mut self,
        device: &Device,
        pipeline: &ComputePipeline,
        push: &[u8],
        [x, y, z]: [u32; 3],
    ) {
        if self.compute != pipeline.raw {
            if self.compute == vk::Pipeline::null() {
                device.bind_heap(self.raw, vk::PipelineBindPoint::COMPUTE);
            }
            // SAFETY: the command buffer is recording on the graphics queue,
            // whose family runs compute work, and the pipeline outlives the
            // frame's work.
            unsafe {
                device.raw.cmd_bind_pipeline(
                    self.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.raw,
                );
            }
            self.compute = pipeline.raw;
        }
        self.push(device, push);
        // SAFETY: as above; the workgroup counts are within the device's
        // limits.
        unsafe { device.raw.cmd_dispatch(self.raw, x, y, z) };
    }
}
