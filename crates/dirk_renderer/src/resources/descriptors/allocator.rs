//! Typed bind-group factory.

use std::{marker::PhantomData, sync::Arc};

use dirk_rhi::{
    Backend as _, BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindingResource, BindingType,
};

use crate::{
    Result,
    resources::{
        ActiveBuffer, ActiveImageView, ActiveRhi, ActiveSampler,
        descriptors::{layouts::SetLayout, set::DescriptorSet},
        device::RenderDevice,
    },
};

/// Owns one typed bind-group layout and creates groups implementing it.
pub struct DescriptorAllocator<L: SetLayout> {
    rhi: Arc<ActiveRhi>,
    layout: <ActiveRhi as dirk_rhi::Backend>::BindGroupLayout,
    _layout: PhantomData<L>,
}

impl<L: SetLayout> DescriptorAllocator<L> {
    /// Creates the backend layout. The capacity hint remains accepted so call
    /// sites do not need to model backend-owned descriptor pooling.
    pub fn new(device: &RenderDevice, _initial_capacity: u32) -> Result<Self> {
        Ok(Self {
            rhi: device.rhi.clone(),
            layout: device.rhi.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "renderer bind-group layout",
                entries: L::BINDINGS,
            })?,
            _layout: PhantomData,
        })
    }

    /// Creates a set containing one uniform-buffer binding.
    pub fn uniform_buffer(
        &self,
        binding: u32,
        buffer: &ActiveBuffer,
        size: u64,
    ) -> Result<DescriptorSet<L>> {
        Self::require_binding(
            binding,
            BindingType::UniformBuffer {
                dynamic_offset: false,
            },
        )?;
        self.create(&[BindGroupEntry {
            binding,
            resource: BindingResource::Buffer {
                buffer,
                offset: 0,
                size,
            },
        }])
    }

    /// Creates a set containing one sampled-image binding.
    pub fn sampled_image(
        &self,
        binding: u32,
        view: &ActiveImageView,
        sampler: &ActiveSampler,
    ) -> Result<DescriptorSet<L>> {
        Self::require_binding(binding, BindingType::SampledImage)?;
        self.create(&[BindGroupEntry {
            binding,
            resource: BindingResource::SampledImage { view, sampler },
        }])
    }

    fn create(&self, entries: &[BindGroupEntry<'_, ActiveRhi>]) -> Result<DescriptorSet<L>> {
        Ok(DescriptorSet::new(self.rhi.create_bind_group(
            &BindGroupDesc {
                label: "renderer bind group",
                layout: &self.layout,
                entries,
            },
        )?))
    }

    fn require_binding(binding: u32, ty: BindingType) -> Result<()> {
        match L::BINDINGS.iter().find(|entry| entry.binding == binding) {
            Some(entry) if entry.ty == ty => Ok(()),
            _ => Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Mismatch).into()),
        }
    }
}
