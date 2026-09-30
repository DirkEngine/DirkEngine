//! Typed binding declarations and immutable groups.
use crate::buffer::UniformBuffer;
use bytemuck::NoUninit;
use dirk_rhi::BindGroupLayoutEntry;
use dirk_rhi::{
    BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindingResource, BindingType,
    ImageView, Rhi, Sampler,
};
use std::marker::PhantomData;

/// Type-level description of one bind-group layout.
pub trait SetLayout {
    /// Ordered shader-visible bindings in this set.
    const BINDINGS: &'static [BindGroupLayoutEntry];
}

/// A backend bind group tagged with its type-level layout.
pub struct DescriptorSet<L: SetLayout> {
    inner: BindGroup,
    _layout: PhantomData<L>,
}

impl<L: SetLayout> DescriptorSet<L> {
    pub(super) fn new(inner: BindGroup) -> Self {
        Self {
            inner,
            _layout: PhantomData,
        }
    }

    /// Borrows the native-independent RHI group.
    #[must_use]
    pub fn group(&self) -> &BindGroup {
        &self.inner
    }
}

/// Owns one typed bind-group layout and creates groups implementing it.
pub struct BindingLayout<L: SetLayout> {
    layout: dirk_rhi::BindGroupLayout,
    _layout: PhantomData<L>,
}

impl<L: SetLayout> BindingLayout<L> {
    /// Creates a layout from the type-level declarations.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn new(device: &Rhi) -> dirk_rhi::Result<Self> {
        Ok(Self {
            layout: device.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "renderer bind-group layout",
                entries: L::BINDINGS,
            })?,
            _layout: PhantomData,
        })
    }

    /// Creates a set binding one whole uniform record.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn uniform_buffer<T: NoUninit>(
        &self,
        rhi: &Rhi,
        binding: u32,
        uniform: &UniformBuffer<T>,
    ) -> dirk_rhi::Result<DescriptorSet<L>> {
        Self::require_binding(
            binding,
            BindingType::UniformBuffer {
                dynamic_offset: false,
            },
        )?;
        self.create(
            rhi,
            &[BindGroupEntry {
                binding,
                resource: BindingResource::Buffer {
                    buffer: uniform.buffer(),
                    offset: 0,
                    size: uniform.buffer().size(),
                },
            }],
        )
    }

    /// Creates a set containing one sampled-image binding.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn sampled_image(
        &self,
        rhi: &Rhi,
        binding: u32,
        view: &ImageView,
        sampler: &Sampler,
    ) -> dirk_rhi::Result<DescriptorSet<L>> {
        Self::require_binding(binding, BindingType::SampledImage)?;
        self.create(
            rhi,
            &[BindGroupEntry {
                binding,
                resource: BindingResource::SampledImage { view, sampler },
            }],
        )
    }

    fn create(
        &self,
        rhi: &Rhi,
        entries: &[BindGroupEntry<'_, Rhi>],
    ) -> dirk_rhi::Result<DescriptorSet<L>> {
        Ok(DescriptorSet::new(rhi.create_bind_group(
            &BindGroupDesc {
                label: "renderer bind group",
                layout: &self.layout,
                entries,
            },
        )?))
    }

    fn require_binding(binding: u32, ty: BindingType) -> dirk_rhi::Result<()> {
        match L::BINDINGS.iter().find(|entry| entry.binding == binding) {
            Some(entry) if entry.ty == ty => Ok(()),
            _ => Err(dirk_rhi::InvalidResourceKind::Mismatch
                .with_detail(format!(
                    "set layout does not declare binding {binding} as {ty:?}"
                ))
                .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirk_rhi::ShaderStages;

    struct Set;
    impl SetLayout for Set {
        const BINDINGS: &'static [BindGroupLayoutEntry] = &[
            BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                },
                visibility: ShaderStages::VERTEX,
            },
            BindGroupLayoutEntry {
                binding: 2,
                ty: BindingType::SampledImage,
                visibility: ShaderStages::FRAGMENT,
            },
        ];
    }

    #[test]
    fn bindings_must_exist_with_the_requested_type() {
        let uniform = BindingType::UniformBuffer {
            dynamic_offset: false,
        };
        assert!(BindingLayout::<Set>::require_binding(0, uniform).is_ok());
        assert!(BindingLayout::<Set>::require_binding(2, BindingType::SampledImage).is_ok());
        assert!(BindingLayout::<Set>::require_binding(2, uniform).is_err());
        assert!(BindingLayout::<Set>::require_binding(1, BindingType::SampledImage).is_err());
        let dynamic = BindingType::UniformBuffer {
            dynamic_offset: true,
        };
        assert!(BindingLayout::<Set>::require_binding(0, dynamic).is_err());
    }
}
