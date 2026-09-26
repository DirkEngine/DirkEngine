//! Typed binding declarations and immutable groups.
use dirk_rhi::BindGroupLayoutEntry;
use dirk_rhi::{
    BindGroup, BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindingResource, BindingType,
    Buffer, ImageView, Rhi, Sampler,
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

    /// Creates a set containing one uniform-buffer binding.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn uniform_buffer(
        &self,
        rhi: &Rhi,
        binding: u32,
        buffer: &Buffer,
        size: u64,
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
                    buffer,
                    offset: 0,
                    size,
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
            _ => Err(dirk_rhi::InvalidResourceKind::Mismatch.into()),
        }
    }
}
