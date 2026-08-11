//! Typed bind group owned by the active RHI.

use std::marker::PhantomData;

#[cfg(feature = "editor")]
use ash::vk;

use crate::resources::{ActiveBindGroup, descriptors::layouts::SetLayout};

/// A backend bind group tagged with its type-level renderer layout.
pub struct DescriptorSet<L: SetLayout> {
    inner: ActiveBindGroup,
    _layout: PhantomData<L>,
}

impl<L: SetLayout> DescriptorSet<L> {
    pub(super) fn new(inner: ActiveBindGroup) -> Self {
        Self {
            inner,
            _layout: PhantomData,
        }
    }

    pub(crate) fn group(&self) -> &ActiveBindGroup {
        &self.inner
    }

    /// Returns the Vulkan descriptor set used by the temporary editor adapter.
    #[cfg(feature = "editor")]
    pub fn raw(&self) -> vk::DescriptorSet {
        self.inner.raw()
    }
}
