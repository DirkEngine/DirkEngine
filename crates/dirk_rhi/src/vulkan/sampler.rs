use std::sync::Arc;

use ash::vk;

use dirk_shaders::bindless::heap;

use super::{
    Check, Device,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{AddressMode, Filter, Result, SamplerDesc};

/// A Vulkan sampler.
pub(crate) struct Sampler {
    device: Arc<Device>,
    pub(super) raw: vk::Sampler,
    slot: Slot,
}

impl Sampler {
    /// Creates a sampler for a validated description.
    pub(crate) fn new(device: &Arc<Device>, desc: &SamplerDesc<'_>) -> Result<Self> {
        let [u, v, w] = desc.address_modes.map(AddressMode::vk);
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(desc.mag_filter.vk())
            .min_filter(desc.min_filter.vk())
            .mipmap_mode(match desc.mipmap_filter {
                Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
                Filter::Linear => vk::SamplerMipmapMode::LINEAR,
            })
            .address_mode_u(u)
            .address_mode_v(v)
            .address_mode_w(w)
            .anisotropy_enable(desc.max_anisotropy > 1)
            .max_anisotropy(f32::from(desc.max_anisotropy))
            .min_lod(desc.min_lod)
            .max_lod(desc.max_lod);
        // SAFETY: `info` is valid for the device; anisotropy is only enabled
        // when the device supports it, as validated against its limit.
        let raw = unsafe { device.raw.create_sampler(&info, None) }.check("vkCreateSampler")?;
        device.name(raw, desc.label);
        let slot = device
            .add_to_heap(heap::SAMPLERS, Descriptor::Sampler(raw))
            .inspect_err(|_| {
                // SAFETY: nothing uses the sampler yet.
                unsafe { device.raw.destroy_sampler(raw, None) };
            })?;
        Ok(Self {
            device: device.clone(),
            raw,
            slot,
        })
    }

    /// The sampler's slot in the heap.
    pub(crate) fn slot(&self) -> u32 {
        self.slot.index
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.device.bury(Garbage::Sampler(self.raw, self.slot));
    }
}

impl Filter {
    /// The equivalent Vulkan filter.
    pub(super) const fn vk(self) -> vk::Filter {
        match self {
            Self::Nearest => vk::Filter::NEAREST,
            Self::Linear => vk::Filter::LINEAR,
        }
    }
}

impl AddressMode {
    /// The equivalent Vulkan address mode.
    const fn vk(self) -> vk::SamplerAddressMode {
        match self {
            Self::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            Self::Repeat => vk::SamplerAddressMode::REPEAT,
            Self::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        }
    }
}
