use std::sync::Arc;

use dirk_shaders::bindless::heap;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLDevice, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLSamplerState,
};

use super::{
    Check, Device, Object,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{AddressMode, Filter, Result, SamplerDesc};

/// A Metal sampler state.
pub(crate) struct Sampler {
    device: Arc<Device>,
    raw: Object<dyn MTLSamplerState>,
    slot: Slot,
}

impl Sampler {
    /// Creates a sampler for a validated description.
    pub(crate) fn new(device: &Arc<Device>, desc: &SamplerDesc<'_>) -> Result<Self> {
        let raw = Self::create(&device.raw, desc)?;
        let slot = device.add_to_heap(heap::SAMPLERS, &Descriptor::Sampler(&raw))?;
        Ok(Self {
            device: device.clone(),
            raw,
            slot,
        })
    }

    /// Creates a sampler state shaders can read from the heap.
    pub(super) fn create(
        device: &Object<dyn MTLDevice>,
        desc: &SamplerDesc<'_>,
    ) -> Result<Object<dyn MTLSamplerState>> {
        let sampler = MTLSamplerDescriptor::new();
        let [u, v, w] = desc.address_modes.map(AddressMode::mtl);
        sampler.setMagFilter(desc.mag_filter.mtl());
        sampler.setMinFilter(desc.min_filter.mtl());
        sampler.setMipFilter(match desc.mipmap_filter {
            Filter::Nearest => MTLSamplerMipFilter::Nearest,
            Filter::Linear => MTLSamplerMipFilter::Linear,
        });
        sampler.setSAddressMode(u);
        sampler.setTAddressMode(v);
        sampler.setRAddressMode(w);
        sampler.setLodMinClamp(desc.min_lod);
        sampler.setLodMaxClamp(desc.max_lod);
        sampler.setSupportArgumentBuffers(true);
        sampler.setLabel(Some(&NSString::from_str(desc.label)));
        sampler.setMaxAnisotropy(usize::from(desc.max_anisotropy));
        Ok(device
            .newSamplerStateWithDescriptor(&sampler)
            .check("newSamplerStateWithDescriptor")?
            .into())
    }

    /// The sampler's slot in the heap.
    pub(crate) fn slot(&self) -> u32 {
        self.slot.index
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.device.bury(Garbage {
            slots: [Some(self.slot), None],
            object: Box::new(self.raw.clone()),
        });
    }
}

impl Filter {
    /// The equivalent Metal filter.
    pub(super) const fn mtl(self) -> MTLSamplerMinMagFilter {
        match self {
            Self::Nearest => MTLSamplerMinMagFilter::Nearest,
            Self::Linear => MTLSamplerMinMagFilter::Linear,
        }
    }
}

impl AddressMode {
    /// The equivalent Metal address mode.
    const fn mtl(self) -> MTLSamplerAddressMode {
        match self {
            Self::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
            Self::Repeat => MTLSamplerAddressMode::Repeat,
            Self::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
        }
    }
}
