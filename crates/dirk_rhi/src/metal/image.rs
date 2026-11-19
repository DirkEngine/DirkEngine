use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLResource, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage};

use super::{
    Device, Object,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{ImageDesc, ImageDimension, ImageUsage, Result, SampleCount, ViewType};

/// Identifies an image while it exists.
pub(crate) type ImageId = *const ProtocolObject<dyn MTLTexture>;

/// A Metal texture, which is also its own view of every mip and layer.
pub(crate) struct Image {
    pub(super) device: Arc<Device>,
    pub(super) raw: Object<dyn MTLTexture>,
    /// Whether the texture is released through deferred destruction; a
    /// drawable's is released as soon as it is replaced, since the command
    /// buffers using it retain it and it fills no heap slot.
    deferred: bool,
    /// The texture's slots in the heap: sampled with sampled usage, and
    /// storage with storage usage when its view type has a storage array.
    slots: [Option<Slot>; 2],
}

impl Image {
    /// Creates a texture for a validated description, filling heap slots
    /// for the view type `view` if given.
    pub(crate) fn new(
        device: &Arc<Device>,
        desc: &ImageDesc<'_>,
        view: Option<ViewType>,
    ) -> Result<Self> {
        let texture = MTLTextureDescriptor::new();
        let cubes = desc.array_layers / 6;
        let (texture_type, layers) = match desc.dimension {
            ImageDimension::D2 if desc.samples != SampleCount::X1 => {
                (MTLTextureType::Type2DMultisample, 1)
            }
            ImageDimension::D2 if desc.array_layers == 1 => (MTLTextureType::Type2D, 1),
            ImageDimension::D2 => (MTLTextureType::Type2DArray, desc.array_layers),
            ImageDimension::D3 => (MTLTextureType::Type3D, 1),
            ImageDimension::Cube if cubes == 1 => (MTLTextureType::TypeCube, 1),
            ImageDimension::Cube => (MTLTextureType::TypeCubeArray, cubes),
        };
        texture.setTextureType(texture_type);
        texture.setPixelFormat(desc.format.mtl());
        texture.setUsage(desc.usage.mtl(desc.format.is_depth_stencil()));
        // SAFETY: the description was validated against the device's limits.
        unsafe {
            texture.setWidth(desc.extent.width as usize);
            texture.setHeight(desc.extent.height as usize);
            texture.setDepth(desc.extent.depth as usize);
            texture.setMipmapLevelCount(desc.mip_levels as usize);
            texture.setArrayLength(layers as usize);
            texture.setSampleCount(desc.samples.count());
        }
        let raw: Object<_> = device.allocator.texture(&device.raw, &texture)?.into();
        raw.setLabel(Some(&NSString::from_str(desc.label)));
        // Owned from here on, so a failure below frees the slots taken.
        let mut image = Self {
            device: device.clone(),
            raw,
            deferred: true,
            slots: [None; 2],
        };
        if let Some(view_type) = view {
            let (sampled, storage) = view_type.heap_bindings();
            let descriptor = Descriptor::Texture(&image.raw);
            if desc.usage.contains(ImageUsage::SAMPLED) {
                image.slots[0] = Some(device.add_to_heap(sampled, &descriptor)?);
            }
            if let Some(storage) = storage.filter(|_| desc.usage.contains(ImageUsage::STORAGE)) {
                image.slots[1] = Some(device.add_to_heap(storage, &descriptor)?);
            }
        }
        Ok(image)
    }

    /// Wraps the texture of a drawable, which its swapchain replaces with
    /// each acquisition.
    pub(super) fn drawable(device: &Arc<Device>, raw: Object<dyn MTLTexture>) -> Self {
        Self {
            device: device.clone(),
            raw,
            deferred: false,
            slots: [None; 2],
        }
    }

    /// Identifies the image while it exists: until the GPU is done with it.
    pub(crate) fn id(&self) -> ImageId {
        &raw const *self.raw
    }

    /// The heap slots of the image's view: sampled and storage.
    pub(crate) fn slots(&self) -> [Option<u32>; 2] {
        self.slots.map(|slot| slot.map(|slot| slot.index))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.deferred {
            self.device.bury(Garbage {
                slots: self.slots,
                object: Box::new(self.raw.clone()),
            });
        }
    }
}

impl ImageUsage {
    /// The equivalent Metal usage. Copies and blits of color images sample
    /// and render to them, since Metal draws scaling blits.
    fn mtl(self, depth_stencil: bool) -> MTLTextureUsage {
        let mut usage = MTLTextureUsage::Unknown;
        if self.intersects(Self::SAMPLED | Self::STORAGE)
            || (self.contains(Self::COPY_SRC) && !depth_stencil)
        {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if self.contains(Self::STORAGE) {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        if self.intersects(Self::COLOR_ATTACHMENT | Self::DEPTH_STENCIL_ATTACHMENT)
            || (self.contains(Self::COPY_DST) && !depth_stencil)
        {
            usage |= MTLTextureUsage::RenderTarget;
        }
        usage
    }
}

impl SampleCount {
    /// Samples per texel.
    pub(super) const fn count(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }
}
