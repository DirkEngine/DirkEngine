use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLResource, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use super::{
    Check, Device, Object,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{ImageDesc, ImageDimension, ImageUsage, Result, SampleCount, ViewType, image::ViewOf};

/// Identifies an image while it exists.
pub(crate) type ImageId = *const ProtocolObject<dyn MTLTexture>;

/// A Metal texture, which is also its own view of every mip and layer, and,
/// when rendered to or stored per mip or layer, a texture view of each mip
/// and layer.
pub(crate) struct Image {
    pub(super) device: Arc<Device>,
    pub(super) raw: Object<dyn MTLTexture>,
    /// Single-mip, single-layer 2D texture views, layer by layer, mip by
    /// mip.
    subviews: Vec<Subview>,
    /// Whether the texture is released through deferred destruction; a
    /// drawable's is released as soon as it is replaced, since the command
    /// buffers using it retain it and it fills no heap slot.
    deferred: bool,
    /// The texture's slots in the heap: sampled with sampled usage, and
    /// storage with storage usage when its view type has a storage array.
    slots: [Option<Slot>; 2],
}

/// A texture view of one mip and layer, and its heap slots.
struct Subview {
    texture: Object<dyn MTLTexture>,
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
            subviews: Vec::new(),
            deferred: true,
            slots: [None; 2],
        };
        if let Some(view_type) = view {
            image.slots = Self::fill_slots(device, desc, desc.usage, view_type, &image.raw)?;
        }
        for range in desc.subview_ranges() {
            let levels = NSRange::new(range.base_mip_level as usize, 1);
            let slices = NSRange::new(range.base_layer as usize, 1);
            // SAFETY: the mip and layer are inside the texture, and a 2D view
            // of the same format is compatible with 2D and cube textures.
            let texture: Object<_> = unsafe {
                image
                    .raw
                    .newTextureViewWithPixelFormat_textureType_levels_slices(
                        desc.format.mtl(),
                        MTLTextureType::Type2D,
                        levels,
                        slices,
                    )
            }
            .check("newTextureViewWithPixelFormat")?
            .into();
            // Pushed first, so a failure below releases it with the image.
            image.subviews.push(Subview {
                texture: texture.clone(),
                slots: [None; 2],
            });
            let slots =
                Self::fill_slots(device, desc, desc.subview_usage(), ViewType::D2, &texture)?;
            if let Some(subview) = image.subviews.last_mut() {
                subview.slots = slots;
            }
        }
        Ok(image)
    }

    /// Adds `texture`, a view of type `view_type`, to the heap arrays
    /// `usage`, part of the usage of `desc`, needs, returning its sampled and
    /// storage slots.
    fn fill_slots(
        device: &Device,
        desc: &ImageDesc<'_>,
        usage: ImageUsage,
        view_type: ViewType,
        texture: &ProtocolObject<dyn MTLTexture>,
    ) -> Result<[Option<Slot>; 2]> {
        let descriptor = Descriptor::Texture(texture);
        let mut slots = [None; 2];
        for ((needed, binding), slot) in view_type
            .heap_bindings(desc.format)
            .into_iter()
            .zip(&mut slots)
        {
            if let Some(binding) = binding.filter(|_| usage.contains(needed)) {
                match device.add_to_heap(binding, &descriptor) {
                    Ok(added) => *slot = Some(added),
                    Err(error) => {
                        // Nothing uses the texture yet; free the slots taken.
                        device.bury(Garbage {
                            slots,
                            object: Box::new(()),
                        });
                        return Err(error);
                    }
                }
            }
        }
        Ok(slots)
    }

    /// Wraps the texture of a drawable, which its swapchain replaces with
    /// each acquisition.
    pub(super) fn drawable(device: &Arc<Device>, raw: Object<dyn MTLTexture>) -> Self {
        Self {
            device: device.clone(),
            raw,
            subviews: Vec::new(),
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

    /// The heap slots of each subview, in order.
    pub(crate) fn subview_slots(&self) -> impl Iterator<Item = [Option<u32>; 2]> {
        self.subviews
            .iter()
            .map(|subview| subview.slots.map(|slot| slot.map(|slot| slot.index)))
    }

    /// The texture rendered to: the texture itself, or a subview.
    pub(super) fn attachment_texture(&self, view: ViewOf) -> &ProtocolObject<dyn MTLTexture> {
        match view {
            ViewOf::Full => &self.raw,
            ViewOf::Subview(index) => &self.subviews[index].texture,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.deferred {
            for subview in self.subviews.drain(..) {
                self.device.bury(Garbage {
                    slots: subview.slots,
                    object: Box::new(subview.texture),
                });
            }
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
