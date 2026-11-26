use std::sync::Arc;

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};

use super::{
    Check, Device,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{Format, ImageAccess, ImageDesc, ImageDimension, ImageUsage, SampleCount, ViewType};

/// Identifies an image while it exists.
pub(crate) type ImageId = vk::Image;

/// A Vulkan image, its memory and its view of every mip and layer.
pub(crate) struct Image {
    pub(super) device: Arc<Device>,
    pub(super) raw: vk::Image,
    allocation: Allocation,
    view: Option<vk::ImageView>,
    /// Whether the image is destroyed with this object; swapchain images
    /// belong to their swapchain.
    owned: bool,
    /// The view's slots in the heap: sampled with sampled usage, and storage
    /// with storage usage when its view type has a storage array.
    slots: [Option<Slot>; 2],
}

impl Image {
    /// Creates an image for a validated description, with a view of type
    /// `view` if given.
    pub(crate) fn new(
        device: &Arc<Device>,
        desc: &ImageDesc<'_>,
        view: Option<ViewType>,
    ) -> crate::Result<Self> {
        let (image_type, flags) = match desc.dimension {
            ImageDimension::D2 => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
            ImageDimension::D3 => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
            ImageDimension::Cube => (
                vk::ImageType::TYPE_2D,
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
        };
        let format = desc.format.vk();
        let info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(desc.samples.vk())
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage.vk())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `info` is valid for the device.
        let raw = unsafe { device.raw.create_image(&info, None) }.check("vkCreateImage")?;
        // Owned from here on, so a failure below destroys what exists.
        let mut image = Self {
            device: device.clone(),
            raw,
            allocation: Allocation::default(),
            view: None,
            owned: true,
            slots: [None; 2],
        };
        // SAFETY: `raw` was created from this device.
        let requirements = unsafe { device.raw.get_image_memory_requirements(raw) };
        image.allocation =
            device.allocate(desc.label, requirements, MemoryLocation::GpuOnly, false)?;
        // SAFETY: the allocation satisfies the image's requirements and is
        // bound once.
        unsafe {
            device
                .raw
                .bind_image_memory(raw, image.allocation.memory(), image.allocation.offset())
        }
        .check("vkBindImageMemory")?;
        device.name(raw, desc.label);

        if let Some(view_type) = view {
            let view = image.create_view(view_type, desc.format, desc.label)?;
            let storage = desc.usage.contains(ImageUsage::STORAGE);
            for ((usage, binding), slot) in view_type
                .heap_bindings(desc.format)
                .into_iter()
                .zip(&mut image.slots)
            {
                if let Some(binding) = binding.filter(|_| desc.usage.contains(usage)) {
                    let layout = if usage == ImageUsage::STORAGE {
                        vk::ImageLayout::GENERAL
                    } else {
                        ImageAccess::sampled_layout(storage)
                    };
                    *slot = Some(device.add_to_heap(binding, Descriptor::Image(view, layout))?);
                }
            }
        }
        Ok(image)
    }

    /// Identifies the image while it exists: until the GPU is done with it.
    pub(crate) fn id(&self) -> ImageId {
        self.raw
    }

    /// Wraps the swapchain image `raw` of `format`, which its swapchain owns,
    /// with a view.
    pub(super) fn swapchain(
        device: &Arc<Device>,
        raw: vk::Image,
        format: Format,
        label: &str,
    ) -> crate::Result<Self> {
        // Dropping it destroys only the view.
        let mut image = Self {
            device: device.clone(),
            raw,
            allocation: Allocation::default(),
            view: None,
            owned: false,
            slots: [None; 2],
        };
        image.create_view(ViewType::D2, format, label)?;
        Ok(image)
    }

    /// Creates the image's view of every mip and layer, as `view_type`.
    fn create_view(
        &mut self,
        view_type: ViewType,
        format: Format,
        label: &str,
    ) -> crate::Result<vk::ImageView> {
        let info = vk::ImageViewCreateInfo::default()
            .image(self.raw)
            .view_type(view_type.vk())
            .format(format.vk())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: format.vk_aspects(),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            });
        // SAFETY: the image is bound to memory and the view matches it.
        let view =
            unsafe { self.device.raw.create_image_view(&info, None) }.check("vkCreateImageView")?;
        self.device.name(view, label);
        self.view = Some(view);
        Ok(view)
    }

    /// The view rendered to when the image is an attachment, which its
    /// attachment usage gave it.
    pub(super) fn attachment_view(&self) -> vk::ImageView {
        self.view.unwrap_or_default()
    }

    /// The heap slots of the image's view: sampled and storage.
    pub(crate) fn slots(&self) -> [Option<Slot>; 2] {
        self.slots
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // Its first transition, if still queued, stays until the next frame
        // begins, before the image, or a swapchain image's swapchain, is
        // destroyed: commands recorded before the drop expect it. Noted
        // before burying the image, so a frame that begins meanwhile and
        // misses it tags it.
        self.device.forget_initialization_next_frame(self.raw);
        self.device.bury(Garbage::Image {
            raw: self.owned.then_some(self.raw),
            view: self.view,
            allocation: std::mem::take(&mut self.allocation),
            slots: self.slots,
        });
    }
}

impl ImageUsage {
    /// The equivalent Vulkan usage.
    pub(super) fn vk(self) -> vk::ImageUsageFlags {
        [
            (Self::COPY_SRC, vk::ImageUsageFlags::TRANSFER_SRC),
            (Self::COPY_DST, vk::ImageUsageFlags::TRANSFER_DST),
            (Self::SAMPLED, vk::ImageUsageFlags::SAMPLED),
            (Self::STORAGE, vk::ImageUsageFlags::STORAGE),
            (
                Self::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                Self::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
        ]
        .into_iter()
        .filter(|(usage, _)| self.contains(*usage))
        .fold(vk::ImageUsageFlags::empty(), |all, (_, usage)| all | usage)
    }
}

impl SampleCount {
    /// The equivalent Vulkan sample count.
    pub(super) const fn vk(self) -> vk::SampleCountFlags {
        match self {
            Self::X1 => vk::SampleCountFlags::TYPE_1,
            Self::X2 => vk::SampleCountFlags::TYPE_2,
            Self::X4 => vk::SampleCountFlags::TYPE_4,
            Self::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

impl ViewType {
    /// The equivalent Vulkan view type.
    const fn vk(self) -> vk::ImageViewType {
        match self {
            Self::D2 => vk::ImageViewType::TYPE_2D,
            Self::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::D3 => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}
