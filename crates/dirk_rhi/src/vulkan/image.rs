use std::sync::Arc;

use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::Allocation};

use super::{
    Check, Device,
    device::Garbage,
    heap::{Descriptor, Slot},
};
use crate::{
    Format, ImageDesc, ImageDimension, ImageRange, ImageUsage, SampleCount, ViewType, image::ViewOf,
};

/// Identifies an image while it exists.
pub(crate) type ImageId = vk::Image;

/// A Vulkan image, its memory, its view of every mip and layer and, when
/// rendered to or stored per mip or layer, its view of each mip and layer.
pub(crate) struct Image {
    pub(super) device: Arc<Device>,
    pub(super) raw: vk::Image,
    allocation: Allocation,
    view: Option<View>,
    /// Single-mip, single-layer 2D views, layer by layer, mip by mip.
    subviews: Vec<View>,
    /// Whether the image is destroyed with this object; swapchain images
    /// belong to their swapchain.
    owned: bool,
}

/// An image view and its slots in the heap: sampled with sampled usage, and
/// storage with storage usage when its view type has a storage array.
pub(super) struct View {
    pub(super) raw: vk::ImageView,
    pub(super) slots: [Option<Slot>; 2],
}

impl Image {
    /// Creates an image for a validated description, with a view of type
    /// `view` if given, and subviews if the description has them.
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
            subviews: Vec::new(),
            owned: true,
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
            image.view = Some(image.create_view(desc, desc.usage, view_type, ImageRange::ALL)?);
        }
        for range in desc.subview_ranges() {
            let view = image.create_view(desc, desc.subview_usage(), ViewType::D2, range)?;
            image.subviews.push(view);
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
            subviews: Vec::new(),
            owned: false,
        };
        image.view = Some(View {
            raw: image.create_raw_view(format, ViewType::D2, ImageRange::ALL, label)?,
            slots: [None; 2],
        });
        Ok(image)
    }

    /// Creates a view of `range` as `view_type`, in the heap arrays `usage`,
    /// part of the usage of `desc`, needs.
    fn create_view(
        &self,
        desc: &ImageDesc<'_>,
        usage: ImageUsage,
        view_type: ViewType,
        range: ImageRange,
    ) -> crate::Result<View> {
        let raw = self.create_raw_view(desc.format, view_type, range, desc.label)?;
        let mut view = View {
            raw,
            slots: [None; 2],
        };
        for ((needed, binding), slot) in view_type
            .heap_bindings(desc.format)
            .into_iter()
            .zip(&mut view.slots)
        {
            if let Some(binding) = binding.filter(|_| usage.contains(needed)) {
                match self.device.add_to_heap(binding, Descriptor::Image(raw)) {
                    Ok(added) => *slot = Some(added),
                    Err(error) => {
                        // Nothing uses the view yet; bury it with its slots.
                        self.device.bury(Garbage::View(view));
                        return Err(error);
                    }
                }
            }
        }
        Ok(view)
    }

    /// Creates a native view of `range` as `view_type`.
    fn create_raw_view(
        &self,
        format: Format,
        view_type: ViewType,
        range: ImageRange,
        label: &str,
    ) -> crate::Result<vk::ImageView> {
        let info = vk::ImageViewCreateInfo::default()
            .image(self.raw)
            .view_type(view_type.vk())
            .format(format.vk())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: format.vk_aspects(),
                base_mip_level: range.base_mip_level,
                level_count: range.mip_level_count,
                base_array_layer: range.base_layer,
                layer_count: range.layer_count,
            });
        // SAFETY: the image is bound to memory and the view matches it.
        let view =
            unsafe { self.device.raw.create_image_view(&info, None) }.check("vkCreateImageView")?;
        self.device.name(view, label);
        Ok(view)
    }

    /// The view rendered to, which the image's usage and shape gave it.
    pub(super) fn attachment_view(&self, view: ViewOf) -> vk::ImageView {
        match view {
            ViewOf::Full => self.view.as_ref().map_or_default(|view| view.raw),
            ViewOf::Subview(index) => self.subviews[index].raw,
        }
    }

    /// The heap slots of the image's full view: sampled and storage.
    pub(crate) fn slots(&self) -> [Option<u32>; 2] {
        self.view.as_ref().map_or([None; 2], View::slot_indices)
    }

    /// The heap slots of each subview, in order.
    pub(crate) fn subview_slots(&self) -> impl Iterator<Item = [Option<u32>; 2]> {
        self.subviews.iter().map(View::slot_indices)
    }
}

impl View {
    /// The indices of the view's slots.
    fn slot_indices(&self) -> [Option<u32>; 2] {
        self.slots.map(|slot| slot.map(|slot| slot.index))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        self.device.forget_initialization(self.raw);
        self.device.bury(Garbage::Image {
            raw: self.owned.then_some(self.raw),
            views: self
                .view
                .take()
                .into_iter()
                .chain(self.subviews.drain(..))
                .collect(),
            allocation: std::mem::take(&mut self.allocation),
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
