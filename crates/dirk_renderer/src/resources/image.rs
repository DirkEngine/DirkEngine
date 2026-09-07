//! This module houses the vulkan image abstraction

pub mod layouts;
pub mod mips;

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::{
    Renderer, Result,
    models::Texture,
    resources::{
        buffer::CustomBuffer,
        device::{Garbage, RenderDevice},
    },
};

/// An abstraction around vulkan windows.
pub struct Image {
    device: RenderDevice,
    raw: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    aspect_flags: vk::ImageAspectFlags,
}

pub struct ImageCreateInfo {
    pub size: vk::Extent2D,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub location: MemoryLocation,
    pub mip_levels: u32,
    pub num_samples: vk::SampleCountFlags,
    pub aspect_flags: vk::ImageAspectFlags,
}

impl Image {
    pub fn create_image(device: &RenderDevice, info: &ImageCreateInfo) -> Result<Self> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(info.format)
            .extent(vk::Extent3D {
                width: info.size.width,
                height: info.size.height,
                depth: 1,
            })
            .mip_levels(info.mip_levels)
            .array_layers(1)
            .samples(info.num_samples)
            .tiling(info.tiling)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.device.create_image(&image_info, None)? };
        let requirements = unsafe { device.device.get_image_memory_requirements(image) };

        let allocation = device.allocate(&AllocationCreateDesc {
            name: "image",
            requirements,
            location: info.location,
            linear: info.tiling == vk::ImageTiling::LINEAR,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            device
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        };

        Ok(Self {
            device: device.clone(),
            raw: image,
            view: Self::create_image_view(
                device,
                image,
                info.format,
                info.aspect_flags,
                info.mip_levels,
            )?,
            allocation: Some(allocation),
            aspect_flags: info.aspect_flags,
        })
    }
    pub fn image(&self) -> vk::Image {
        self.raw
    }
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    pub fn aspect_flags(&self) -> vk::ImageAspectFlags {
        self.aspect_flags
    }

    pub fn upload_texture(device: &RenderDevice, tex: &gltf::image::Data) -> Result<Texture> {
        let pixels = match tex.format {
            gltf::image::Format::R8G8B8 => tex
                .pixels
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
            gltf::image::Format::R8G8B8A8 => tex.pixels.clone(),
            fmt => panic!("Unsuported glTF image format: {fmt:?}"),
        };

        let mip_levels = Self::mip_levels(tex.width, tex.height);
        let size = (pixels.len()) as vk::DeviceSize;
        let format = vk::Format::R8G8B8A8_SRGB;

        let staging_buf = CustomBuffer::create_custom(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;

        unsafe {
            let ptr = staging_buf
                .mapped()
                .expect("the buffer should be host visible")
                .as_ptr()
                .cast::<u8>();
            ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        }

        let create_info = ImageCreateInfo {
            size: vk::Extent2D {
                width: tex.width,
                height: tex.height,
            },
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            // TRANSFER_SRC needed for mip creation
            usage: vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            location: MemoryLocation::GpuOnly,
            mip_levels,
            num_samples: vk::SampleCountFlags::TYPE_1,
            aspect_flags: vk::ImageAspectFlags::COLOR,
        };
        let mut image = Self::create_image(device, &create_info)?;

        let cmd = device.graphics_pool.begin_single_time()?;

        image.transition_image_layout(
            &cmd,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels, // all mip levels start undefined
            0,
        )?;

        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: tex.width,
                height: tex.height,
                depth: 1,
            });

        cmd.copy_buffer_to_image(
            staging_buf.buffer(),
            image.image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        image.generate_mipmaps(&cmd, tex.width, tex.height, mip_levels)?;
        cmd.end_and_submit(&device.queues)?;

        let old_view = image.view;
        image.view = Self::create_image_view(
            device,
            image.image(),
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;
        unsafe {
            device.device.destroy_image_view(old_view, None);
        }
        let sampler = Renderer::create_sampler(device, mip_levels)?;

        Ok(Texture {
            device: device.clone(),
            image,
            sampler,
        })
    }

    /// Utility function. Not a member as it is used to create swapchain images
    fn create_image_view(
        device: &RenderDevice,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Result<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        Ok(unsafe { device.device.create_image_view(&create_info, None)? })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        self.device.destroy(Garbage::Image(self.raw));
        self.device.destroy(Garbage::ImageView(self.view));
        if let Some(alloc) = self.allocation.take() {
            self.device.destroy(Garbage::Allocation(alloc));
        }
    }
}
