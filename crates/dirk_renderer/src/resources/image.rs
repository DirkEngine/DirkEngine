//! Renderer images and texture uploads backed by the RHI.

use ash::vk;
use dirk_rhi::{
    BufferImageCopy, CommandBuffer as _, DependencyInfo, Extent3d, FilterMode, ImageAspects,
    ImageBarrier, ImageBlit, ImageDesc, ImageState, ImageUsages, ImageViewDesc, ImageViewType,
    SampleCount, SamplerDesc,
};
use dirk_rhi_vulkan::{VulkanImage, VulkanImageView};
use gpu_allocator::MemoryLocation;

use crate::{
    Result,
    models::Texture,
    resources::{buffer::CustomBuffer, device::RenderDevice},
};

pub struct Image {
    inner: VulkanImage,
    view: VulkanImageView,
    aspects: ImageAspects,
    mip_levels: u32,
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
        if info.mip_levels == 0 {
            return Err(dirk_rhi::Error::InvalidResource(
                "renderer images require at least one mip level",
            )
            .into());
        }
        if info.tiling != vk::ImageTiling::OPTIMAL || info.location != MemoryLocation::GpuOnly {
            return Err(dirk_rhi::Error::Unsupported(
                "renderer images currently require optimal GPU-local memory",
            )
            .into());
        }
        let format = rhi_format(info.format)?;
        let aspects = rhi_aspects(info.aspect_flags);
        let inner = device.rhi.create_image(&ImageDesc {
            label: "renderer image",
            extent: Extent3d::new_2d(info.size.width, info.size.height),
            format,
            usage: rhi_usage(info.usage),
            mip_levels: info.mip_levels,
            array_layers: 1,
            samples: rhi_samples(info.num_samples)?,
        })?;
        let view = device.rhi.create_image_view(&ImageViewDesc {
            label: "renderer image view",
            image: &inner,
            view_type: ImageViewType::TwoD,
            aspects,
            base_mip_level: 0,
            mip_level_count: info.mip_levels,
            base_array_layer: 0,
            array_layer_count: 1,
        })?;
        Ok(Self {
            inner,
            view,
            aspects,
            mip_levels: info.mip_levels,
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.view.raw()
    }

    pub(crate) fn rhi_image(&self) -> &VulkanImage {
        &self.inner
    }

    pub(crate) fn rhi_view(&self) -> &VulkanImageView {
        &self.view
    }

    pub(crate) fn rhi_aspects(&self) -> ImageAspects {
        self.aspects
    }

    pub fn upload_texture(device: &RenderDevice, texture: &gltf::image::Data) -> Result<Texture> {
        let pixels = match texture.format {
            gltf::image::Format::R8G8B8 => texture
                .pixels
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
            gltf::image::Format::R8G8B8A8 => texture.pixels.clone(),
            _ => {
                return Err(dirk_rhi::Error::Unsupported(
                    "only RGB8 and RGBA8 glTF textures are supported",
                )
                .into());
            }
        };
        let mip_levels = Self::mip_levels(texture.width, texture.height);
        let staging = CustomBuffer::create_custom(
            device,
            u64::try_from(pixels.len())
                .map_err(|_| dirk_rhi::Error::InvalidResource("texture upload is too large"))?,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        device.rhi.write_buffer(staging.rhi(), 0, &pixels)?;

        let image = Self::create_image(
            device,
            &ImageCreateInfo {
                size: vk::Extent2D {
                    width: texture.width,
                    height: texture.height,
                },
                format: vk::Format::R8G8B8A8_SRGB,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                location: MemoryLocation::GpuOnly,
                mip_levels,
                num_samples: vk::SampleCountFlags::TYPE_1,
                aspect_flags: vk::ImageAspectFlags::COLOR,
            },
        )?;
        let mut command = device.graphics_pool.begin_single_time()?;
        command.rhi_mut().barrier(&DependencyInfo {
            image_barriers: &[ImageBarrier {
                image: image.rhi_image(),
                old_state: ImageState::Undefined,
                new_state: ImageState::CopyDestination,
                aspects: ImageAspects::COLOR,
                base_mip_level: 0,
                mip_level_count: mip_levels,
                base_array_layer: 0,
                array_layer_count: 1,
            }],
        });
        command.rhi_mut().copy_buffer_to_image(
            staging.rhi(),
            image.rhi_image(),
            &[BufferImageCopy {
                buffer_offset: 0,
                mip_level: 0,
                base_array_layer: 0,
                array_layer_count: 1,
                extent: Extent3d::new_2d(texture.width, texture.height),
                aspects: ImageAspects::COLOR,
            }],
        );
        image.record_mips(command.rhi_mut(), texture.width, texture.height)?;
        command.end_and_submit()?;

        let sampler = device.rhi.create_sampler(&SamplerDesc {
            label: "renderer texture sampler",
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: dirk_rhi::AddressMode::Repeat,
            address_v: dirk_rhi::AddressMode::Repeat,
            address_w: dirk_rhi::AddressMode::Repeat,
            max_anisotropy: device.rhi.capabilities().max_sampler_anisotropy,
            lod_min: 0.0,
            // Renderer textures cannot approach the precision limit of an f32.
            #[allow(clippy::cast_precision_loss)]
            lod_max: mip_levels as f32,
        })?;
        Ok(Texture { image, sampler })
    }

    fn record_mips(
        &self,
        command: &mut dirk_rhi_vulkan::VulkanCommandBuffer,
        width: u32,
        height: u32,
    ) -> dirk_rhi::Result<()> {
        for mip in 1..self.mip_levels {
            command.barrier(&DependencyInfo {
                image_barriers: &[ImageBarrier {
                    image: &self.inner,
                    old_state: ImageState::CopyDestination,
                    new_state: ImageState::CopySource,
                    aspects: self.aspects,
                    base_mip_level: mip - 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                }],
            });
            command.blit_image(
                &self.inner,
                &self.inner,
                &[ImageBlit {
                    src_mip_level: mip - 1,
                    dst_mip_level: mip,
                    src_extent: Extent3d::new_2d(
                        (width >> (mip - 1)).max(1),
                        (height >> (mip - 1)).max(1),
                    ),
                    dst_extent: Extent3d::new_2d((width >> mip).max(1), (height >> mip).max(1)),
                }],
                FilterMode::Linear,
            )?;
            command.barrier(&DependencyInfo {
                image_barriers: &[ImageBarrier {
                    image: &self.inner,
                    old_state: ImageState::CopySource,
                    new_state: ImageState::ShaderRead,
                    aspects: self.aspects,
                    base_mip_level: mip - 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                }],
            });
        }
        command.barrier(&DependencyInfo {
            image_barriers: &[ImageBarrier {
                image: &self.inner,
                old_state: ImageState::CopyDestination,
                new_state: ImageState::ShaderRead,
                aspects: self.aspects,
                base_mip_level: self.mip_levels - 1,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
            }],
        });
        Ok(())
    }

    fn mip_levels(width: u32, height: u32) -> u32 {
        u32::BITS - width.max(height).max(1).leading_zeros()
    }
}

pub(crate) fn rhi_format(format: vk::Format) -> Result<dirk_rhi::Format> {
    Ok(match format {
        vk::Format::R8G8B8A8_UNORM => dirk_rhi::Format::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => dirk_rhi::Format::Rgba8Srgb,
        vk::Format::B8G8R8A8_UNORM => dirk_rhi::Format::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => dirk_rhi::Format::Bgra8Srgb,
        vk::Format::D16_UNORM => dirk_rhi::Format::Depth16Unorm,
        vk::Format::D24_UNORM_S8_UINT => dirk_rhi::Format::Depth24UnormStencil8,
        vk::Format::D32_SFLOAT => dirk_rhi::Format::Depth32Float,
        vk::Format::D32_SFLOAT_S8_UINT => dirk_rhi::Format::Depth32FloatStencil8,
        _ => return Err(dirk_rhi::Error::Unsupported("renderer image format").into()),
    })
}

pub(crate) fn rhi_usage(usage: vk::ImageUsageFlags) -> ImageUsages {
    let mut result = ImageUsages::NONE;
    for (vulkan, rhi) in [
        (vk::ImageUsageFlags::TRANSFER_SRC, ImageUsages::COPY_SRC),
        (vk::ImageUsageFlags::TRANSFER_DST, ImageUsages::COPY_DST),
        (vk::ImageUsageFlags::SAMPLED, ImageUsages::SAMPLED),
        (vk::ImageUsageFlags::STORAGE, ImageUsages::STORAGE),
        (
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ImageUsages::COLOR_ATTACHMENT,
        ),
        (
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageUsages::DEPTH_STENCIL_ATTACHMENT,
        ),
        (
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            ImageUsages::TRANSIENT_ATTACHMENT,
        ),
    ] {
        if usage.contains(vulkan) {
            result |= rhi;
        }
    }
    result
}

pub(crate) fn rhi_samples(samples: vk::SampleCountFlags) -> Result<SampleCount> {
    Ok(match samples {
        vk::SampleCountFlags::TYPE_1 => SampleCount::One,
        vk::SampleCountFlags::TYPE_2 => SampleCount::Two,
        vk::SampleCountFlags::TYPE_4 => SampleCount::Four,
        vk::SampleCountFlags::TYPE_8 => SampleCount::Eight,
        _ => return Err(dirk_rhi::Error::Unsupported("renderer sample count").into()),
    })
}

fn rhi_aspects(aspects: vk::ImageAspectFlags) -> ImageAspects {
    let mut result = ImageAspects::NONE;
    if aspects.contains(vk::ImageAspectFlags::COLOR) {
        result |= ImageAspects::COLOR;
    }
    if aspects.contains(vk::ImageAspectFlags::DEPTH) {
        result |= ImageAspects::DEPTH;
    }
    if aspects.contains(vk::ImageAspectFlags::STENCIL) {
        result |= ImageAspects::STENCIL;
    }
    result
}
