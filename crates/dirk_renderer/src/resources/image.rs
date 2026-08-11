//! Renderer images and texture uploads backed by the RHI.

use dirk_rhi::{
    Backend as _, Buffer as _, BufferImageCopy, CommandBuffer as _, DependencyInfo, Extent3d,
    FilterMode, Format, ImageAspects, ImageBarrier, ImageBlit, ImageDesc, ImageState, ImageUsages,
    ImageViewDesc, ImageViewType, MemoryDomain, SampleCount, SamplerDesc,
};

use crate::{
    Result,
    models::Texture,
    resources::{
        ActiveCommandBuffer, ActiveImage, ActiveImageView, buffer::CustomBuffer,
        device::RenderDevice,
    },
};

pub struct Image {
    inner: ActiveImage,
    view: ActiveImageView,
    aspects: ImageAspects,
    mip_levels: u32,
}

pub struct ImageCreateInfo {
    pub extent: Extent3d,
    pub format: Format,
    pub usage: ImageUsages,
    pub mip_levels: u32,
    pub samples: SampleCount,
    pub aspects: ImageAspects,
}

impl Image {
    pub fn create_image(device: &RenderDevice, info: &ImageCreateInfo) -> Result<Self> {
        if info.mip_levels == 0 {
            return Err(dirk_rhi::Error::InvalidResource(
                "renderer images require at least one mip level",
            )
            .into());
        }
        let inner = device.rhi.create_image(&ImageDesc {
            label: "renderer image",
            extent: info.extent,
            format: info.format,
            usage: info.usage,
            mip_levels: info.mip_levels,
            array_layers: 1,
            samples: info.samples,
        })?;
        let view = device.rhi.create_image_view(&ImageViewDesc {
            label: "renderer image view",
            image: &inner,
            view_type: ImageViewType::TwoD,
            aspects: info.aspects,
            base_mip_level: 0,
            mip_level_count: info.mip_levels,
            base_array_layer: 0,
            array_layer_count: 1,
        })?;
        Ok(Self {
            inner,
            view,
            aspects: info.aspects,
            mip_levels: info.mip_levels,
        })
    }

    pub(crate) fn rhi_image(&self) -> &ActiveImage {
        &self.inner
    }

    pub(crate) fn rhi_view(&self) -> &ActiveImageView {
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
            dirk_rhi::BufferUsages::COPY_SRC,
            MemoryDomain::Upload,
        )?;
        staging.rhi().write(0, &pixels)?;

        let image = Self::create_image(
            device,
            &ImageCreateInfo {
                extent: Extent3d::new_2d(texture.width, texture.height),
                format: Format::Rgba8Srgb,
                usage: ImageUsages::COPY_DST | ImageUsages::COPY_SRC | ImageUsages::SAMPLED,
                mip_levels,
                samples: SampleCount::One,
                aspects: ImageAspects::COLOR,
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
        command: &mut ActiveCommandBuffer,
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
