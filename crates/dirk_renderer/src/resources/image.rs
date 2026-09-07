//! Renderer images and texture uploads backed by the RHI.

use dirk_rhi::{
    DependencyInfo, Extent3d, FilterMode, ImageBarrier, ImageBlit, ImageDesc, ImageDimension,
    ImageState, ImageUsages, Origin3d, SampleCount, SamplerDesc, TextureFormat,
};

use crate::{
    Result,
    models::Texture,
    resources::{
        ActiveCommandBuffer, ActiveImage, ActiveImageView,
        device::RenderDevice,
        upload::{ImageUpload, RgbaMip},
    },
};

pub struct Image {
    inner: ActiveImage,
    view: ActiveImageView,
}

pub struct ImageCreateInfo {
    pub extent: Extent3d,
    pub format: TextureFormat,
    pub usage: ImageUsages,
    pub mip_levels: u32,
    pub samples: SampleCount,
}

impl Image {
    pub fn create_image(device: &RenderDevice, info: &ImageCreateInfo) -> Result<Self> {
        if info.mip_levels == 0 {
            return Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty).into());
        }
        let inner = device.rhi.create_image(&ImageDesc {
            label: "renderer image",
            dimension: ImageDimension::TwoD,
            extent: info.extent,
            format: info.format,
            usage: info.usage,
            mip_levels: info.mip_levels,
            array_layers: 1,
            samples: info.samples,
        })?;
        let view = device.rhi.view(&inner)?;
        Ok(Self { inner, view })
    }

    pub(crate) fn rhi_image(&self) -> &ActiveImage {
        &self.inner
    }

    pub(crate) fn rhi_view(&self) -> &ActiveImageView {
        &self.view
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
                return Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Mismatch).into());
            }
        };
        let mip_levels = Self::mip_levels(texture.width, texture.height);
        let image = Self::create_image(
            device,
            &ImageCreateInfo {
                extent: Extent3d::new_2d(texture.width, texture.height),
                format: TextureFormat::Rgba8Srgb,
                usage: ImageUsages::COPY_DST | ImageUsages::COPY_SRC | ImageUsages::SAMPLED,
                mip_levels,
                samples: SampleCount::One,
            },
        )?;
        let mut command = device.graphics_pool.begin_single_time()?;
        let mut mip = RgbaMip {
            width: texture.width,
            height: texture.height,
            pixels,
        };
        device.upload_image(
            &mut command,
            &ImageUpload {
                image: image.rhi_image(),
                mip: 0,
                origin: Origin3d::default(),
                extent: Extent3d::new_2d(mip.width, mip.height),
                pixels: &mip.pixels,
            },
        )?;
        if device
            .rhi
            .format_capabilities(TextureFormat::Rgba8Srgb)
            .blit
            .supports(FilterMode::Linear)
        {
            image.record_mips(&mut command, texture.width, texture.height)?;
        } else {
            for level in 1..mip_levels {
                mip = mip.next();
                device.upload_image(
                    &mut command,
                    &ImageUpload {
                        image: image.rhi_image(),
                        mip: level,
                        origin: Origin3d::default(),
                        extent: Extent3d::new_2d(mip.width, mip.height),
                        pixels: &mip.pixels,
                    },
                )?;
            }
            command.transition(
                image.rhi_image(),
                ImageState::ShaderRead(dirk_rhi::ShaderStages::FRAGMENT),
                dirk_rhi::ImageSubresourceRange::WHOLE,
            )?;
        }
        device.graphics_pool.submit_and_wait(command)?;

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
        for mip in 1..self.inner.description().mip_levels {
            command.barrier(&DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[ImageBarrier {
                    image: &self.inner,
                    old_state: ImageState::CopyDestination,
                    new_state: ImageState::CopySource,
                    aspects: self.inner.description().format.aspects(),
                    base_mip_level: mip - 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    queue_transfer: None,
                }],
            })?;
            command.blit_image(
                &self.inner,
                &self.inner,
                &[ImageBlit {
                    src_mip_level: mip - 1,
                    dst_mip_level: mip,
                    src_base_array_layer: 0,
                    dst_base_array_layer: 0,
                    array_layer_count: 1,
                    src_origin: dirk_rhi::Origin3d::default(),
                    dst_origin: dirk_rhi::Origin3d::default(),
                    src_extent: Extent3d::new_2d(
                        (width >> (mip - 1)).max(1),
                        (height >> (mip - 1)).max(1),
                    ),
                    dst_extent: Extent3d::new_2d((width >> mip).max(1), (height >> mip).max(1)),
                    aspects: self.inner.description().format.aspects(),
                }],
                FilterMode::Linear,
            )?;
            command.barrier(&DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[ImageBarrier {
                    image: &self.inner,
                    old_state: ImageState::CopySource,
                    new_state: ImageState::ShaderRead(dirk_rhi::ShaderStages::FRAGMENT),
                    aspects: self.inner.description().format.aspects(),
                    base_mip_level: mip - 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    queue_transfer: None,
                }],
            })?;
        }
        command.barrier(&DependencyInfo {
            memory_barriers: &[],
            buffer_barriers: &[],
            image_barriers: &[ImageBarrier {
                image: &self.inner,
                old_state: ImageState::CopyDestination,
                new_state: ImageState::ShaderRead(dirk_rhi::ShaderStages::FRAGMENT),
                aspects: self.inner.description().format.aspects(),
                base_mip_level: self.inner.description().mip_levels - 1,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                queue_transfer: None,
            }],
        })?;
        Ok(())
    }

    fn mip_levels(width: u32, height: u32) -> u32 {
        u32::BITS - width.max(height).max(1).leading_zeros()
    }
}
