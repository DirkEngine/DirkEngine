//! Renderer images and texture uploads backed by the RHI.

use dirk_rhi::{
    Extent3d, FilterMode, ImageDesc, ImageDimension, ImageUsages, SampleCount, SamplerDesc,
    TextureFormat,
};

use crate::{
    Result,
    models::Texture,
    resources::{ImageView, Rhi, RhiImage, upload::RgbaMip},
};

pub struct Image {
    inner: RhiImage,
    view: ImageView,
}

pub struct ImageCreateInfo {
    pub extent: Extent3d,
    pub format: TextureFormat,
    pub usage: ImageUsages,
    pub mip_levels: u32,
    pub samples: SampleCount,
}

impl Image {
    pub fn create_image(device: &Rhi, info: &ImageCreateInfo) -> Result<Self> {
        if info.mip_levels == 0 {
            return Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Empty).into());
        }
        let inner = device.create_image(&ImageDesc {
            label: "renderer image",
            dimension: ImageDimension::TwoD,
            extent: info.extent,
            format: info.format,
            usage: info.usage,
            mip_levels: info.mip_levels,
            array_layers: 1,
            samples: info.samples,
        })?;
        let view = device.view(&inner)?;
        Ok(Self { inner, view })
    }

    pub(crate) fn rhi_image(&self) -> &RhiImage {
        &self.inner
    }

    pub(crate) fn rhi_view(&self) -> &ImageView {
        &self.view
    }

    pub fn upload_texture(
        device: &Rhi,
        uploads: &mut dirk_render_utils::upload::UploadBatch,
        texture: &gltf::image::Data,
    ) -> Result<Texture> {
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
        let mut mip = RgbaMip {
            width: texture.width,
            height: texture.height,
            pixels,
        };
        let mut levels = Vec::with_capacity(mip_levels as usize);
        for level in 0..mip_levels {
            if level != 0 {
                mip = mip.next();
            }
            levels.push(mip.pixels.clone());
        }
        let slices = levels.iter().map(Vec::as_slice).collect::<Vec<_>>();
        // SAFETY: a new allocation with no previous GPU use; this owner remains in the texture registry.
        unsafe {
            uploads.image(device, image.rhi_image(), &slices)?;
        }

        let sampler = device.create_sampler(&SamplerDesc {
            label: "renderer texture sampler",
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: dirk_rhi::AddressMode::Repeat,
            address_v: dirk_rhi::AddressMode::Repeat,
            address_w: dirk_rhi::AddressMode::Repeat,
            max_anisotropy: device.capabilities().max_sampler_anisotropy,
            lod_min: 0.0,
            // Renderer textures cannot approach the precision limit of an f32.
            #[allow(clippy::cast_precision_loss)]
            lod_max: mip_levels as f32,
        })?;
        Ok(Texture { image, sampler })
    }

    fn mip_levels(width: u32, height: u32) -> u32 {
        u32::BITS - width.max(height).max(1).leading_zeros()
    }
}
