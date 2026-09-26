//! Renderer images and texture uploads backed by the RHI.

use anyhow::Context;
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
        let pixels = Self::rgba8(texture)?;
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

    /// Convert the 8- and 16-bit formats produced by glTF import to the
    /// renderer's RGBA8 texture format. Float images need tone mapping and are
    /// rejected with an asset error instead of being silently misinterpreted.
    fn rgba8(texture: &gltf::image::Data) -> Result<Vec<u8>> {
        use gltf::image::Format;
        let (channels, bytes_per_channel): (usize, usize) = match texture.format {
            Format::R8 => (1, 1),
            Format::R8G8 => (2, 1),
            Format::R8G8B8 => (3, 1),
            Format::R8G8B8A8 => (4, 1),
            Format::R16 => (1, 2),
            Format::R16G16 => (2, 2),
            Format::R16G16B16 => (3, 2),
            Format::R16G16B16A16 => (4, 2),
            Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
                return Err(anyhow::anyhow!(
                    "floating-point glTF base-color images are unsupported"
                )
                .into());
            }
        };
        let pixel_count = usize::try_from(texture.width)
            .ok()
            .and_then(|width| {
                usize::try_from(texture.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .context("glTF texture dimensions exceed host address space")?;
        let stride = channels * bytes_per_channel;
        if texture.pixels.len()
            != pixel_count
                .checked_mul(stride)
                .context("glTF texture byte count overflows")?
        {
            return Err(anyhow::anyhow!("glTF texture pixel data has an invalid length").into());
        }
        let capacity = pixel_count
            .checked_mul(4)
            .context("glTF RGBA texture byte count overflows")?;
        let mut rgba = Vec::with_capacity(capacity);
        for pixel in texture.pixels.chunks_exact(stride) {
            let component = |index: usize| {
                let begin = index * bytes_per_channel;
                if bytes_per_channel == 1 {
                    pixel[begin]
                } else {
                    u16::from_ne_bytes([pixel[begin], pixel[begin + 1]]).to_be_bytes()[0]
                }
            };
            let red = component(0);
            let (green, blue, alpha) = match channels {
                1 => (red, red, 255),
                2 => (red, red, component(1)),
                3 => (component(1), component(2), 255),
                4 => (component(1), component(2), component(3)),
                _ => unreachable!(),
            };
            rgba.extend_from_slice(&[red, green, blue, alpha]);
        }
        Ok(rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::Image;

    #[test]
    fn converts_grayscale_alpha_and_sixteen_bit_images() {
        let grayscale = gltf::image::Data {
            pixels: vec![64, 128],
            format: gltf::image::Format::R8G8,
            width: 1,
            height: 1,
        };
        assert_eq!(Image::rgba8(&grayscale).unwrap(), [64, 64, 64, 128]);

        let high_depth = gltf::image::Data {
            pixels: [0x1234_u16, 0xabcd, 0xffff]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect(),
            format: gltf::image::Format::R16G16B16,
            width: 1,
            height: 1,
        };
        assert_eq!(Image::rgba8(&high_depth).unwrap(), [0x12, 0xab, 0xff, 0xff]);
    }

    #[test]
    fn invalid_image_length_is_an_error() {
        let image = gltf::image::Data {
            pixels: vec![1, 2],
            format: gltf::image::Format::R8G8B8,
            width: 1,
            height: 1,
        };
        assert!(Image::rgba8(&image).is_err());
    }
}
