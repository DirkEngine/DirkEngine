//! Texture mip chain generation.
//!
//! Mips are generated on the GPU with filtered blits recorded on the graphics
//! queue. Formats the device cannot blit with linear filtering fall back to
//! [`RgbaMip`], which prepares every level on the CPU.

use std::sync::LazyLock;

use dirk_render_utils::upload::ImageUpload;
use dirk_rhi::{
    CommandEncoder, DependencyInfo, FilterMode, ImageAspects, ImageBarrier, ImageBlit, ImageState,
    Origin3d, ShaderStages, TextureFormat,
};

use crate::{
    Result,
    resources::{RecordedCommands, Rhi, RhiImage},
};

/// Graphics-queue recording that uploads base texture levels and blits their
/// remaining mips.
#[derive(Default)]
pub struct MipGeneration {
    encoder: Option<CommandEncoder>,
}

impl MipGeneration {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether mips of `format` can be generated with filtered GPU blits.
    pub fn supports(rhi: &Rhi, format: TextureFormat) -> bool {
        rhi.format_capabilities(format)
            .blit
            .supports(FilterMode::Linear)
    }

    /// Uploads `pixels` into mip 0 of `image`, fills every other mip by blitting
    /// the previous one, and leaves the image readable by fragment shaders.
    ///
    /// # Safety
    /// The image must be uninitialized, not in GPU use, created with copy
    /// source and destination usage, and remain alive through all subsequent
    /// uses. The recorded commands must be submitted in the current cycle.
    pub unsafe fn record(&mut self, rhi: &Rhi, image: &RhiImage, pixels: &[u8]) -> Result<()> {
        let encoder = self.encoder(rhi)?;
        let info = image.description();
        let barrier = |base_mip_level, mip_level_count, old_state, new_state| ImageBarrier {
            image,
            old_state,
            new_state,
            aspects: ImageAspects::COLOR,
            base_mip_level,
            mip_level_count,
            base_array_layer: 0,
            array_layer_count: 1,
            queue_transfer: None,
        };
        let last = info.mip_levels.saturating_sub(1);
        let shader_read = ImageState::ShaderRead(ShaderStages::FRAGMENT);
        let final_barriers = [
            barrier(0, last, ImageState::CopySource, shader_read),
            barrier(last, 1, ImageState::CopyDestination, shader_read),
        ];
        // SAFETY: the caller guarantees exclusive use of this fresh image, and
        // each barrier orders the previous copy or blit of the mips it covers.
        unsafe {
            encoder.barrier(&dependency(&[barrier(
                0,
                info.mip_levels,
                ImageState::Undefined,
                ImageState::CopyDestination,
            )]))?;
            ImageUpload {
                image,
                mip: 0,
                origin: Origin3d::default(),
                extent: info.mip_extent(0)?,
                pixels,
            }
            .record(rhi, encoder)?;

            for mip in 1..info.mip_levels {
                encoder.barrier(&dependency(&[barrier(
                    mip - 1,
                    1,
                    ImageState::CopyDestination,
                    ImageState::CopySource,
                )]))?;
                encoder.blit_image(
                    image,
                    image,
                    &[ImageBlit {
                        src_mip_level: mip - 1,
                        dst_mip_level: mip,
                        src_base_array_layer: 0,
                        dst_base_array_layer: 0,
                        array_layer_count: 1,
                        src_origin: Origin3d::default(),
                        dst_origin: Origin3d::default(),
                        src_extent: info.mip_extent(mip - 1)?,
                        dst_extent: info.mip_extent(mip)?,
                        aspects: ImageAspects::COLOR,
                    }],
                    FilterMode::Linear,
                )?;
            }

            // A single-mip image was never used as a blit source.
            let skip_sources = usize::from(last == 0);
            encoder.barrier(&dependency(&final_barriers[skip_sources..]))?;
        }
        Ok(())
    }

    /// Finishes the recording, if any texture was recorded.
    pub fn finish(self) -> Result<Option<RecordedCommands>> {
        Ok(self.encoder.map(CommandEncoder::finish).transpose()?)
    }

    fn encoder(&mut self, rhi: &Rhi) -> Result<&mut CommandEncoder> {
        let encoder = match self.encoder.take() {
            Some(encoder) => encoder,
            None => rhi.create_encoder("texture mip generation")?,
        };
        Ok(self.encoder.insert(encoder))
    }
}

fn dependency<'a>(image_barriers: &'a [ImageBarrier<'a>]) -> DependencyInfo<'a> {
    DependencyInfo {
        memory_barriers: &[],
        buffer_barriers: &[],
        image_barriers,
    }
}

/// CPU-prepared mip of an sRGB RGBA8 image, for formats the device cannot
/// blit with linear filtering.
pub(super) struct RgbaMip {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
impl RgbaMip {
    // Dimensions are constrained by image allocation limits; conversion back to
    // bytes follows clamping to the representable range.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn next(&self) -> Self {
        let decode = &*SRGB_TO_LINEAR;
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0_f32; 4];
                let mut count = 0;
                for sy in y * self.height / height..(y + 1) * self.height / height {
                    for sx in x * self.width / width..(x + 1) * self.width / width {
                        let offset = (sy as usize * self.width as usize + sx as usize) * 4;
                        let texel = &self.pixels[offset..offset + 4];
                        for (sum, &value) in sum[..3].iter_mut().zip(texel) {
                            *sum += decode[usize::from(value)];
                        }
                        sum[3] += f32::from(texel[3]) / 255.0;
                        count += 1;
                    }
                }
                let [r, g, b, a] = sum.map(|sum| sum / count as f32);
                pixels.extend([Self::srgb(r), Self::srgb(g), Self::srgb(b)]);
                pixels.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Encodes a linear value as the nearest sRGB byte.
    // The threshold table has 255 entries, so the partition point fits in a byte.
    #[allow(clippy::cast_possible_truncation)]
    fn srgb(value: f32) -> u8 {
        SRGB_THRESHOLDS.partition_point(|&threshold| threshold <= value) as u8
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear value of every sRGB byte.
static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut table = [0.0; 256];
    for (byte, value) in (0..=u8::MAX).zip(&mut table) {
        *value = srgb_to_linear(f32::from(byte) / 255.0);
    }
    table
});

/// Linear values halfway between consecutive sRGB bytes; a value encodes to
/// the number of thresholds at or below it.
static SRGB_THRESHOLDS: LazyLock<[f32; 255]> = LazyLock::new(|| {
    let mut table = [0.0; 255];
    for (byte, value) in (0..u8::MAX).zip(&mut table) {
        *value = srgb_to_linear((f32::from(byte) + 0.5) / 255.0);
    }
    table
});

#[cfg(test)]
mod tests {
    use super::{MipGeneration, RgbaMip, SRGB_TO_LINEAR};
    use dirk_rhi::{
        AccessTypes, BufferDesc, BufferImageCopy, BufferUsages, DependencyInfo, Extent3d, Graphics,
        Image, ImageAspects, ImageBarrier, ImageDesc, ImageDimension, ImageState, ImageUsages,
        MemoryBarrier, MemoryDomain, Origin3d, PipelineStages, RecordedCommands, Rhi,
        RhiCreateInfo, SampleCount, ShaderStages, SubmitInfo, TextureFormat,
    };

    #[test]
    fn odd_srgb_mips_include_edge_pixels_and_average_in_linear_light() {
        let mip = RgbaMip {
            width: 3,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255],
        }
        .next();
        assert_eq!((mip.width, mip.height), (1, 1));
        assert_eq!(mip.pixels, [156, 156, 156, 255]);
    }

    #[test]
    fn srgb_encoding_round_trips_every_byte() {
        for byte in 0..=u8::MAX {
            assert_eq!(RgbaMip::srgb(SRGB_TO_LINEAR[usize::from(byte)]), byte);
        }
    }

    #[test]
    #[ignore = "requires a native Vulkan or Metal device"]
    fn gpu_mips_average_in_linear_light() -> anyhow::Result<()> {
        let mut rhi = Rhi::new(&RhiCreateInfo {
            engine_name: "DirkEngine",
            engine_version: (0, 1, 0),
            application_name: "mip generation validation",
            application_version: (0, 1, 0),
            validation: true,
            compatible_surface: None,
        })?;
        if !MipGeneration::supports(&rhi, TextureFormat::Rgba8Srgb) {
            return Ok(());
        }
        let image = rhi.create_image(&ImageDesc {
            label: "mip generation test",
            dimension: ImageDimension::TwoD,
            extent: Extent3d::new_2d(4, 4),
            format: TextureFormat::Rgba8Srgb,
            usage: ImageUsages::COPY_SRC | ImageUsages::COPY_DST | ImageUsages::SAMPLED,
            mip_levels: 3,
            array_layers: 1,
            samples: SampleCount::One,
        })?;
        // Black and white columns average to linear 0.5, which encodes as 188.
        let pixels = (0..16)
            .flat_map(|texel| {
                let value = if texel % 2 == 0 { 0 } else { 255 };
                [value, value, value, 255]
            })
            .collect::<Vec<_>>();
        let mut mips = MipGeneration::new();
        // SAFETY: the image is fresh and the recording is submitted below.
        unsafe {
            mips.record(&rhi, &image, &pixels)?;
        }
        let Some(generated) = mips.finish()? else {
            anyhow::bail!("mip generation recorded no commands");
        };

        let texel = read_texel(&mut rhi, &image, 2, generated)?;
        assert!(
            texel[..3].iter().all(|value| value.abs_diff(188) <= 1),
            "{texel:?}"
        );
        assert_eq!(texel[3], 255);
        rhi.finish_cycle()?;
        rhi.wait_idle()?;
        assert_eq!(
            rhi.validation_error_count(),
            0,
            "native mip validation errors"
        );
        Ok(())
    }

    /// Reads the first texel of a fragment-readable `mip` after submitting `before`.
    fn read_texel(
        rhi: &mut Rhi,
        image: &Image,
        mip: u32,
        before: RecordedCommands,
    ) -> anyhow::Result<[u8; 4]> {
        let layout =
            dirk_rhi::UploadLayout::new(1, 1, TextureFormat::Rgba8Srgb, rhi.capabilities())?;
        let readback = rhi.create_buffer(&BufferDesc {
            label: "generated mip",
            size: u64::from(layout.bytes_per_row.get()),
            usage: BufferUsages::COPY_DST,
            memory: MemoryDomain::Readback,
        })?;
        let mut cmd = rhi.create_encoder::<Graphics>("generated mip readback")?;
        // SAFETY: `before` is submitted first in the same batch; CPU readback
        // happens only after submission completion.
        unsafe {
            cmd.barrier(&super::dependency(&[ImageBarrier {
                image,
                old_state: ImageState::ShaderRead(ShaderStages::FRAGMENT),
                new_state: ImageState::CopySource,
                aspects: ImageAspects::COLOR,
                base_mip_level: mip,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                queue_transfer: None,
            }]))?;
            cmd.copy_image_to_buffer(
                image,
                &readback,
                &[BufferImageCopy {
                    buffer_offset: 0,
                    buffer_bytes_per_row: layout.bytes_per_row,
                    buffer_rows_per_image: layout.rows,
                    mip_level: mip,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    image_origin: Origin3d::default(),
                    extent: Extent3d::new_2d(1, 1),
                    aspects: ImageAspects::COLOR,
                }],
            )?;
            cmd.barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_stages: PipelineStages::COPY,
                    src_access: AccessTypes::COPY_WRITE,
                    dst_stages: PipelineStages::HOST,
                    dst_access: AccessTypes::HOST_READ,
                }],
                buffer_barriers: &[],
                image_barriers: &[],
            })?;
            rhi.queue::<Graphics>()
                .submit(vec![before, cmd.finish()?], &SubmitInfo::default())?
                .wait(u64::MAX)?;
        }
        let mut texel = [0; 4];
        // SAFETY: GPU completion was awaited above.
        unsafe {
            readback.read(0, &mut texel)?;
        }
        Ok(texel)
    }
}
