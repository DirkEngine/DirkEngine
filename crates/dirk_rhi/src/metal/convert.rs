//! Conversions between shared RHI types and their Metal equivalents.

use objc2_metal::{MTLDevice, MTLPixelFormat, MTLReadWriteTextureTier};

use super::Object;
use crate::{Format, FormatCapabilities, FormatFeatures, ImageUsage, SampleCounts};

impl Format {
    /// The equivalent Metal pixel format.
    pub(super) const fn mtl(self) -> MTLPixelFormat {
        match self {
            Self::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
            Self::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
            Self::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
            Self::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
            Self::R16Float => MTLPixelFormat::R16Float,
            Self::Rg16Float => MTLPixelFormat::RG16Float,
            Self::Rgba16Float => MTLPixelFormat::RGBA16Float,
            Self::R32Float => MTLPixelFormat::R32Float,
            Self::Rg32Float => MTLPixelFormat::RG32Float,
            Self::Rgba32Float => MTLPixelFormat::RGBA32Float,
            Self::Rg11b10Float => MTLPixelFormat::RG11B10Float,
            Self::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
            Self::Depth24UnormStencil8 => MTLPixelFormat::Depth24Unorm_Stencil8,
            Self::Depth32Float => MTLPixelFormat::Depth32Float,
            Self::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
        }
    }
}

impl FormatCapabilities {
    /// Support for `format` on `device`. Metal cannot be queried per format,
    /// so this follows Apple's feature set tables for Metal 3 GPUs,
    /// conservatively: 32-bit float targets neither blend nor multisample,
    /// depth is not filtered, and storage images, which shaders read and
    /// write, are limited to formats with read-write access.
    pub(super) fn metal(format: Format, device: &Object<dyn MTLDevice>) -> Self {
        let float32 = matches!(
            format,
            Format::R32Float | Format::Rg32Float | Format::Rgba32Float
        );
        let read_write = match device.readWriteTextureSupport() {
            MTLReadWriteTextureTier::Tier2 => matches!(
                format,
                Format::Rgba8Unorm
                    | Format::R16Float
                    | Format::Rgba16Float
                    | Format::R32Float
                    | Format::Rgba32Float
            ),
            MTLReadWriteTextureTier::Tier1 => format == Format::R32Float,
            _ => false,
        };
        let copies = ImageUsage::COPY_SRC | ImageUsage::COPY_DST;
        let (usage, features) = if format.is_depth_stencil() {
            let supported = format != Format::Depth24UnormStencil8
                || device.isDepth24Stencil8PixelFormatSupported();
            if !supported {
                return Self::default();
            }
            (
                copies | ImageUsage::SAMPLED | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                FormatFeatures::empty(),
            )
        } else {
            let mut usage = copies | ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT;
            if read_write {
                usage |= ImageUsage::STORAGE;
            }
            let mut features = FormatFeatures::BLIT;
            if !float32 || device.supports32BitFloatFiltering() {
                features |= FormatFeatures::FILTER_LINEAR;
            }
            if !float32 {
                features |= FormatFeatures::BLEND;
            }
            (usage, features)
        };
        let sample_counts = if float32 {
            SampleCounts::X1
        } else {
            [
                (SampleCounts::X2, 2),
                (SampleCounts::X4, 4),
                (SampleCounts::X8, 8),
            ]
            .into_iter()
            .filter(|&(_, count)| device.supportsTextureSampleCount(count))
            .fold(SampleCounts::X1, |all, (counts, _)| all | counts)
        };
        Self {
            usage,
            features,
            sample_counts,
        }
    }
}
