//! Conversions between shared RHI types and their Vulkan equivalents.

use ash::vk;

use crate::{Format, SampleCounts};

impl Format {
    /// The equivalent Vulkan format.
    pub(super) const fn vk(self) -> vk::Format {
        match self {
            Self::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Self::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::Rg16Float => vk::Format::R16G16_SFLOAT,
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::R32Float => vk::Format::R32_SFLOAT,
            Self::Rg32Float => vk::Format::R32G32_SFLOAT,
            Self::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
            Self::Depth16Unorm => vk::Format::D16_UNORM,
            Self::Depth24UnormStencil8 => vk::Format::D24_UNORM_S8_UINT,
            Self::Depth32Float => vk::Format::D32_SFLOAT,
            Self::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
        }
    }
}

impl Format {
    /// Every aspect the format stores.
    pub(super) fn vk_aspects(self) -> vk::ImageAspectFlags {
        match (self.is_depth_stencil(), self.has_stencil()) {
            (false, _) => vk::ImageAspectFlags::COLOR,
            (true, false) => vk::ImageAspectFlags::DEPTH,
            (true, true) => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        }
    }
}

impl SampleCounts {
    /// The counts in `flags` the RHI exposes; higher counts are dropped.
    pub(super) fn from_vk(flags: vk::SampleCountFlags) -> Self {
        // Vulkan's 1, 2, 4 and 8 sample bits match ours.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bits above 8 samples are truncated away"
        )]
        let bits = flags.as_raw() as u8;
        Self::from_bits_truncate(bits)
    }
}
