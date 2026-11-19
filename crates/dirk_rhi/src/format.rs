use bitflags::bitflags;

/// Declares [`Format`] together with the list of every format, so capability
/// tables indexed by `format as usize` cannot fall out of sync.
macro_rules! formats {
    ($($(#[$doc:meta])* $name:ident,)+) => {
        /// Texel formats of images and views.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Format {
            $($(#[$doc])* $name,)+
        }

        impl Format {
            /// Every format, in declaration order.
            pub const ALL: &[Self] = &[$(Self::$name),+];
            /// Number of formats.
            pub const COUNT: usize = Self::ALL.len();
        }
    };
}

formats! {
    /// Four 8-bit normalized channels.
    Rgba8Unorm,
    /// Four 8-bit sRGB-encoded channels.
    Rgba8Srgb,
    /// Four 8-bit normalized channels in blue-green-red-alpha order.
    Bgra8Unorm,
    /// Four 8-bit sRGB-encoded channels in blue-green-red-alpha order.
    Bgra8Srgb,
    /// One 16-bit float channel.
    R16Float,
    /// Two 16-bit float channels.
    Rg16Float,
    /// Four 16-bit float channels.
    Rgba16Float,
    /// One 32-bit float channel.
    R32Float,
    /// Two 32-bit float channels.
    Rg32Float,
    /// Four 32-bit float channels.
    Rgba32Float,
    /// Packed unsigned 11-, 11- and 10-bit float channels.
    Rg11b10Float,
    /// 16-bit normalized depth.
    Depth16Unorm,
    /// 24-bit normalized depth with 8-bit stencil.
    Depth24UnormStencil8,
    /// 32-bit float depth.
    Depth32Float,
    /// 32-bit float depth with 8-bit stencil.
    Depth32FloatStencil8,
}

impl Format {
    /// Whether this format stores depth and/or stencil rather than color.
    #[must_use]
    pub const fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth24UnormStencil8
                | Self::Depth32Float
                | Self::Depth32FloatStencil8
        )
    }

    /// Whether this format stores colors sRGB-encoded, encoding them when
    /// written and decoding them when read, so shaders see linear colors.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bgra8Srgb)
    }

    /// Size of one texel in bytes.
    #[must_use]
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R16Float | Self::Depth16Unorm => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rg16Float
            | Self::R32Float
            | Self::Rg11b10Float
            | Self::Depth24UnormStencil8
            | Self::Depth32Float => 4,
            Self::Rgba16Float | Self::Rg32Float | Self::Depth32FloatStencil8 => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Alignment of buffer offsets in copies between buffers and images of
    /// this format, in bytes: a multiple of four and of the texel size.
    #[must_use]
    pub const fn copy_offset_alignment(self) -> u64 {
        let texel = self.bytes_per_texel() as u64;
        if texel > 4 { texel } else { 4 }
    }

    /// Whether this format has a stencil component.
    #[must_use]
    pub const fn has_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8
        )
    }
}

bitflags! {
    /// Ways an image may be used. Declared when the image is created.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        /// Source of copies and blits.
        const COPY_SRC = 1 << 0;
        /// Destination of copies and blits.
        const COPY_DST = 1 << 1;
        /// Sampled by shaders.
        const SAMPLED = 1 << 2;
        /// Read and written by shaders as a storage image.
        const STORAGE = 1 << 3;
        /// Rendered to as a color attachment.
        const COLOR_ATTACHMENT = 1 << 4;
        /// Rendered to as a depth/stencil attachment.
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
    }
}

bitflags! {
    /// Sample counts supported for multisampling.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct SampleCounts: u8 {
        /// One sample per texel.
        const X1 = 1 << 0;
        /// Two samples per texel.
        const X2 = 1 << 1;
        /// Four samples per texel.
        const X4 = 1 << 2;
        /// Eight samples per texel.
        const X8 = 1 << 3;
    }
}

bitflags! {
    /// Optional operations a format supports beyond its [`ImageUsage`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct FormatFeatures: u8 {
        /// Linear filtering when sampled or blitted.
        const FILTER_LINEAR = 1 << 0;
        /// Blending as a color attachment.
        const BLEND = 1 << 1;
        /// Source and destination of blits.
        const BLIT = 1 << 2;
    }
}

/// What the device supports for one [`Format`], with optimal tiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FormatCapabilities {
    /// Supported image usages.
    pub usage: ImageUsage,
    /// Supported optional operations.
    pub features: FormatFeatures,
    /// Sample counts supported when used as an attachment. Only
    /// [`SampleCounts::X1`] when the format cannot be an attachment.
    pub sample_counts: SampleCounts,
}
