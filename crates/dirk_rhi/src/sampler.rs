use crate::{CompareOp, Error, Result, Rhi, SamplerIndex, native};

/// How texels are filtered between samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    /// The nearest texel.
    Nearest,
    /// Linear interpolation of neighbouring texels.
    #[default]
    Linear,
}

/// How coordinates outside an image are resolved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Clamp to the edge texel.
    #[default]
    ClampToEdge,
    /// Repeat the image.
    Repeat,
    /// Repeat the image, mirroring every other repetition.
    MirrorRepeat,
}

/// Describes a sampler to create.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc<'a> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// Filter when magnifying.
    pub mag_filter: Filter,
    /// Filter when minifying.
    pub min_filter: Filter,
    /// Filter between mip levels.
    pub mipmap_filter: Filter,
    /// Address modes along the U, V and W axes.
    pub address_modes: [AddressMode; 3],
    /// Maximum anisotropy; one disables anisotropic filtering.
    pub max_anisotropy: u16,
    /// Lowest mip level sampled.
    pub min_lod: f32,
    /// Highest mip level sampled.
    pub max_lod: f32,
    /// Makes a comparison sampler: depth comparisons with it return how
    /// many of the filtered texels pass `reference op texel`, from zero to
    /// one, as shadow maps are filtered. Shaders use comparison samplers
    /// only to compare depth images, and other samplers never to compare.
    pub compare: Option<CompareOp>,
}

impl Default for SamplerDesc<'_> {
    /// Trilinear filtering, clamped to the edge, over every mip level.
    fn default() -> Self {
        Self {
            label: "",
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_filter: Filter::Linear,
            address_modes: [AddressMode::ClampToEdge; 3],
            max_anisotropy: 1,
            min_lod: 0.0,
            max_lod: f32::MAX,
            compare: None,
        }
    }
}

/// Immutable properties of a [`Sampler`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerInfo {
    /// Filter when magnifying.
    pub mag_filter: Filter,
    /// Filter when minifying.
    pub min_filter: Filter,
    /// Filter between mip levels.
    pub mipmap_filter: Filter,
    /// Address modes along the U, V and W axes.
    pub address_modes: [AddressMode; 3],
    /// Maximum anisotropy.
    pub max_anisotropy: u16,
    /// Lowest mip level sampled.
    pub min_lod: f32,
    /// Highest mip level sampled.
    pub max_lod: f32,
    /// The comparison of a comparison sampler.
    pub compare: Option<CompareOp>,
}

/// How shaders sample images. Destroyed when dropped.
pub struct Sampler {
    raw: native::Sampler,
    info: SamplerInfo,
    label: Box<str>,
}

impl Rhi {
    /// Creates a sampler.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the anisotropy is zero or above the
    /// device limit, or the level-of-detail range is not finite, negative or
    /// reversed.
    pub fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<Sampler> {
        let invalid = |reason| Err(Error::invalid("sampler", desc.label, reason));
        let limit = self.capabilities().limits.max_sampler_anisotropy;
        if !(1..=limit).contains(&desc.max_anisotropy) {
            return invalid(format!(
                "anisotropy {} is outside the device's 1..={limit}",
                desc.max_anisotropy
            ));
        }
        if !(desc.min_lod.is_finite()
            && desc.max_lod.is_finite()
            && 0.0 <= desc.min_lod
            && desc.min_lod <= desc.max_lod)
        {
            return invalid(format!(
                "level of detail range {}..={} must be finite, non-negative and ordered",
                desc.min_lod, desc.max_lod
            ));
        }
        Ok(Sampler {
            raw: native::Sampler::new(&self.device, desc)?,
            info: SamplerInfo {
                mag_filter: desc.mag_filter,
                min_filter: desc.min_filter,
                mipmap_filter: desc.mipmap_filter,
                address_modes: desc.address_modes,
                max_anisotropy: desc.max_anisotropy,
                min_lod: desc.min_lod,
                max_lod: desc.max_lod,
                compare: desc.compare,
            },
            label: desc.label.into(),
        })
    }
}

impl Sampler {
    /// The sampler's immutable properties.
    #[must_use]
    pub fn info(&self) -> &SamplerInfo {
        &self.info
    }

    /// The sampler's index in the heap's sampler array.
    #[must_use]
    pub fn index(&self) -> SamplerIndex {
        SamplerIndex::new(self.raw.slot())
    }

    /// The sampler's debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}
