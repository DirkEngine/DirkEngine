use crate::{
    Capabilities, Error, Features, Format, ImageUsage, Limits, Result, Rhi, SampleCounts, native,
};

/// Width, height and depth of an image, in texels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Extent3d {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth in texels; one for 2D and cube images.
    pub depth: u32,
}

impl Extent3d {
    /// A 2D extent with a depth of one.
    #[must_use]
    pub const fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }
}

/// Shape of an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    /// A 2D image, or an array of them.
    #[default]
    D2,
    /// A 3D image. Has exactly one array layer.
    D3,
    /// A cube image, or an array of them. Square, with six layers per cube.
    Cube,
}

/// Samples per texel of a multisampled image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SampleCount {
    /// One sample per texel: not multisampled.
    #[default]
    X1,
    /// Two samples per texel.
    X2,
    /// Four samples per texel.
    X4,
    /// Eight samples per texel.
    X8,
}

impl SampleCount {
    /// This count as a flag in a set of counts.
    #[must_use]
    pub const fn flag(self) -> SampleCounts {
        match self {
            Self::X1 => SampleCounts::X1,
            Self::X2 => SampleCounts::X2,
            Self::X4 => SampleCounts::X4,
            Self::X8 => SampleCounts::X8,
        }
    }
}

/// Describes an image to create.
#[derive(Clone, Copy, Debug)]
pub struct ImageDesc<'a> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// Shape of the image.
    pub dimension: ImageDimension,
    /// Size of mip 0.
    pub extent: Extent3d,
    /// Texel format.
    pub format: Format,
    /// Ways the image may be used.
    pub usage: ImageUsage,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of array layers; a multiple of six for cube images.
    pub array_layers: u32,
    /// Samples per texel.
    pub samples: SampleCount,
}

/// Immutable properties of an [`Image`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    /// Shape of the image.
    pub dimension: ImageDimension,
    /// Size of mip 0.
    pub extent: Extent3d,
    /// Texel format.
    pub format: Format,
    /// Ways the image may be used.
    pub usage: ImageUsage,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Samples per texel.
    pub samples: SampleCount,
}

/// How a view presents its image to shaders and attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewType {
    /// A single 2D image.
    D2,
    /// An array of 2D images.
    D2Array,
    /// A 3D image.
    D3,
    /// A cube image.
    Cube,
    /// An array of cube images.
    CubeArray,
}

/// Usages that read or write an image through a view.
const VIEW_USAGE: ImageUsage = ImageUsage::SAMPLED
    .union(ImageUsage::STORAGE)
    .union(ImageUsage::COLOR_ATTACHMENT)
    .union(ImageUsage::DEPTH_STENCIL_ATTACHMENT);

/// A GPU image. Destroyed, with its view, when dropped.
pub struct Image {
    #[expect(dead_code, reason = "read when copies are recorded")]
    raw: native::Image,
    info: ImageInfo,
    view: Option<ImageView>,
    label: Box<str>,
}

/// A view of every mip and layer of an [`Image`], used by shaders and as an
/// attachment. The image owns its view, so a view never outlives the memory
/// it reads.
pub struct ImageView {
    #[expect(
        dead_code,
        reason = "read when render passes and the bindless heap use views"
    )]
    raw: native::ImageView,
    view_type: ViewType,
}

impl Rhi {
    /// Creates an image and, when its usage reads or writes it through a view
    /// (sampled, storage or attachment usage), its view. Transfer-only images
    /// have no view.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the description is inconsistent, exceeds
    /// a device limit or asks for something the format or device does not
    /// support, or a native error if memory cannot be allocated.
    pub fn create_image(&self, desc: &ImageDesc<'_>) -> Result<Image> {
        let view_type = desc.check(self.capabilities())?;
        let raw = native::Image::new(&self.device, desc, view_type)?;
        Ok(Image {
            view: view_type
                .zip(raw.view())
                .map(|(view_type, raw)| ImageView { raw, view_type }),
            raw,
            info: ImageInfo {
                dimension: desc.dimension,
                extent: desc.extent,
                format: desc.format,
                usage: desc.usage,
                mip_levels: desc.mip_levels,
                array_layers: desc.array_layers,
                samples: desc.samples,
            },
            label: desc.label.into(),
        })
    }
}

impl ImageDesc<'_> {
    /// Validates the description against `capabilities` and returns the type
    /// of the view it needs, if any.
    fn check(&self, capabilities: &Capabilities) -> Result<Option<ViewType>> {
        let view_type = self.check_shape(&capabilities.limits)?;
        self.check_usage(capabilities)?;
        if !self.usage.intersects(VIEW_USAGE) {
            return Ok(None);
        }
        if view_type == ViewType::CubeArray
            && !capabilities.features.contains(Features::IMAGE_CUBE_ARRAY)
        {
            return Err(self.invalid("cube array views are not supported on this device"));
        }
        Ok(Some(view_type))
    }

    /// Checks the extent, dimension, mips and layers against `limits` and
    /// returns the type of a view of the whole image.
    fn check_shape(&self, limits: &Limits) -> Result<ViewType> {
        let Extent3d {
            width,
            height,
            depth,
        } = self.extent;
        if width == 0 || height == 0 || depth == 0 {
            return Err(self.invalid(format!("extent {:?} is empty", self.extent)));
        }
        if self.mip_levels == 0 || self.array_layers == 0 {
            return Err(self.invalid("mip level and array layer counts must be at least one"));
        }
        let (limit, view_type) = match self.dimension {
            ImageDimension::D2 if depth != 1 => {
                return Err(self.invalid(format!("2D images have a depth of one, not {depth}")));
            }
            ImageDimension::D2 if self.array_layers == 1 => {
                (limits.max_image_dimension_2d, ViewType::D2)
            }
            ImageDimension::D2 => (limits.max_image_dimension_2d, ViewType::D2Array),
            ImageDimension::D3 if self.array_layers != 1 => {
                return Err(self.invalid("3D images have one array layer"));
            }
            ImageDimension::D3 => (limits.max_image_dimension_3d, ViewType::D3),
            ImageDimension::Cube
                if depth != 1 || width != height || !self.array_layers.is_multiple_of(6) =>
            {
                return Err(self.invalid(format!(
                    "cube images are square with a depth of one and six layers per cube, \
                     not {:?} with {} layers",
                    self.extent, self.array_layers
                )));
            }
            ImageDimension::Cube if self.array_layers == 6 => {
                (limits.max_image_dimension_cube, ViewType::Cube)
            }
            ImageDimension::Cube => (limits.max_image_dimension_cube, ViewType::CubeArray),
        };
        let largest = width.max(height).max(depth);
        if largest > limit {
            return Err(self.invalid(format!(
                "extent {:?} exceeds the device limit of {limit} texels",
                self.extent
            )));
        }
        if self.array_layers > limits.max_image_array_layers {
            return Err(self.invalid(format!(
                "{} array layers exceed the device limit of {}",
                self.array_layers, limits.max_image_array_layers
            )));
        }
        if self.mip_levels > largest.bit_width() {
            return Err(self.invalid(format!(
                "{} mip levels exceed the {} of a {largest}-texel image",
                self.mip_levels,
                largest.bit_width()
            )));
        }
        Ok(view_type)
    }

    /// Checks the usage and sample count against what the format supports.
    fn check_usage(&self, capabilities: &Capabilities) -> Result<()> {
        if self.usage.is_empty() {
            return Err(self.invalid("usage is empty"));
        }
        let format = capabilities.format(self.format);
        let unsupported = self.usage.difference(format.usage);
        if !unsupported.is_empty() {
            return Err(self.invalid(format!(
                "{:?} does not support {unsupported:?} on this device",
                self.format
            )));
        }
        if !format.sample_counts.contains(self.samples.flag()) {
            return Err(self.invalid(format!(
                "{:?} does not support {:?} samples on this device",
                self.format, self.samples
            )));
        }
        if self.samples != SampleCount::X1
            && (self.dimension != ImageDimension::D2
                || self.mip_levels != 1
                || self
                    .usage
                    .intersects(ImageUsage::SAMPLED | ImageUsage::STORAGE))
        {
            return Err(self.invalid(
                "multisampled images are single-mip 2D attachments, never sampled or storage",
            ));
        }
        if self.format.has_stencil() && self.usage.contains(ImageUsage::SAMPLED) {
            return Err(self.invalid(format!(
                "sampling {:?} is not supported; its view covers both depth and stencil",
                self.format
            )));
        }
        Ok(())
    }

    /// Rejects this description.
    fn invalid(&self, reason: impl Into<String>) -> Error {
        Error::invalid("image", self.label, reason)
    }
}

impl Image {
    /// The image's immutable properties.
    #[must_use]
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// The image's debug label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The image's view, or `None` if its usage needs no view.
    #[must_use]
    pub fn view(&self) -> Option<&ImageView> {
        self.view.as_ref()
    }
}

impl ImageView {
    /// How the view presents its image.
    #[must_use]
    pub fn view_type(&self) -> ViewType {
        self.view_type
    }
}
