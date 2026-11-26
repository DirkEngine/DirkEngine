use dirk_shaders::bindless::heap;

use crate::{
    Capabilities, DepthTextureIndex, Error, Features, Format, ImageAccess, ImageUsage, Limits,
    Result, Rhi, SampleCounts, StorageImageIndex, TextureIndex, command::Transition, heap::Slot,
    native,
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

    /// The extent of mip `level`: halved per level, at least one texel.
    #[must_use]
    pub fn mip(self, level: u32) -> Self {
        let shrink = |size: u32| size.checked_shr(level).unwrap_or(0).max(1);
        Self {
            width: shrink(self.width),
            height: shrink(self.height),
            depth: shrink(self.depth),
        }
    }
}

/// Position of a texel in an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Origin3d {
    /// Horizontal position.
    pub x: u32,
    /// Vertical position.
    pub y: u32,
    /// Depth position.
    pub z: u32,
}

/// A box of texels in some layers of one mip level, copied or blitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageRegion {
    /// The mip level.
    pub mip_level: u32,
    /// First array layer.
    pub base_layer: u32,
    /// Number of array layers.
    pub layer_count: u32,
    /// Corner of the box.
    pub origin: Origin3d,
    /// Size of the box.
    pub extent: Extent3d,
}

/// Mip levels and array layers of an image a barrier applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageRange {
    /// First mip level.
    pub base_mip_level: u32,
    /// Number of mip levels; [`ImageRange::REMAINING`] for the rest.
    pub mip_level_count: u32,
    /// First array layer.
    pub base_layer: u32,
    /// Number of array layers; [`ImageRange::REMAINING`] for the rest.
    pub layer_count: u32,
}

impl ImageRange {
    /// A count covering every remaining mip level or layer.
    pub const REMAINING: u32 = u32::MAX;
    /// Every mip level and layer.
    pub const ALL: Self = Self {
        base_mip_level: 0,
        mip_level_count: Self::REMAINING,
        base_layer: 0,
        layer_count: Self::REMAINING,
    };

    /// Every layer of mip `level`.
    #[must_use]
    pub const fn mip(level: u32) -> Self {
        Self {
            base_mip_level: level,
            mip_level_count: 1,
            ..Self::ALL
        }
    }

    /// The resolved ranges covering the subresources of `regions`, each in
    /// one mip, merging those that share or neighbour layers.
    pub(crate) fn covering(regions: impl IntoIterator<Item = ImageRegion>) -> Vec<Self> {
        let mut ranges: Vec<Self> = Vec::new();
        for region in regions {
            let (mut start, mut end) = (region.base_layer, region.base_layer + region.layer_count);
            // Absorbs the ranges the region touches, then adds their union.
            ranges.retain(|range| {
                let (other_start, other_end) =
                    (range.base_layer, range.base_layer + range.layer_count);
                let touches = range.base_mip_level == region.mip_level
                    && other_start <= end
                    && start <= other_end;
                if touches {
                    (start, end) = (start.min(other_start), end.max(other_end));
                }
                !touches
            });
            ranges.push(Self {
                base_mip_level: region.mip_level,
                mip_level_count: 1,
                base_layer: start,
                layer_count: end - start,
            });
        }
        ranges
    }

    /// Each mip level and layer of this resolved range.
    pub(crate) fn subresources(self) -> impl Iterator<Item = (u32, u32)> {
        let mips = self.base_mip_level..self.base_mip_level + self.mip_level_count;
        let layers = self.base_layer..self.base_layer + self.layer_count;
        mips.flat_map(move |mip| layers.clone().map(move |layer| (mip, layer)))
    }

    /// Whether this resolved range holds mip `mip` of layer `layer`.
    pub(crate) fn contains(self, mip: u32, layer: u32) -> bool {
        (self.base_mip_level..self.base_mip_level + self.mip_level_count).contains(&mip)
            && (self.base_layer..self.base_layer + self.layer_count).contains(&layer)
    }

    /// Every mip of layer `layer`.
    #[must_use]
    pub const fn layer(layer: u32) -> Self {
        Self {
            base_layer: layer,
            layer_count: 1,
            ..Self::ALL
        }
    }
}

/// One mip level of one array layer of an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Subresource {
    /// The mip level.
    pub mip_level: u32,
    /// The array layer; a cube's faces are six consecutive layers.
    pub layer: u32,
}

impl From<Subresource> for ImageRange {
    fn from(subresource: Subresource) -> Self {
        Self {
            base_mip_level: subresource.mip_level,
            mip_level_count: 1,
            base_layer: subresource.layer,
            layer_count: 1,
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
    /// Samples per texel. Multisampled images are single-mip, single-layer
    /// 2D attachments, never sampled or storage.
    pub samples: SampleCount,
    /// The access the image rests in between submissions, which its usage
    /// allows; `None` for [`ImageAccess::resting`] of its usage.
    pub rest: Option<ImageAccess>,
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
    /// The access the image rests in between submissions. Safe operations
    /// move the image from it and back, the first submission after its
    /// creation moves it there from undefined contents, and submitting an
    /// encoder whose raw barriers left it elsewhere returns it there, unless
    /// they released it to another queue.
    pub rest: ImageAccess,
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

/// Usages that render to or store an image, which give 2D and cube images of
/// several mips or layers a view of each.
const SUBVIEW_USAGE: ImageUsage = ImageUsage::STORAGE
    .union(ImageUsage::COLOR_ATTACHMENT)
    .union(ImageUsage::DEPTH_STENCIL_ATTACHMENT);

/// One of an image's views, holding one subresource alone.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ViewOf {
    /// The view of every mip and layer of a single-mip, single-layer image.
    Full,
    /// The subview at this index.
    Subview(usize),
}

/// A GPU image. Destroyed, with its views, when dropped.
///
/// Its [view](Image::view) covers every mip and layer. A 2D or cube image of
/// several mips or layers that is rendered to or stored, by attachment or
/// storage usage, also owns a single-mip, single-layer 2D view of each mip
/// and layer, its [subviews](Image::subview), so passes can render to a cube
/// face, a shadow cascade or a mip, or sample one mip while rendering the
/// next.
///
/// Between submissions, an image is in its resting access,
/// [`ImageInfo::rest`], so safe commands need no state from their callers.
/// Its contents start undefined.
pub struct Image {
    pub(crate) raw: native::Image,
    info: ImageInfo,
    view: Option<ImageView>,
    /// Layer by layer, mip by mip.
    subviews: Box<[ImageView]>,
    label: Box<str>,
}

/// A view of an [`Image`], used by shaders and as an attachment: of every mip
/// and layer, or of one of each. The image owns its views, so a view never
/// outlives the memory it reads.
pub struct ImageView {
    view_type: ViewType,
    texture: Option<TextureIndex>,
    depth_texture: Option<DepthTextureIndex>,
    storage: Option<StorageImageIndex>,
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
        let (view_type, rest) = desc.check(self.capabilities())?;
        let raw = native::Image::new(&self.device, desc, view_type)?;
        let view = |view_type, slots: [Option<Slot>; 2]| {
            let [sampled, storage] = slots.map(|slot| slot.map(|slot| slot.index));
            let depth = desc.format.is_depth_stencil();
            ImageView {
                view_type,
                texture: sampled.filter(|_| !depth).map(TextureIndex::new),
                depth_texture: sampled.filter(|_| depth).map(DepthTextureIndex::new),
                storage: storage.map(StorageImageIndex::new),
            }
        };
        let image = Image {
            view: view_type.map(|view_type| view(view_type, raw.slots())),
            subviews: raw
                .subview_slots()
                .map(|slots| view(ViewType::D2, slots))
                .collect(),
            raw,
            info: ImageInfo {
                dimension: desc.dimension,
                extent: desc.extent,
                format: desc.format,
                usage: desc.usage,
                mip_levels: desc.mip_levels,
                array_layers: desc.array_layers,
                samples: desc.samples,
                rest,
            },
            label: desc.label.into(),
        };
        self.device
            .initialize(image.transition(ImageRange::ALL, ImageAccess::Undefined, rest));
        Ok(image)
    }
}

impl Image {
    /// Wraps a native image created elsewhere, such as a swapchain's, with
    /// a view of `view_type` outside the heap.
    pub(crate) fn from_native(
        raw: native::Image,
        info: ImageInfo,
        view_type: ViewType,
        label: &str,
    ) -> Self {
        Self {
            raw,
            info,
            view: Some(ImageView {
                view_type,
                texture: None,
                depth_texture: None,
                storage: None,
            }),
            subviews: Box::default(),
            label: label.into(),
        }
    }
}

impl ImageDesc<'_> {
    /// Validates the description against `capabilities` and returns the type
    /// of the view it needs, if any, and its resting access.
    fn check(&self, capabilities: &Capabilities) -> Result<(Option<ViewType>, ImageAccess)> {
        let view_type = self.check_shape(&capabilities.limits)?;
        self.check_usage(capabilities)?;
        let rest = self.check_rest()?;
        if !self.usage.intersects(VIEW_USAGE) {
            return Ok((None, rest));
        }
        if view_type == ViewType::CubeArray
            && !capabilities.features.contains(Features::IMAGE_CUBE_ARRAY)
        {
            return Err(self.invalid("cube array views are not supported on this device"));
        }
        Ok((Some(view_type), rest))
    }

    /// The resting access, checking that one chosen is an access the usage
    /// allows between submissions.
    fn check_rest(&self) -> Result<ImageAccess> {
        let Some(rest) = self.rest else {
            return Ok(ImageAccess::resting(self.usage));
        };
        if matches!(rest, ImageAccess::Undefined | ImageAccess::Present)
            || rest.shader_stages().is_some_and(|stages| stages.is_empty())
            || !self.usage.contains(rest.usage())
        {
            return Err(self.invalid(format!(
                "cannot rest in {rest:?}, which is undefined or presentation's, names no \
                 shader stage or needs a usage beyond {:?}",
                self.usage
            )));
        }
        Ok(rest)
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
            ImageDimension::D3 if self.format.is_depth_stencil() => {
                return Err(self.invalid("depth images are 2D or cubes, not 3D"));
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
                || self.array_layers != 1
                || self
                    .usage
                    .intersects(ImageUsage::SAMPLED | ImageUsage::STORAGE))
        {
            return Err(self.invalid(
                "multisampled images are single-mip, single-layer 2D attachments, never sampled \
                 or storage",
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

    /// The usage giving each subview its heap slots: storage, for per-mip
    /// writes, and sampled for color images, to sample one mip while
    /// rendering the next. Depth subviews get no sampled slot: shaders
    /// sample depth images, such as shadow maps, through their full view.
    pub(crate) fn subview_usage(&self) -> ImageUsage {
        if self.format.is_depth_stencil() {
            self.usage.difference(ImageUsage::SAMPLED)
        } else {
            self.usage
        }
    }

    /// The range of each subview the image owns, layer by layer, mip by
    /// mip; none unless it is a 2D or cube image of several mips or layers
    /// with attachment or storage usage.
    pub(crate) fn subview_ranges(&self) -> impl Iterator<Item = ImageRange> + use<> {
        let (mips, layers) = (self.mip_levels, self.array_layers);
        let subviews = self.dimension != ImageDimension::D3
            && self.usage.intersects(SUBVIEW_USAGE)
            && (mips > 1 || layers > 1);
        (0..if subviews { layers } else { 0 }).flat_map(move |layer| {
            (0..mips).map(move |mip_level| ImageRange::from(Subresource { mip_level, layer }))
        })
    }
}

impl ImageInfo {
    /// Every texel of mip `level`, in every layer.
    #[must_use]
    pub fn mip_region(&self, level: u32) -> ImageRegion {
        ImageRegion {
            mip_level: level,
            base_layer: 0,
            layer_count: self.array_layers,
            origin: Origin3d::default(),
            extent: self.extent.mip(level),
        }
    }

    /// Why `region` is not inside this image, if it is not.
    pub(crate) fn check_region(&self, region: &ImageRegion) -> Result<(), String> {
        let ImageRegion {
            mip_level,
            base_layer,
            layer_count,
            origin,
            extent,
        } = *region;
        if mip_level >= self.mip_levels {
            return Err(format!(
                "mip level {mip_level} is outside its {} levels",
                self.mip_levels
            ));
        }
        if layer_count == 0
            || base_layer
                .checked_add(layer_count)
                .is_none_or(|end| end > self.array_layers)
        {
            return Err(format!(
                "{layer_count} layers from layer {base_layer} are outside its {} layers",
                self.array_layers
            ));
        }
        let size = self.extent.mip(mip_level);
        let fits = |origin: u32, length: u32, size: u32| {
            length > 0 && origin.checked_add(length).is_some_and(|end| end <= size)
        };
        if !(fits(origin.x, extent.width, size.width)
            && fits(origin.y, extent.height, size.height)
            && fits(origin.z, extent.depth, size.depth))
        {
            return Err(format!(
                "box {extent:?} at {origin:?} is empty or outside mip {mip_level}, which is {size:?}"
            ));
        }
        Ok(())
    }

    /// `range` with remaining counts resolved, or why it is outside this
    /// image.
    pub(crate) fn resolve(&self, range: ImageRange) -> Result<ImageRange, String> {
        let resolve = |base: u32, count: u32, total: u32, what: &str| {
            let count = if count == ImageRange::REMAINING {
                total.saturating_sub(base)
            } else {
                count
            };
            if count == 0 || base.checked_add(count).is_none_or(|end| end > total) {
                Err(format!(
                    "{count} {what} from {base} are outside its {total}"
                ))
            } else {
                Ok(count)
            }
        };
        Ok(ImageRange {
            mip_level_count: resolve(
                range.base_mip_level,
                range.mip_level_count,
                self.mip_levels,
                "mip levels",
            )?,
            layer_count: resolve(
                range.base_layer,
                range.layer_count,
                self.array_layers,
                "layers",
            )?,
            ..range
        })
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

    /// A barrier moving `range` of the image from `before` to `after`.
    pub(crate) fn transition(
        &self,
        range: ImageRange,
        before: ImageAccess,
        after: ImageAccess,
    ) -> Transition {
        Transition {
            image: self.raw.id(),
            format: self.info.format,
            range,
            before,
            after,
            rest: self.info.rest,
            storage: self.info.usage.contains(ImageUsage::STORAGE),
            transfer: None,
        }
    }

    /// The image's view of every mip and layer, or `None` if its usage
    /// needs no view.
    #[must_use]
    pub fn view(&self) -> Option<&ImageView> {
        self.view.as_ref()
    }

    /// The image's single-mip, single-layer 2D view of `subresource`: its
    /// full view if it has one mip and layer, or the subview it owns,
    /// rendered to or stored per mip or layer. `None` if it has neither.
    #[must_use]
    pub fn subview(&self, subresource: Subresource) -> Option<&ImageView> {
        match self.view_of(subresource)? {
            ViewOf::Full => self.view(),
            ViewOf::Subview(index) => self.subviews.get(index),
        }
    }

    /// Which of the image's views holds `subresource` alone, if one does.
    pub(crate) fn view_of(&self, subresource: Subresource) -> Option<ViewOf> {
        let info = &self.info;
        let Subresource { mip_level, layer } = subresource;
        if mip_level >= info.mip_levels || layer >= info.array_layers {
            return None;
        }
        if self.subviews.is_empty() {
            (info.dimension == ImageDimension::D2 && info.mip_levels == 1 && info.array_layers == 1)
                .then_some(ViewOf::Full)
        } else {
            Some(ViewOf::Subview(
                (layer * info.mip_levels + mip_level) as usize,
            ))
        }
    }
}

impl ViewType {
    /// The heap arrays images of `format` viewed this way are sampled and
    /// stored through, if any, with the usage each needs. Depth images are
    /// sampled through the depth texture arrays and never stored.
    pub(crate) const fn heap_bindings(self, format: Format) -> [(ImageUsage, Option<u32>); 2] {
        let (sampled, storage) = match (self, format.is_depth_stencil()) {
            (Self::D2, false) => (Some(heap::TEXTURES_2D), Some(heap::STORAGE_IMAGES_2D)),
            (Self::D2Array, false) => (
                Some(heap::TEXTURES_2D_ARRAY),
                Some(heap::STORAGE_IMAGES_2D_ARRAY),
            ),
            (Self::Cube, false) => (Some(heap::TEXTURES_CUBE), None),
            (Self::CubeArray, false) => (Some(heap::TEXTURES_CUBE_ARRAY), None),
            (Self::D3, false) => (Some(heap::TEXTURES_3D), Some(heap::STORAGE_IMAGES_3D)),
            (Self::D2, true) => (Some(heap::DEPTH_TEXTURES_2D), None),
            (Self::D2Array, true) => (Some(heap::DEPTH_TEXTURES_2D_ARRAY), None),
            (Self::Cube, true) => (Some(heap::DEPTH_TEXTURES_CUBE), None),
            (Self::CubeArray, true) => (Some(heap::DEPTH_TEXTURES_CUBE_ARRAY), None),
            // Creation rejects 3D depth images.
            (Self::D3, true) => (None, None),
        };
        [
            (ImageUsage::SAMPLED, sampled),
            (ImageUsage::STORAGE, storage),
        ]
    }
}

impl ImageView {
    /// How the view presents its image.
    #[must_use]
    pub fn view_type(&self) -> ViewType {
        self.view_type
    }

    /// The view's index in the heap's texture array of its view type, if
    /// the image has [`ImageUsage::SAMPLED`] and a color format.
    #[must_use]
    pub fn index(&self) -> Option<TextureIndex> {
        self.texture
    }

    /// The view's index in the heap's depth texture array of its view type,
    /// if the image has [`ImageUsage::SAMPLED`] and a depth format. Every
    /// sampled depth format can be compared with a comparison sampler.
    #[must_use]
    pub fn depth_index(&self) -> Option<DepthTextureIndex> {
        self.depth_texture
    }

    /// The view's index in the heap's storage image array of its view type,
    /// if the image has [`ImageUsage::STORAGE`] and is not a cube. Storage
    /// image arrays declare no format, so shaders accessing them need the
    /// `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat`
    /// capabilities.
    #[must_use]
    pub fn storage_index(&self) -> Option<StorageImageIndex> {
        self.storage
    }
}
