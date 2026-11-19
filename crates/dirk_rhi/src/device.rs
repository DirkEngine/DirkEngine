use std::sync::Arc;

use bitflags::bitflags;

use crate::{FRAMES_IN_FLIGHT, Format, FormatCapabilities, Result, frame::FrameSlot, native};

/// Application information and policy used to create an [`Rhi`].
#[derive(Clone, Copy, Debug)]
pub struct RhiCreateInfo<'a> {
    /// Application name reported to drivers and debugging tools.
    pub application_name: &'a str,
    /// Enables the native validation layer when it is installed. Errors it
    /// reports are logged and counted by [`Rhi::validation_error_count`].
    pub validation: bool,
    /// Bytes of host-visible memory each frame slot allocates from, with
    /// [`Frame::alloc`](crate::Frame::alloc); 8 MiB by default. Allocations
    /// that do not fit get buffers of their own, so this never grows. Zero
    /// gives every allocation a buffer of its own.
    pub frame_memory: u64,
}

impl Default for RhiCreateInfo<'_> {
    fn default() -> Self {
        Self {
            application_name: "",
            validation: false,
            frame_memory: 8 << 20,
        }
    }
}

/// The render hardware interface: one device and everything created from it.
///
/// Dropping it waits for the GPU to finish all submitted work.
pub struct Rhi {
    pub(crate) device: Arc<native::Device>,
    /// The number of the last frame begun; zero before the first.
    pub(crate) frame: u64,
    pub(crate) slots: [FrameSlot; FRAMES_IN_FLIGHT],
}

impl Rhi {
    /// Selects the most capable device that meets the RHI's requirements and
    /// creates it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableDevice`](crate::Error::NoSuitableDevice)
    /// listing why each device was rejected, [`Error::Invalid`](crate::Error::Invalid) if the frame
    /// memory exceeds the device's buffer size limit, or a native error if
    /// the graphics API cannot be loaded or initialized or memory cannot be
    /// allocated.
    pub fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        let mut rhi = Self {
            device: Arc::new(native::Device::new(info)?),
            frame: 0,
            slots: Default::default(),
        };
        if info.frame_memory > 0 {
            for slot in 0..FRAMES_IN_FLIGHT {
                rhi.slots[slot].memory = Some(rhi.create_frame_memory(info.frame_memory)?);
            }
        }
        Ok(rhi)
    }

    /// What the selected device supports. Queried once at creation.
    #[must_use]
    pub fn capabilities(&self) -> &Capabilities {
        &self.device.capabilities
    }

    /// Number of errors the validation layer has reported since creation.
    /// Always zero when validation is disabled or unavailable.
    #[must_use]
    pub fn validation_error_count(&self) -> usize {
        self.device.validation_error_count()
    }
}

/// What the selected device supports. Queried once when the [`Rhi`] is
/// created; requests beyond these are rejected at object creation.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Device name reported by the driver.
    pub name: String,
    /// Numeric limits.
    pub limits: Limits,
    /// Optional features the device supports and the RHI enabled.
    pub features: Features,
    /// Whether copies run on a queue separate from graphics work.
    pub dedicated_copy_queue: bool,
    pub(crate) formats: [FormatCapabilities; Format::COUNT],
}

impl Capabilities {
    /// Support for `format`.
    #[must_use]
    pub fn format(&self, format: Format) -> FormatCapabilities {
        self.formats[format as usize]
    }
}

/// Numeric device limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest buffer, in bytes.
    pub max_buffer_size: u64,
    /// Largest width or height of a 2D image.
    pub max_image_dimension_2d: u32,
    /// Largest width, height or depth of a 3D image.
    pub max_image_dimension_3d: u32,
    /// Largest width or height of a cube image.
    pub max_image_dimension_cube: u32,
    /// Most array layers in an image.
    pub max_image_array_layers: u32,
    /// Highest sampler anisotropy, as a whole level. One when anisotropic
    /// filtering is unsupported.
    pub max_sampler_anisotropy: u16,
    /// Most color targets a pipeline or render pass may have.
    pub max_color_targets: u32,
    /// Largest push data, in bytes.
    pub max_push_data_size: u32,
    /// Largest buffer with [`BufferUsage::STORAGE`](crate::BufferUsage::STORAGE),
    /// in bytes.
    pub max_storage_buffer_size: u64,
}

bitflags! {
    /// Optional device features. The RHI enables every one the device
    /// supports.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Features: u32 {
        /// Anisotropic sampler filtering.
        const SAMPLER_ANISOTROPY = 1 << 0;
        /// Views of arrays of cube images.
        const IMAGE_CUBE_ARRAY = 1 << 1;
        /// Different blend states per color attachment.
        const INDEPENDENT_BLEND = 1 << 2;
        /// Clamped depth bias.
        const DEPTH_BIAS_CLAMP = 1 << 3;
    }
}
