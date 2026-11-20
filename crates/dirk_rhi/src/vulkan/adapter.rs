use std::ffi::{CStr, c_char};

use ash::vk;

use super::{Check, heap::Heap, instance::Instance};
use crate::{
    Capabilities, Error, Features, Format, FormatCapabilities, FormatFeatures, ImageUsage, Limits,
    Result, SampleCounts,
};

/// Declares the features every device must support. The RHI enables all of
/// them, so the whole design can rely on them.
macro_rules! required_features {
    ($($chain:ident: $ty:ident { $($feature:ident,)+ })+) => {
        /// Required features, chained into feature queries and device
        /// creation.
        #[derive(Default)]
        pub(super) struct RequiredFeatures {
            $(pub(super) $chain: vk::$ty<'static>,)+
        }

        impl RequiredFeatures {
            /// Every required feature, enabled.
            pub(super) fn enabled() -> Self {
                Self { $($chain: vk::$ty::default()$(.$feature(true))+,)+ }
            }

            /// Names of the required features `self` lacks.
            fn missing(&self) -> Vec<&'static str> {
                let mut missing = Vec::new();
                $($(if self.$chain.$feature != vk::TRUE {
                    missing.push(stringify!($feature));
                })+)+
                missing
            }
        }
    };
}

required_features! {
    vulkan12: PhysicalDeviceVulkan12Features {
        timeline_semaphore,
        vulkan_memory_model,
        descriptor_indexing,
        runtime_descriptor_array,
        descriptor_binding_partially_bound,
        descriptor_binding_update_unused_while_pending,
        descriptor_binding_sampled_image_update_after_bind,
        descriptor_binding_storage_image_update_after_bind,
        descriptor_binding_storage_buffer_update_after_bind,
        shader_sampled_image_array_non_uniform_indexing,
        shader_storage_image_array_non_uniform_indexing,
        shader_storage_buffer_array_non_uniform_indexing,
    }
    vulkan13: PhysicalDeviceVulkan13Features {
        dynamic_rendering,
        synchronization2,
    }
    robustness2: PhysicalDeviceRobustness2FeaturesEXT {
        null_descriptor,
    }
}

/// Queue families serving each kind of work.
#[derive(Clone, Copy, Debug)]
pub(super) struct QueueFamilies {
    /// Graphics and compute.
    pub(super) graphics: u32,
    /// Copies. Equal to `graphics` without a dedicated transfer family.
    pub(super) copy: u32,
}

/// A physical device that meets every requirement.
pub(super) struct Adapter {
    pub(super) raw: vk::PhysicalDevice,
    pub(super) families: QueueFamilies,
    /// Device extensions to enable.
    pub(super) extensions: Vec<*const c_char>,
    pub(super) capabilities: Capabilities,
    pub(super) non_coherent_atom_size: u64,
    score: u32,
}

impl Adapter {
    /// Picks the highest-scoring suitable device.
    pub(super) fn select(instance: &Instance) -> Result<Self> {
        // SAFETY: the instance is valid.
        let devices = unsafe { instance.raw.enumerate_physical_devices() }
            .check("vkEnumeratePhysicalDevices")?;
        let mut rejected = Vec::new();
        let mut best: Option<Self> = None;
        for raw in devices {
            match Self::inspect(instance, raw)? {
                Ok(adapter) => {
                    if best.as_ref().is_none_or(|best| adapter.score > best.score) {
                        best = Some(adapter);
                    }
                }
                Err(reason) => rejected.push(reason),
            }
        }
        best.ok_or_else(|| {
            Error::NoSuitableDevice(if rejected.is_empty() {
                "no Vulkan devices found".to_owned()
            } else {
                rejected.join("; ")
            })
        })
    }

    /// Checks `raw` against the requirements. The inner error names the
    /// device and every requirement it misses.
    fn inspect(instance: &Instance, raw: vk::PhysicalDevice) -> Result<Result<Self, String>> {
        let instance = &instance.raw;
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        // SAFETY: `raw` belongs to this instance.
        unsafe { instance.get_physical_device_properties2(raw, &mut properties) };
        let properties = properties.properties;
        let name = properties
            .device_name_as_c_str()
            .map_or_else(|_| "unnamed device".into(), CStr::to_string_lossy)
            .into_owned();

        let mut supported = RequiredFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported.vulkan12)
            .push_next(&mut supported.vulkan13)
            .push_next(&mut supported.robustness2);
        // SAFETY: `raw` belongs to this instance.
        unsafe { instance.get_physical_device_features2(raw, &mut features) };
        let features = features.features;
        // SAFETY: `raw` belongs to this instance.
        let available = unsafe { instance.enumerate_device_extension_properties(raw) }
            .check("vkEnumerateDeviceExtensionProperties")?;
        // SAFETY: `raw` belongs to this instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(raw) };

        let mut missing = Vec::new();
        if properties.api_version < vk::API_VERSION_1_3 {
            missing.push("Vulkan 1.3");
        }
        if features.robust_buffer_access != vk::TRUE {
            missing.push("robust_buffer_access");
        }
        let (extensions, missing_extensions) = Self::extensions(&available);
        missing.extend(missing_extensions);
        missing.extend(supported.missing());
        if !Heap::fits(&vulkan12) {
            missing.push("update-after-bind descriptor limits that fit the bindless heap");
        }
        let families = QueueFamilies::find(&families);
        if families.is_none() {
            missing.push("a graphics and compute queue");
        }
        let (Some(families), true) = (families, missing.is_empty()) else {
            return Ok(Err(format!("{name} lacks {}", missing.join(", "))));
        };

        let optional = [
            (features.sampler_anisotropy, Features::SAMPLER_ANISOTROPY),
            (features.image_cube_array, Features::IMAGE_CUBE_ARRAY),
            (features.independent_blend, Features::INDEPENDENT_BLEND),
            (features.depth_bias_clamp, Features::DEPTH_BIAS_CLAMP),
        ]
        .into_iter()
        .filter(|(supported, _)| *supported == vk::TRUE)
        .fold(Features::empty(), |all, (_, feature)| all | feature);
        let limits = &properties.limits;
        let capabilities = Capabilities {
            name,
            limits: Limits {
                max_buffer_size: vulkan13.max_buffer_size,
                max_image_dimension_2d: limits.max_image_dimension2_d,
                max_image_dimension_3d: limits.max_image_dimension3_d,
                max_image_dimension_cube: limits.max_image_dimension_cube,
                max_image_array_layers: limits.max_image_array_layers,
                max_sampler_anisotropy: if optional.contains(Features::SAMPLER_ANISOTROPY) {
                    // Saturating float-to-int cast; no device exceeds 16x.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "the value is clamped to 1..=16 first"
                    )]
                    let level = limits.max_sampler_anisotropy.clamp(1.0, 16.0) as u16;
                    level
                } else {
                    1
                },
                max_color_targets: limits.max_color_attachments,
                max_push_data_size: limits.max_push_constants_size,
                max_viewport_dimensions: limits.max_viewport_dimensions,
                viewport_bounds: limits.viewport_bounds_range,
                max_storage_buffer_size: u64::from(limits.max_storage_buffer_range),
            },
            features: optional,
            dedicated_copy_queue: families.copy != families.graphics,
            formats: std::array::from_fn(|index| {
                let format = Format::ALL[index];
                // SAFETY: `raw` belongs to this instance.
                let properties =
                    unsafe { instance.get_physical_device_format_properties(raw, format.vk()) };
                FormatCapabilities::new(format, properties.optimal_tiling_features, limits)
            }),
        };
        Ok(Ok(Self {
            raw,
            families,
            extensions,
            non_coherent_atom_size: limits.non_coherent_atom_size,
            capabilities,
            score: Self::score(properties.device_type),
        }))
    }
}

impl Adapter {
    /// How much a device of `device_type` is preferred.
    fn score(device_type: vk::PhysicalDeviceType) -> u32 {
        match device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 1,
            _ => 0,
        }
    }

    /// The device extensions to enable: robustness2 for null descriptors,
    /// and the portability subset whenever a device offers it. Also returns
    /// the required ones `available` lacks.
    fn extensions(
        available: &[vk::ExtensionProperties],
    ) -> (Vec<*const c_char>, Vec<&'static str>) {
        let has = |name: &CStr| {
            available
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(name))
        };
        let required = [Some(ash::ext::robustness2::NAME)];
        let missing = required
            .iter()
            .flatten()
            .filter(|name| !has(name))
            .map(|name| name.to_str().unwrap_or("a required extension"))
            .collect();
        let portability = Some(ash::khr::portability_subset::NAME).filter(|name| has(name));
        let enabled = required
            .into_iter()
            .chain([portability])
            .flatten()
            .map(CStr::as_ptr)
            .collect();
        (enabled, missing)
    }
}

impl QueueFamilies {
    /// Finds a graphics and compute family, and a transfer-only family for
    /// copies when the device has one.
    fn find(families: &[vk::QueueFamilyProperties]) -> Option<Self> {
        let find = |wanted: fn(vk::QueueFlags) -> bool| {
            families
                .iter()
                .position(|family| family.queue_count > 0 && wanted(family.queue_flags))
                .and_then(|index| u32::try_from(index).ok())
        };
        let graphics =
            find(|flags| flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))?;
        let copy = find(|flags| {
            flags.contains(vk::QueueFlags::TRANSFER)
                && !flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .unwrap_or(graphics);
        Some(Self { graphics, copy })
    }
}

impl FormatCapabilities {
    /// Support for `format` given its optimal-tiling features and the device
    /// limits that bound attachment sample counts.
    fn new(
        format: Format,
        features: vk::FormatFeatureFlags,
        limits: &vk::PhysicalDeviceLimits,
    ) -> Self {
        let usage = [
            (vk::FormatFeatureFlags::TRANSFER_SRC, ImageUsage::COPY_SRC),
            (vk::FormatFeatureFlags::TRANSFER_DST, ImageUsage::COPY_DST),
            (vk::FormatFeatureFlags::SAMPLED_IMAGE, ImageUsage::SAMPLED),
            (vk::FormatFeatureFlags::STORAGE_IMAGE, ImageUsage::STORAGE),
            (
                vk::FormatFeatureFlags::COLOR_ATTACHMENT,
                ImageUsage::COLOR_ATTACHMENT,
            ),
            (
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
                ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ),
        ]
        .into_iter()
        .filter(|(feature, _)| features.contains(*feature))
        .fold(ImageUsage::empty(), |all, (_, usage)| all | usage);
        let feature_flags = [
            (
                vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
                FormatFeatures::FILTER_LINEAR,
            ),
            (
                vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
                FormatFeatures::BLEND,
            ),
            (
                vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST,
                FormatFeatures::BLIT,
            ),
        ]
        .into_iter()
        .filter(|(feature, _)| features.contains(*feature))
        .fold(FormatFeatures::empty(), |all, (_, feature)| all | feature);
        let sample_counts = if usage.contains(ImageUsage::COLOR_ATTACHMENT) {
            limits.framebuffer_color_sample_counts
        } else if usage.contains(ImageUsage::DEPTH_STENCIL_ATTACHMENT) {
            if format.has_stencil() {
                limits.framebuffer_depth_sample_counts & limits.framebuffer_stencil_sample_counts
            } else {
                limits.framebuffer_depth_sample_counts
            }
        } else {
            vk::SampleCountFlags::TYPE_1
        };
        Self {
            usage,
            features: feature_flags,
            sample_counts: SampleCounts::from_vk(sample_counts),
        }
    }
}
