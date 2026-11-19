use ash::{khr::surface, vk};

pub struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub vulkan12_features: vk::PhysicalDeviceVulkan12Features<'static>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub extensions: Vec<vk::ExtensionProperties>,
}

impl PhysicalDeviceInfo {
    pub fn query(instance: &ash::Instance, handle: vk::PhysicalDevice) -> Self {
        unsafe {
            let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut features2 =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan12_features);
            instance.get_physical_device_features2(handle, &mut features2);

            Self {
                handle,
                properties: instance.get_physical_device_properties(handle),
                features: instance.get_physical_device_features(handle),
                vulkan12_features,
                memory_properties: instance.get_physical_device_memory_properties(handle),
                queue_families: instance.get_physical_device_queue_family_properties(handle),
                extensions: instance
                    .enumerate_device_extension_properties(handle)
                    .unwrap_or_default(),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Queue family indices never approach u32::MAX."
    )]
    pub fn resolve(
        info: &PhysicalDeviceInfo,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Option<Self> {
        let families = &info.queue_families;

        let graphics = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))?
            as u32;

        let compute = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .unwrap_or(graphics as usize) as u32;

        let transfer = families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !f.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .unwrap_or(compute as usize) as u32;

        let present = families.iter().enumerate().position(|(i, _)| unsafe {
            surface_loader
                .get_physical_device_surface_support(info.handle, i as u32, surface)
                .unwrap_or(false)
        })? as u32;

        Some(Self {
            graphics,
            compute,
            transfer,
            present,
        })
    }
}

fn score_device(info: &PhysicalDeviceInfo) -> u32 {
    let mut score = 0u32;

    // Strongly prefer discrete GPUs
    score += match info.properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 10_000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1_000,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 500,
        _ => 0,
    };

    // Reward larger device-local VRAM
    let vram_mb = info
        .memory_properties
        .memory_heaps
        .iter()
        .take(info.memory_properties.memory_heap_count as usize)
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size / (1024 * 1024))
        .sum::<u64>();
    score += (vram_mb / 512).min(5_000) as u32; // cap contribution at ~2.5 GB equivalent

    // Reward higher Vulkan API version support
    score += vk::api_version_minor(info.properties.api_version) * 100;

    score
}

type Requirement = dyn Fn(&PhysicalDeviceInfo) -> bool;

/// A simple struct to help select a physical device for vulkan.
/// Add requirements that implement the [`DeviceRequirement`] trait.
/// When selecting, will make sure all requirements are met, or will
/// return [None].
pub struct PhysicalDeviceSelector {
    requirements: Vec<Box<Requirement>>,
}

impl PhysicalDeviceSelector {
    pub fn new() -> Self {
        Self {
            requirements: vec![],
        }
    }

    pub fn require_extensions(self, extensions: &'static [&'static str]) -> Self {
        self.require(move |info: &PhysicalDeviceInfo| {
            extensions.iter().all(|&extension| {
                info.extensions.iter().any(|e| {
                    unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }
                        .to_str()
                        .unwrap_or("")
                        == extension
                })
            })
        })
    }

    pub fn require<F: Fn(&PhysicalDeviceInfo) -> bool + 'static>(mut self, req: F) -> Self {
        self.requirements.push(Box::new(req));
        self
    }

    pub fn select(
        &self,
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Option<(PhysicalDeviceInfo, QueueFamilyIndices)> {
        let devices = unsafe { instance.enumerate_physical_devices().ok()? };
        devices
            .into_iter()
            .map(|handle| PhysicalDeviceInfo::query(instance, handle))
            .filter(|info| {
                self.requirements.iter().all(|req| req(info)) // ← just call it
            })
            .filter_map(|info| {
                QueueFamilyIndices::resolve(&info, surface_loader, surface).map(|q| (info, q))
            })
            .max_by_key(|(info, _)| score_device(info))
    }
}
