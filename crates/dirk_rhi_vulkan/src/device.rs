use std::{
    collections::HashSet,
    ffi::{CStr, CString, c_void},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use dirk_rhi::{Capabilities, Error, Format, QueueType, Result, RhiCreateInfo, SampleCount};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;
use tracing::{debug, error, info, warn};

use crate::{backend_error, convert::QueueKind, vk_error};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";
const RETIREMENT_DELAY: u64 = 3;

pub(crate) type Retained = Arc<dyn Send + Sync>;
type DebugMessenger = (debug_utils::Instance, vk::DebugUtilsMessengerEXT);

#[derive(Clone, Copy, Debug)]
pub(crate) struct QueueFamilies {
    pub(crate) graphics: u32,
    pub(crate) compute: u32,
    pub(crate) copy: u32,
    pub(crate) present: u32,
}

pub(crate) struct Queues {
    pub(crate) graphics: vk::Queue,
    pub(crate) compute: vk::Queue,
    pub(crate) copy: vk::Queue,
    pub(crate) present: vk::Queue,
}

impl Queues {
    pub(crate) fn get(&self, kind: QueueKind) -> vk::Queue {
        match kind {
            QueueKind::Graphics => self.graphics,
            QueueKind::Compute => self.compute,
            QueueKind::Copy => self.copy,
        }
    }
}

pub(crate) enum Garbage {
    Buffer {
        raw: vk::Buffer,
        allocation: Allocation,
    },
    Image {
        raw: vk::Image,
        allocation: Allocation,
    },
    ImageView(vk::ImageView),
    Sampler(vk::Sampler),
    Shader(vk::ShaderModule),
    BindGroupLayout(vk::DescriptorSetLayout),
    DescriptorPool(vk::DescriptorPool),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    CommandPool(vk::CommandPool),
    Fence(vk::Fence),
    Semaphore(vk::Semaphore),
    Surface(vk::SurfaceKHR),
    Swapchain {
        raw: vk::SwapchainKHR,
        views: Vec<vk::ImageView>,
        semaphores: Vec<vk::Semaphore>,
    },
}

struct PendingGarbage {
    retire_at: u64,
    value: Garbage,
}

pub(crate) struct Context {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) surface_loader: surface::Instance,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) families: QueueFamilies,
    pub(crate) queues: Queues,
    pub(crate) capabilities: Capabilities,
    pub(crate) sampler_anisotropy: bool,
    pub(crate) non_coherent_atom_size: u64,
    pub(crate) enabled_instance_extensions: HashSet<String>,
    allocator: Mutex<Option<Allocator>>,
    garbage: Mutex<Vec<PendingGarbage>>,
    retirement_epoch: AtomicU64,
    debug_messenger: Option<DebugMessenger>,
}

struct Bootstrap {
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    debug_messenger: Option<DebugMessenger>,
}

struct Initialized {
    instance: ash::Instance,
    device: ash::Device,
    debug_messenger: Option<DebugMessenger>,
}

impl Bootstrap {
    fn finish(mut self) -> Result<Initialized> {
        let instance = self
            .instance
            .take()
            .ok_or(Error::Backend("Vulkan instance bootstrap was empty".into()))?;
        let device = self
            .device
            .take()
            .ok_or(Error::Backend("Vulkan device bootstrap was empty".into()))?;
        Ok(Initialized {
            instance,
            device,
            debug_messenger: self.debug_messenger.take(),
        })
    }
}

impl Drop for Bootstrap {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = self.device.take() {
                device.destroy_device(None);
            }
            if let Some((loader, messenger)) = self.debug_messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            if let Some(instance) = self.instance.take() {
                instance.destroy_instance(None);
            }
        }
    }
}

impl Context {
    #[allow(
        clippy::too_many_lines,
        reason = "Vulkan instance and device setup is kept linear so borrowed create-info data remains auditable"
    )]
    pub(crate) fn new(info: &RhiCreateInfo<'_>) -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load() }.map_err(backend_error)?;
        let application_name = CString::new(info.application_name)
            .map_err(|_| Error::InvalidResource("application name contains a null byte"))?;
        let engine_name = CString::new(info.engine_name)
            .map_err(|_| Error::InvalidResource("engine name contains a null byte"))?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(version(info.application_version))
            .engine_name(&engine_name)
            .engine_version(version(info.engine_version))
            .api_version(vk::API_VERSION_1_3);

        let mut extensions = if let Some(surface) = info.compatible_surface {
            ash_window::enumerate_required_extensions(surface.display)
                .map_err(vk_error)?
                .to_vec()
        } else {
            Vec::new()
        };
        let available_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(vk_error)?;
        if extension_available(
            &available_extensions,
            ash::khr::portability_enumeration::NAME,
        ) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }
        if info.validation {
            extensions.push(debug_utils::NAME.as_ptr());
        }
        extensions.sort_unstable();
        extensions.dedup();
        let enabled_instance_extensions = extensions
            .iter()
            .map(|&extension| {
                unsafe { CStr::from_ptr(extension) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

        for &extension in &extensions {
            let name = unsafe { CStr::from_ptr(extension) };
            if !extension_available(&available_extensions, name) {
                return Err(Error::Unsupported("required Vulkan instance extension"));
            }
        }

        let mut layers = Vec::new();
        if info.validation {
            let available_layers =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(vk_error)?;
            if available_layers.iter().any(|layer| unsafe {
                CStr::from_ptr(layer.layer_name.as_ptr()) == VALIDATION_LAYER
            }) {
                layers.push(VALIDATION_LAYER.as_ptr());
            } else {
                warn!(
                    "Vulkan validation was requested, but the Khronos validation layer is unavailable"
                );
            }
        }

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        if extensions
            .iter()
            .any(|&extension| extension == ash::khr::portability_enumeration::NAME.as_ptr())
        {
            create_info = create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(vk_error)?;
        let mut bootstrap = Bootstrap {
            instance: Some(instance),
            device: None,
            debug_messenger: None,
        };
        let instance = bootstrap
            .instance
            .as_ref()
            .ok_or(Error::Backend("Vulkan instance bootstrap was empty".into()))?;
        bootstrap.debug_messenger = if info.validation
            && extensions
                .iter()
                .any(|&extension| extension == debug_utils::NAME.as_ptr())
        {
            let loader = debug_utils::Instance::new(&entry, instance);
            let messenger =
                unsafe { loader.create_debug_utils_messenger(&debug_create_info(), None) }
                    .map_err(vk_error)?;
            Some((loader, messenger))
        } else {
            None
        };

        let surface_loader = surface::Instance::new(&entry, instance);
        let temporary_surface = if let Some(surface) = info.compatible_surface {
            Some(
                unsafe {
                    ash_window::create_surface(
                        &entry,
                        instance,
                        surface.display,
                        surface.window,
                        None,
                    )
                }
                .map_err(vk_error)?,
            )
        } else {
            None
        };

        let selection = select_physical_device(instance, &surface_loader, temporary_surface);
        if let Some(surface) = temporary_surface {
            unsafe { surface_loader.destroy_surface(surface, None) };
        }
        let selected = selection?;

        info!(
            device = %selected.name,
            api = %selected.api_version,
            "selected Vulkan physical device"
        );

        let (device, queues) = create_device(instance, &selected)?;
        bootstrap.device = Some(device);
        let device = bootstrap
            .device
            .as_ref()
            .ok_or(Error::Backend("Vulkan device bootstrap was empty".into()))?;
        let swapchain_loader = swapchain::Device::new(instance, device);
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: selected.raw,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: AllocationSizes::default(),
        })
        .map_err(backend_error)?;
        let Initialized {
            instance,
            device,
            debug_messenger,
        } = bootstrap.finish()?;

        Ok(Arc::new(Self {
            entry,
            instance,
            physical_device: selected.raw,
            device,
            surface_loader,
            swapchain_loader,
            families: selected.families,
            queues,
            capabilities: selected.capabilities,
            sampler_anisotropy: selected.sampler_anisotropy,
            non_coherent_atom_size: selected.non_coherent_atom_size,
            enabled_instance_extensions,
            allocator: Mutex::new(Some(allocator)),
            garbage: Mutex::new(Vec::new()),
            retirement_epoch: AtomicU64::new(0),
            debug_messenger,
        }))
    }

    pub(crate) fn allocate(
        &self,
        desc: &gpu_allocator::vulkan::AllocationCreateDesc<'_>,
    ) -> Result<Allocation> {
        self.allocator
            .lock()
            .as_mut()
            .expect("Vulkan allocator exists until context destruction")
            .allocate(desc)
            .map_err(backend_error)
    }

    pub(crate) fn allocator_free(&self, allocation: Allocation) {
        self.free(allocation);
    }

    pub(crate) fn retire(&self, value: Garbage) {
        let retire_at = self.retirement_epoch.load(Ordering::Relaxed) + RETIREMENT_DELAY;
        self.garbage
            .lock()
            .push(PendingGarbage { retire_at, value });
    }

    pub(crate) fn collect_garbage(&self) {
        let epoch = self.retirement_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let mut garbage = self.garbage.lock();
        let mut index = 0;
        while index < garbage.len() {
            if garbage[index].retire_at <= epoch {
                let pending = garbage.swap_remove(index);
                self.destroy(pending.value);
            } else {
                index += 1;
            }
        }
    }

    pub(crate) fn collect_all_garbage(&self) {
        let pending = std::mem::take(&mut *self.garbage.lock());
        for pending in pending {
            self.destroy(pending.value);
        }
    }

    pub(crate) fn queue(&self, queue: QueueType) -> vk::Queue {
        self.queues.get(crate::convert::queue(queue))
    }

    pub(crate) fn queue_family(&self, queue: QueueType) -> u32 {
        match queue {
            QueueType::Graphics => self.families.graphics,
            QueueType::Compute => self.families.compute,
            QueueType::Copy => self.families.copy,
        }
    }

    fn destroy(&self, garbage: Garbage) {
        unsafe {
            match garbage {
                Garbage::Buffer { raw, allocation } => {
                    self.device.destroy_buffer(raw, None);
                    self.free(allocation);
                }
                Garbage::Image { raw, allocation } => {
                    self.device.destroy_image(raw, None);
                    self.free(allocation);
                }
                Garbage::ImageView(raw) => self.device.destroy_image_view(raw, None),
                Garbage::Sampler(raw) => self.device.destroy_sampler(raw, None),
                Garbage::Shader(raw) => self.device.destroy_shader_module(raw, None),
                Garbage::BindGroupLayout(raw) => {
                    self.device.destroy_descriptor_set_layout(raw, None);
                }
                Garbage::DescriptorPool(raw) => self.device.destroy_descriptor_pool(raw, None),
                Garbage::PipelineLayout(raw) => self.device.destroy_pipeline_layout(raw, None),
                Garbage::Pipeline(raw) => self.device.destroy_pipeline(raw, None),
                Garbage::CommandPool(raw) => self.device.destroy_command_pool(raw, None),
                Garbage::Fence(raw) => self.device.destroy_fence(raw, None),
                Garbage::Semaphore(raw) => self.device.destroy_semaphore(raw, None),
                Garbage::Surface(raw) => self.surface_loader.destroy_surface(raw, None),
                Garbage::Swapchain {
                    raw,
                    views,
                    semaphores,
                } => {
                    for view in views {
                        self.device.destroy_image_view(view, None);
                    }
                    for semaphore in semaphores {
                        self.device.destroy_semaphore(semaphore, None);
                    }
                    self.swapchain_loader.destroy_swapchain(raw, None);
                }
            }
        }
    }

    fn free(&self, allocation: Allocation) {
        self.allocator
            .lock()
            .as_mut()
            .expect("Vulkan allocator exists while garbage is collected")
            .free(allocation)
            .expect("Vulkan allocation was created by this allocator");
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        let pending = self.garbage.get_mut().drain(..).collect::<Vec<_>>();
        for pending in pending {
            self.destroy(pending.value);
        }
        drop(self.allocator.get_mut().take());
        unsafe {
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = &self.debug_messenger {
                loader.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct SelectedDevice {
    raw: vk::PhysicalDevice,
    name: String,
    api_version: String,
    families: QueueFamilies,
    capabilities: Capabilities,
    sampler_anisotropy: bool,
    non_coherent_atom_size: u64,
    extensions: Vec<vk::ExtensionProperties>,
}

fn select_physical_device(
    instance: &ash::Instance,
    surface_loader: &surface::Instance,
    surface: Option<vk::SurfaceKHR>,
) -> Result<SelectedDevice> {
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(vk_error)?;
    devices
        .into_iter()
        .filter_map(|raw| inspect_device(instance, surface_loader, surface, raw))
        .max_by_key(|(score, _)| *score)
        .map(|(_, selected)| selected)
        .ok_or(Error::NoDevice)
}

fn inspect_device(
    instance: &ash::Instance,
    surface_loader: &surface::Instance,
    surface: Option<vk::SurfaceKHR>,
    raw: vk::PhysicalDevice,
) -> Option<(u64, SelectedDevice)> {
    let properties = unsafe { instance.get_physical_device_properties(raw) };
    if properties.api_version < vk::API_VERSION_1_3 {
        return None;
    }

    let extensions = unsafe { instance.enumerate_device_extension_properties(raw) }.ok()?;
    if !extension_available(&extensions, swapchain::NAME) {
        return None;
    }

    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
    {
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        unsafe { instance.get_physical_device_features2(raw, &mut features) };
    }
    if vulkan12.timeline_semaphore != vk::TRUE
        || vulkan12.vulkan_memory_model != vk::TRUE
        || vulkan13.dynamic_rendering != vk::TRUE
        || vulkan13.synchronization2 != vk::TRUE
    {
        return None;
    }

    let queue_properties = unsafe { instance.get_physical_device_queue_family_properties(raw) };
    let families = QueueFamilies::resolve(&queue_properties, surface_loader, surface, raw)?;
    let depth_format = [
        Format::Depth32Float,
        Format::Depth32FloatStencil8,
        Format::Depth24UnormStencil8,
        Format::Depth16Unorm,
    ]
    .into_iter()
    .find(|format| {
        unsafe {
            instance.get_physical_device_format_properties(raw, crate::convert::format(*format))
        }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    })?;

    let sample_flags = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    let max_samples = if sample_flags.contains(vk::SampleCountFlags::TYPE_8) {
        SampleCount::Eight
    } else if sample_flags.contains(vk::SampleCountFlags::TYPE_4) {
        SampleCount::Four
    } else if sample_flags.contains(vk::SampleCountFlags::TYPE_2) {
        SampleCount::Two
    } else {
        SampleCount::One
    };
    let sampler_anisotropy =
        unsafe { instance.get_physical_device_features(raw) }.sampler_anisotropy == vk::TRUE;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a supported Vulkan anisotropy limit is finite, positive, and at most 16"
    )]
    let max_sampler_anisotropy = if sampler_anisotropy {
        properties.limits.max_sampler_anisotropy.floor() as u16
    } else {
        1
    };

    let name = properties
        .device_name_as_c_str()
        .unwrap_or(c"unknown Vulkan device")
        .to_string_lossy()
        .into_owned();
    let mut score = u64::from(vk::api_version_minor(properties.api_version)) * 100;
    score += match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 10_000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1_000,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 500,
        _ => 0,
    };

    Some((
        score,
        SelectedDevice {
            raw,
            name,
            api_version: format!(
                "{}.{}.{}",
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
                vk::api_version_patch(properties.api_version)
            ),
            families,
            capabilities: Capabilities {
                depth_format,
                max_samples,
                max_sampler_anisotropy,
                dedicated_compute_queue: families.compute != families.graphics,
                dedicated_copy_queue: families.copy != families.graphics
                    && families.copy != families.compute,
            },
            sampler_anisotropy,
            non_coherent_atom_size: properties.limits.non_coherent_atom_size,
            extensions,
        },
    ))
}

impl QueueFamilies {
    fn resolve(
        properties: &[vk::QueueFamilyProperties],
        surface_loader: &surface::Instance,
        surface: Option<vk::SurfaceKHR>,
        physical_device: vk::PhysicalDevice,
    ) -> Option<Self> {
        let graphics = queue_index(properties, |flags| flags.contains(vk::QueueFlags::GRAPHICS))?;
        let compute = queue_index(properties, |flags| {
            flags.contains(vk::QueueFlags::COMPUTE) && !flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .unwrap_or(graphics);
        let copy = queue_index(properties, |flags| {
            flags.contains(vk::QueueFlags::TRANSFER)
                && !flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .unwrap_or(compute);
        let present = if let Some(surface) = surface {
            properties.iter().enumerate().find_map(|(index, _)| {
                let index = u32::try_from(index).ok()?;
                unsafe {
                    surface_loader
                        .get_physical_device_surface_support(physical_device, index, surface)
                        .ok()
                        .filter(|supported| *supported)
                        .map(|_| index)
                }
            })?
        } else {
            graphics
        };
        Some(Self {
            graphics,
            compute,
            copy,
            present,
        })
    }
}

fn queue_index(
    properties: &[vk::QueueFamilyProperties],
    predicate: impl Fn(vk::QueueFlags) -> bool,
) -> Option<u32> {
    properties
        .iter()
        .position(|property| property.queue_count > 0 && predicate(property.queue_flags))
        .and_then(|index| u32::try_from(index).ok())
}

fn create_device(
    instance: &ash::Instance,
    selected: &SelectedDevice,
) -> Result<(ash::Device, Queues)> {
    let unique_families = HashSet::from([
        selected.families.graphics,
        selected.families.compute,
        selected.families.copy,
        selected.families.present,
    ]);
    let priorities = [1.0_f32];
    let queue_infos = unique_families
        .iter()
        .map(|family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*family)
                .queue_priorities(&priorities)
        })
        .collect::<Vec<_>>();
    let mut extensions = vec![swapchain::NAME.as_ptr()];
    if extension_available(&selected.extensions, ash::khr::portability_subset::NAME) {
        extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }
    let features =
        vk::PhysicalDeviceFeatures::default().sampler_anisotropy(selected.sampler_anisotropy);
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .vulkan_memory_model(true);
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut vulkan12)
        .push_next(&mut vulkan13);
    let device =
        unsafe { instance.create_device(selected.raw, &create_info, None) }.map_err(vk_error)?;
    let queues = unsafe {
        Queues {
            graphics: device.get_device_queue(selected.families.graphics, 0),
            compute: device.get_device_queue(selected.families.compute, 0),
            copy: device.get_device_queue(selected.families.copy, 0),
            present: device.get_device_queue(selected.families.present, 0),
        }
    };
    Ok((device, queues))
}

fn extension_available(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions
        .iter()
        .any(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) == name })
}

const fn version((major, minor, patch): (u32, u32, u32)) -> u32 {
    vk::make_api_version(0, major, minor, patch)
}

fn debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        callback_data
            .as_ref()
            .and_then(|data| data.p_message.as_ref())
            .map_or(c"missing validation message", |message| {
                CStr::from_ptr(message)
            })
    };
    let message = message.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(target: "vulkan::validation", %message);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: "vulkan::validation", %message);
    } else {
        debug!(target: "vulkan::validation", %message);
    }
    vk::FALSE
}
