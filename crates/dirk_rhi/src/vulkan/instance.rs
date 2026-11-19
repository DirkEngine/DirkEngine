use std::{
    ffi::{CStr, CString, c_void},
    sync::atomic::{AtomicUsize, Ordering},
};

use ash::{ext::debug_utils, vk};
use tracing::{debug, error, warn};

use super::Check;
use crate::{Error, Result, RhiCreateInfo};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// A Vulkan instance and, with validation, the messenger counting its errors.
/// Destroys both when dropped, so a failed device creation cleans up.
pub(super) struct Instance {
    pub(super) raw: ash::Instance,
    messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    /// Errors reported by validation. Boxed so the address handed to the
    /// messenger stays stable; outlives the messenger, which is destroyed first.
    validation_errors: Box<AtomicUsize>,
    /// Whether the validation layer runs and the messenger counts its errors.
    validating: bool,
    /// Keeps the Vulkan library loaded for as long as the instance exists.
    pub(super) entry: ash::Entry,
}

impl Instance {
    /// Creates an instance with the portability, validation and debug
    /// extensions the request needs.
    pub(super) fn new(info: &RhiCreateInfo<'_>) -> Result<Self> {
        // SAFETY: the loaded library is kept alive by the returned instance.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| Error::native("loading the Vulkan library", error))?;
        // SAFETY: `entry` is a loaded Vulkan library.
        let available = unsafe { entry.enumerate_instance_extension_properties(None) }
            .check("vkEnumerateInstanceExtensionProperties")?;
        let has = |extensions: &[vk::ExtensionProperties], name: &CStr| {
            extensions
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(name))
        };

        let mut extensions = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has(&available, ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let mut layers = Vec::new();
        let mut synchronization_validation = false;
        if info.validation {
            if let Some(layer_extensions) = Self::validation_layer(&entry)? {
                layers.push(VALIDATION_LAYER.as_ptr());
                synchronization_validation =
                    has(&layer_extensions, ash::ext::validation_features::NAME);
                if synchronization_validation {
                    extensions.push(ash::ext::validation_features::NAME.as_ptr());
                }
            } else {
                warn!(
                    "Vulkan validation was requested but the Khronos validation layer is not installed"
                );
            }
            if has(&available, debug_utils::NAME) {
                extensions.push(debug_utils::NAME.as_ptr());
            }
        }
        let debug_messages = extensions.contains(&debug_utils::NAME.as_ptr());

        let application_name = CString::new(info.application_name).map_err(|_| {
            Error::invalid(
                "application name",
                info.application_name,
                "contains a NUL byte",
            )
        })?;
        let application = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .engine_name(c"DirkEngine")
            .api_version(vk::API_VERSION_1_3);
        let validation_errors = Box::new(AtomicUsize::new(0));
        let mut messenger_info = Self::messenger_info(&validation_errors);
        let enabled_validation = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_validation);
        let mut create_info = vk::InstanceCreateInfo::default()
            .flags(flags)
            .application_info(&application)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if debug_messages {
            // Also reports problems in instance creation and destruction.
            create_info = create_info.push_next(&mut messenger_info);
        }
        if synchronization_validation {
            create_info = create_info.push_next(&mut validation_features);
        }
        // SAFETY: every pointer in `create_info` outlives the call.
        let raw = unsafe { entry.create_instance(&create_info, None) }.check("vkCreateInstance")?;

        let mut instance = Self {
            raw,
            messenger: None,
            validation_errors,
            validating: !layers.is_empty() && debug_messages,
            entry,
        };
        if debug_messages {
            let loader = debug_utils::Instance::new(&instance.entry, &instance.raw);
            let info = Self::messenger_info(&instance.validation_errors);
            // SAFETY: the counter outlives the messenger, which `Drop` destroys first.
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }
                .check("vkCreateDebugUtilsMessengerEXT")?;
            instance.messenger = Some((loader, messenger));
        }
        Ok(instance)
    }

    /// Whether debug utilities, which name objects for debugging tools, are
    /// enabled.
    pub(super) fn debug_utils(&self) -> bool {
        self.messenger.is_some()
    }

    /// Number of errors validation reported, or `None` if it does not run.
    pub(super) fn validation_errors(&self) -> Option<usize> {
        self.validating
            .then(|| self.validation_errors.load(Ordering::Relaxed))
    }

    /// The validation layer's extensions, or `None` if it is not installed.
    fn validation_layer(entry: &ash::Entry) -> Result<Option<Vec<vk::ExtensionProperties>>> {
        // SAFETY: `entry` is a loaded Vulkan library.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .check("vkEnumerateInstanceLayerProperties")?;
        if !layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
        {
            return Ok(None);
        }
        // SAFETY: the layer is installed.
        unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
            .check("vkEnumerateInstanceExtensionProperties")
            .map(Some)
    }

    /// Messenger reporting warnings and errors, counting errors in `errors`.
    fn messenger_info(errors: &AtomicUsize) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
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
            .pfn_user_callback(Some(Self::message))
            .user_data(std::ptr::from_ref(errors).cast_mut().cast())
    }

    /// Logs a validation message and counts errors.
    unsafe extern "system" fn message(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        _types: vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        errors: *mut c_void,
    ) -> vk::Bool32 {
        // SAFETY: Vulkan passes valid callback data for the duration of the call.
        let message = unsafe { data.as_ref() }
            .and_then(|data| unsafe { data.message_as_c_str() })
            .map_or_else(|| "<no message>".into(), std::ffi::CStr::to_string_lossy);
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            // SAFETY: `errors` is the instance's boxed counter, which outlives
            // the messenger.
            if let Some(errors) = unsafe { errors.cast::<AtomicUsize>().as_ref() } {
                errors.fetch_add(1, Ordering::Relaxed);
            }
            error!(target: "vulkan", "{message}");
        } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
            warn!(target: "vulkan", "{message}");
        } else {
            debug!(target: "vulkan", "{message}");
        }
        vk::FALSE
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every child object was destroyed by its owner, which the
        // instance outlives.
        unsafe {
            if let Some((loader, messenger)) = self.messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}
