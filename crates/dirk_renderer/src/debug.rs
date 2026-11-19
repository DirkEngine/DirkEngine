use std::{ffi::CStr, os::raw::c_void};

use ash::{Entry, vk};
use tracing::{debug, error, info, trace, warn};

use crate::{Error, Result};

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    use ash::vk;

    // SAFETY: Vulkan calls this with valid callback data whose `p_message` is
    // a NUL-terminated string that lives for the duration of the call.
    let message = unsafe { CStr::from_ptr((*callback_data).p_message).to_string_lossy() };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(target: "vulkan::validation", "{}", message);
        }

        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn! (target: "vulkan::validation", "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info! (target: "vulkan::validation", "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!(target: "vulkan::validation", "{}", message);
        }
        _ => trace!(target: "vulkan::validation", "{}", message),
    }

    vk::FALSE
}

pub fn debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    let severity_flags = vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;

    let message_type_flags = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(severity_flags)
        .message_type(message_type_flags)
        .pfn_user_callback(Some(debug_callback))
}

pub fn validate_instance_layers(entry: &Entry, validation_layers: &[*const i8]) -> Result<()> {
    // SAFETY: `entry` holds the loaded Vulkan library.
    let available = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .unwrap_or_default()
    };

    for &required in validation_layers {
        // SAFETY: callers pass pointers to static NUL-terminated C string literals.
        let required = unsafe {
            use std::ffi::CStr;
            CStr::from_ptr(required)
        };
        let found = available
            .iter()
            // SAFETY: Vulkan fills `layer_name` with a NUL-terminated string
            // within the array.
            .any(|ext| unsafe { CStr::from_ptr(ext.layer_name.as_ptr()) } == required);

        if !found {
            return Err(Error::ValidationLayerNotFound(
                required.to_string_lossy().into_owned(),
            ));
        }
    }

    Ok(())
}

pub fn create_debug_messenger(
    entry: &Entry,
    instance: &ash::Instance,
    debug_create_info: &vk::DebugUtilsMessengerCreateInfoEXT<'_>,
) -> Result<vk::DebugUtilsMessengerEXT> {
    use ash::ext::debug_utils;

    let loader = debug_utils::Instance::new(entry, instance);
    // SAFETY: the instance was created with `VK_EXT_debug_utils`, and
    // `debug_create_info` points at an `extern "system"` callback, where a
    // panic aborts instead of unwinding into Vulkan.
    Ok(unsafe { loader.create_debug_utils_messenger(debug_create_info, None)? })
}
