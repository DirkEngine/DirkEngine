//! This module contains all the shared shader types.

/// The Uniform buffer for scene specific data
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SceneUbo {
    /// The view matrix to render the scene
    pub view: glam::Mat4,
    /// The projection matrix to render the scene
    pub proj: glam::Mat4,
}

/// The Uniform buffer for proxy specific data
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProxyUbo {
    /// The model matrix of the proxy
    pub model: glam::Mat4,
}

/// Per-frame parameters used to place egui vertices in clip space.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EguiUbo {
    /// Logical editor dimensions in points.
    pub screen_size: glam::Vec2,
    /// One when the render target performs sRGB encoding, zero otherwise.
    pub output_is_srgb: f32,
    /// Uniform alignment padding.
    pub padding: f32,
}
