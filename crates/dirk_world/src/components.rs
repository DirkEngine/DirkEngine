//! This module has a bunch of frequently used and central [`Component`]s
//!
//! [`Component`]: dirk_universe::components::Component

use std::{collections::HashSet, sync::Arc};

use dirk_assets::{AssetLoad, AssetRegistry, Model};
use dirk_universe::{
    CommandBuffer, Entity, Universe, changes::ComponentChange, components::Component,
    systems::System,
};
use glam::{Mat4, Quat, Vec3};
use tracing::warn;

/// Marks an entity as having a renderable mesh.
///
/// The `model` field is resolved at render time against the engine's asset
/// registry. If no matching asset is found the entity is silently skipped.
///
/// # Examples
/// ```
/// # use dirk_world::components::Renderable;
/// use dirk_assets::{AssetHandle, AssetType};
/// let r = Renderable::new(AssetHandle::from_raw("meshes/cube.glb", AssetType::Model));
/// assert_eq!(r.model.raw(), "meshes/cube.glb");
/// ```
#[derive(Debug, Clone, Component)]
pub struct Renderable {
    /// Asset-registry key for the mesh to render (e.g. `"meshes/cube.glb"`).
    pub model: dirk_assets::AssetHandle,
    /// This is a tokio `JoinHandle` under the hood. This keeps the `Handle<T>`
    /// alive while the `JoinHandle` is alive. This means that this field
    /// is stopping the asset form being unloaded by the renderer.
    ///
    /// Please do not try to await/poll this future, this would drop the handle
    /// and lead the asset to disapear on the renderer
    handle: Option<Arc<AssetLoad<Model>>>,
}

impl Renderable {
    /// Creates a new [`Renderable`] component from an [`AssetHandle`].
    ///
    /// [`AssetHandle`]: dirk_assets::AssetHandle
    #[must_use]
    pub fn new(model: dirk_assets::AssetHandle) -> Self {
        Self {
            model,
            handle: None,
        }
    }
}

/// Loads models referenced by added or updated [`Renderable`] components.
pub struct ModelUploadSystem {
    assets: AssetRegistry,
}

impl ModelUploadSystem {
    /// Creates a new [`ModelUploadSystem`] using the provided [`AssetRegistry`].
    #[must_use]
    pub fn new(assets: AssetRegistry) -> Self {
        Self { assets }
    }
}

impl System for ModelUploadSystem {
    fn run(&mut self, cmd: &mut CommandBuffer, universe: &Universe, _: f64) {
        let mut changed = HashSet::new();
        for change in universe.component_changes::<Renderable>() {
            let (ComponentChange::Added { entity, .. } | ComponentChange::Updated { entity, .. }) =
                change
            else {
                continue;
            };
            // Load only the final live value, once, even after several updates.
            if changed.insert(entity)
                && let Some(component) = universe.component::<Renderable>(entity)
            {
                self.load(cmd, entity, component);
            }
        }
    }
}

impl ModelUploadSystem {
    fn load(&self, cmd: &mut CommandBuffer, entity: Entity, component: &Renderable) {
        if component.handle.is_some() {
            return;
        }
        let handle = self.assets.load_asset::<Model>(&component.model);
        cmd.set_component(
            entity,
            Renderable {
                handle: Some(Arc::new(handle)),
                ..component.clone()
            },
        );
    }
}

/// Spatial transform for an entity: position, orientation, and scale.
///
/// Rotation is stored as a unit quaternion.
///
/// # Examples
/// ```
/// # use dirk_world::components::Transform;
/// # use glam::{Quat, Vec3};
/// let t = Transform {
///     location: Vec3::new(1.0, 0.0, 0.0),
///     rotation: Quat::IDENTITY,
///     scale:    Vec3::ONE,
/// };
/// // The forward vector of an un-rotated transform points along the engine's
/// // canonical forward axis.
/// let fwd = t.forward();
/// assert!((fwd.length() - 1.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Component)]
pub struct Transform {
    /// World-space position.
    pub location: Vec3,
    /// World-space orientation.
    pub rotation: Quat,
    /// Per-axis scale factor. `Vec3::ONE` is the identity scale.
    pub scale: Vec3,
}

impl Default for Transform {
    /// Returns the identity transform: origin, no rotation, unit scale.
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Builds the full model matrix (`T × R × S`) for this transform.
    #[must_use]
    pub fn matrix(&self) -> Mat4 {
        let translation = Mat4::from_translation(self.location);
        let rotation = Mat4::from_quat(self.rotation);
        let scale = Mat4::from_scale(self.scale);

        translation * rotation * scale
    }

    /// Returns the orientation as a unit quaternion.
    #[must_use]
    pub fn rotation_quat(&self) -> glam::Quat {
        self.rotation
    }

    /// Returns the unit vector pointing "forward" from this transform.
    ///
    /// The result is obtained by rotating the engine's canonical forward
    /// direction ([`utils::FORWARD_DIRECTION`]) by the current orientation.
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        self.rotation_quat() * dirk_utils::FORWARD_DIRECTION
    }

    /// Returns the horizontal forward direction of the transform.
    #[must_use]
    pub fn horizontal_forward(&self) -> Vec3 {
        let mut forward = self.forward();
        forward.y = 0.0;
        if forward.length_squared() < f32::EPSILON {
            return dirk_utils::FORWARD_DIRECTION;
        }
        forward.normalize()
    }

    /// Returns the movement direction when an `input` is applied to `self`.
    #[must_use]
    pub fn movement_direction(&self, input: glam::Vec3) -> glam::Vec3 {
        let forward = self.forward();
        let right = dirk_utils::UP_DIRECTION.cross(forward).normalize_or_zero();
        ((right * input.x) + (dirk_utils::UP_DIRECTION * input.y) - (forward * input.z))
            .normalize_or_zero()
    }

    /// Rotates this transform by pointer movement in physical pixels.
    pub fn rotate_by_pointer_delta(&mut self, delta: glam::DVec2, sensitivity: f32) {
        if delta == glam::DVec2::ZERO {
            return;
        }

        #[allow(clippy::cast_possible_truncation)]
        let delta = delta.as_vec2();
        let yaw = Quat::from_axis_angle(dirk_utils::UP_DIRECTION, -delta.x * sensitivity);
        let yawed = yaw * self.rotation;
        let right = (yawed * Vec3::X).normalize_or_zero();
        if right == Vec3::ZERO {
            return;
        }

        let pitch = Quat::from_axis_angle(right, -delta.y * sensitivity);
        self.rotation = (pitch * yawed).normalize();
    }

    /// Builds a **left-handed** view matrix for a camera placed at this
    /// transform, looking in the [`forward`](Self::forward) direction.
    pub fn view(&self) -> Mat4 {
        let forward = self.forward();
        if forward.cross(dirk_utils::UP_DIRECTION).length() < 1e-4 {
            warn!(
                "camera forward {:?} is parallel to UP — view matrix will be NaN",
                forward
            );
        }
        glam::camera::lh::view::look_at_mat4(
            self.location,
            self.location + forward,
            dirk_utils::UP_DIRECTION,
        )
    }
}

impl From<Transform> for Mat4 {
    /// Converts the transform to its model matrix via [`Transform::matrix`].
    fn from(transform: Transform) -> Self {
        transform.matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_input_moves_against_transform_forward() {
        let movement = Transform::default().movement_direction(Vec3::Z);

        assert_eq!(movement, -dirk_utils::FORWARD_DIRECTION);
    }
}
