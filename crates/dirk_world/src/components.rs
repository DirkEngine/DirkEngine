//! This module has a bunch of frequently used and central [`Component`]s
//!
//! [`Component`]: dirk_universe::components::Component

use std::collections::{HashMap, HashSet};

use dirk_assets::{AssetHandle, AssetLoad, AssetRegistry, Handle, Model};
use dirk_universe::{
    Entity,
    changes::ComponentChange,
    components::Component,
    query::QueryView,
    systems::{Changes, System},
};
use glam::{Mat4, Quat, Vec3};
use tracing::error;

/// Marks an entity as having a renderable mesh.
///
/// The `model` field is resolved by [`ModelUploadSystem`] against the engine's
/// asset registry. Changing it requests the new model on the next tick.
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
    pub model: AssetHandle,
}

impl Renderable {
    /// Creates a new [`Renderable`] component from an [`AssetHandle`].
    ///
    /// [`AssetHandle`]: dirk_assets::AssetHandle
    #[must_use]
    pub fn new(model: AssetHandle) -> Self {
        Self { model }
    }
}

/// Loads models referenced by added or updated [`Renderable`] components.
pub struct ModelUploadSystem {
    assets: AssetRegistry,
    requests: HashMap<Entity, ModelRequest>,
}

struct ModelRequest {
    model: AssetHandle,
    status: ModelLoad,
}

enum ModelLoad {
    Pending(AssetLoad<Model>),
    Ready { _handle: Handle<Model> },
    Failed,
}

impl ModelRequest {
    fn new(assets: &AssetRegistry, model: AssetHandle) -> Self {
        let status = ModelLoad::Pending(assets.load_asset::<Model>(&model));
        Self { model, status }
    }

    fn poll(&mut self, entity: Entity) {
        let result = match &mut self.status {
            ModelLoad::Pending(load) => load.try_poll(),
            ModelLoad::Ready { .. } | ModelLoad::Failed => None,
        };
        if let Some(result) = result {
            self.status = match result {
                Ok(handle) => ModelLoad::Ready { _handle: handle },
                Err(error) => {
                    error!(?entity, asset = %self.model, error = ?error, "failed to load model");
                    ModelLoad::Failed
                }
            };
        }
    }
}

impl ModelUploadSystem {
    /// Creates a new [`ModelUploadSystem`] using the provided [`AssetRegistry`].
    #[must_use]
    pub fn new(assets: AssetRegistry) -> Self {
        Self {
            assets,
            requests: HashMap::new(),
        }
    }
}

impl System<(Changes<'_>, QueryView<'_, &Renderable>)> for ModelUploadSystem {
    fn run(&mut self, (changes, renderables): (Changes<'_>, QueryView<'_, &Renderable>)) {
        let mut affected_entities = HashSet::new();
        for change in changes.components::<Renderable>() {
            let entity = match change {
                ComponentChange::Added { entity, .. }
                | ComponentChange::Updated { entity, .. }
                | ComponentChange::Removed { entity, .. } => entity,
            };
            affected_entities.insert(entity);
        }

        // Reconcile once against the final component value, even if it changed
        // several times in this command batch.
        for entity in affected_entities {
            if let Some(component) = renderables.get(entity) {
                let component = component.into_params();
                if self
                    .requests
                    .get(&entity)
                    .is_none_or(|request| request.model != component.model)
                {
                    self.requests.insert(
                        entity,
                        ModelRequest::new(&self.assets, component.model.clone()),
                    );
                }
            } else {
                self.requests.remove(&entity);
            }
        }

        for (entity, request) in &mut self.requests {
            request.poll(*entity);
        }
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
    #[must_use]
    pub fn view(&self) -> Mat4 {
        // The inverse rigid transform preserves camera roll and ignores model scale.
        Mat4::from_rotation_translation(self.rotation, self.location).inverse()
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
    #[test]
    fn camera_view_preserves_roll_and_ignores_scale() {
        let transform = Transform {
            location: Vec3::new(3.0, -2.0, 5.0),
            rotation: Quat::from_rotation_z(0.7) * Quat::from_rotation_y(0.3),
            scale: Vec3::new(2.0, 3.0, 0.0),
        };
        let view = transform.view();
        assert!(
            view.transform_point3(transform.location)
                .abs_diff_eq(Vec3::ZERO, 1e-5)
        );
        assert!(
            view.transform_vector3(transform.rotation * Vec3::Y)
                .abs_diff_eq(Vec3::Y, 1e-5)
        );
        assert!(
            view.transform_vector3(transform.rotation * Vec3::Z)
                .abs_diff_eq(Vec3::Z, 1e-5)
        );
    }
}
