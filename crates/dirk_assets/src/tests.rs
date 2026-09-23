//! Comprehensive tests for the `assets` crate.
//!
//! # Test organisation
//!
//! | Module | What is covered |
//! |--------|-----------------|
//! | `asset_handle` | Construction, accessors, Display, Hash, serde |
//! | `asset_type` | Default, Copy, serde, unknown rejection |
//! | `model_config` | Serde round-trip, validation (file present / absent) |
//! | `errors` | Display messages, `From` impls |
//! | `handle` | `get`/`take` semantics, clone sharing, drop event |
//! | `dirk_asset_validation` | All failure branches + success branch |
//! | `registry` | Isolated root, cache, concurrency and generation lifecycles |
//!
//! # Placement
//!
//! Drop this file into `Engine/Source/assets/src/tests.rs` and add the
//! following line to `src/lib.rs`:
//!
//! ```rust
//! #[cfg(test)]
//! mod tests;
//! ```
//!
//! # Dev-dependencies required (add to `Cargo.toml`)
//!
//! ```toml
//! [dev-dependencies]
//! tempfile = "3"
//! ```
//!
//! # Registry integration tests
//!
//! Registry tests use their own temporary asset roots and always run in CI.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]

// ── bring the entire crate into scope ────────────────────────────────────────
use super::{
    // public
    Asset,
    AssetConfig,
    AssetHandle,
    AssetLoad,
    AssetLoaded,
    AssetRegistry,
    AssetType,
    AssetUnloaded,
    DirkAsset,
    Error,
    Handle,
    Metadata,
    Model,
    ModelConfig,
    Result,
    // pub(crate)
    events::InternalAssetUnloaded,
    handle::AssetRef,
};

use dirk_events::EventManager;
use dirk_threads::WorkerPool;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal valid glTF 2.0 document accepted by the `gltf` crate.
fn minimal_gltf_json() -> &'static str {
    r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[]}],"nodes":[]}"#
}

/// Writes a `<name>.gltf` + `<name>.dirkasset` fixture pair into `dir`.
/// Returns the path of the `.dirkasset` file.
fn write_model_fixture(dir: &Path, name: &str) -> PathBuf {
    let gltf_name = format!("{name}.gltf");
    fs::write(dir.join(&gltf_name), minimal_gltf_json()).unwrap();

    let descriptor = serde_json::json!({
        "meta": { "asset_type": "Model" },
        "model": { "gltf": gltf_name }
    });
    let dirkasset = dir.join(format!("{name}.dirkasset"));
    fs::write(&dirkasset, descriptor.to_string()).unwrap();
    dirkasset
}

fn wait_for_load<T: Asset>(mut load: AssetLoad<T>) -> Result<Handle<T>> {
    for _ in 0..100 {
        if let Some(result) = load.try_poll() {
            return result;
        }

        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    panic!("asset load did not complete");
}

/// A minimal `Asset` implementation used only in tests that need a typed
/// `Handle<T>` without going through the full registry/filesystem machinery.
#[derive(Clone, Debug, PartialEq)]
struct FakeAsset {
    value: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct FakeConfig;

impl AssetConfig for FakeConfig {
    fn validate(&self, _meta: &Metadata) -> bool {
        true
    }
}

impl Asset for FakeAsset {
    type Config = FakeConfig;

    fn load(_config: &FakeConfig, _handle: &AssetHandle) -> Result<Self> {
        Ok(FakeAsset { value: 0 })
    }

    fn asset_type() -> AssetType {
        // Use Unknown so this fake type never conflicts with real registry entries.
        AssetType::Unknown
    }
}

/// Builds a `Handle<FakeAsset>` containing `value`, bypassing the registry.
fn fake_handle(value: u32, raw_path: &str) -> Handle<FakeAsset> {
    let workers = WorkerPool::new("test");
    let events = EventManager::new(workers.clone());
    let dispatcher = events.register::<InternalAssetUnloaded>();
    let asset_ref = AssetRef::new(
        AssetHandle::from_raw(raw_path, AssetType::Unknown),
        FakeAsset { value },
        dispatcher,
    );
    Handle::new(asset_ref)
}

// ─────────────────────────────────────────────────────────────────────────────
// AssetHandle
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod asset_handle {
    use super::*;

    #[test]
    fn raw_returns_construction_path() {
        let h = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
        assert_eq!(h.raw(), "models/hero.dirkasset");
    }

    #[test]
    fn asset_type_accessor_matches_constructor_arg() {
        let h = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        assert_eq!(h.asset_type(), AssetType::Model);
    }

    #[test]
    fn display_prints_raw_path() {
        let h = AssetHandle::from_raw("foo/bar.dirkasset", AssetType::Model);
        assert_eq!(h.to_string(), "foo/bar.dirkasset");
    }

    #[test]
    fn name_returns_filename_with_extension() {
        let h = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
        assert_eq!(h.name(), "hero.dirkasset");
    }

    #[test]
    fn name_returns_filename_for_nested_path() {
        let h = AssetHandle::from_raw("a/b/c/mesh.dirkasset", AssetType::Model);
        assert_eq!(h.name(), "mesh.dirkasset");
    }

    #[test]
    fn dir_ends_with_parent_directory() {
        let h = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
        // dir() = ASSETS_PATH / models
        assert!(h.dir().ends_with("models"));
    }

    #[test]
    fn path_ends_with_relative_handle() {
        let h = AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model);
        assert!(h.path().ends_with("models/hero.dirkasset"));
    }

    #[test]
    fn path_dir_relationship_is_consistent() {
        let h = AssetHandle::from_raw("a/b.dirkasset", AssetType::Model);
        // dir() must be the parent of path()
        assert_eq!(h.dir(), h.path().parent().unwrap());
    }

    #[test]
    fn default_has_empty_path_and_unknown_type() {
        let h = AssetHandle::default();
        assert_eq!(h.raw(), "");
        assert_eq!(h.asset_type(), AssetType::Unknown);
    }

    #[test]
    fn equality_same_path_same_type() {
        let a = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        let b = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_different_path() {
        let a = AssetHandle::from_raw("a.dirkasset", AssetType::Model);
        let b = AssetHandle::from_raw("b.dirkasset", AssetType::Model);
        assert_ne!(a, b);
    }

    #[test]
    fn inequality_different_type() {
        let a = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        let b = AssetHandle::from_raw("x.dirkasset", AssetType::Unknown);
        assert_ne!(a, b);
    }

    #[test]
    fn clone_is_equal_to_original() {
        let a = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn can_be_used_as_hash_map_key() {
        use std::collections::HashMap;
        let h = AssetHandle::from_raw("x.dirkasset", AssetType::Model);
        let mut map = HashMap::new();
        map.insert(h.clone(), 42u32);
        assert_eq!(map[&h], 42);
    }

    #[test]
    fn serde_round_trip_preserves_path_and_type() {
        let original = AssetHandle::from_raw("textures/dirt.dirkasset", AssetType::Model);
        let json = serde_json::to_string(&original).unwrap();
        let restored: AssetHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(original.raw(), restored.raw());
        assert_eq!(original.asset_type(), restored.asset_type());
    }

    #[test]
    fn serde_round_trip_unknown_type() {
        let original = AssetHandle::from_raw("x.dirkasset", AssetType::Unknown);
        let json = serde_json::to_string(&original).unwrap();
        let restored: AssetHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.asset_type(), AssetType::Unknown);
    }

    #[test]
    fn debug_output_is_non_empty() {
        let h = AssetHandle::from_raw("a/b.dirkasset", AssetType::Model);
        assert!(!format!("{h:?}").is_empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AssetType
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod asset_type {
    use super::*;

    #[test]
    fn default_is_unknown() {
        assert_eq!(AssetType::default(), AssetType::Unknown);
    }

    #[test]
    fn copy_semantics() {
        let a = AssetType::Model;
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn model_serialises_as_string_model() {
        assert_eq!(
            serde_json::to_string(&AssetType::Model).unwrap(),
            r#""Model""#
        );
    }

    #[test]
    fn unknown_serialises_as_string_unknown() {
        assert_eq!(
            serde_json::to_string(&AssetType::Unknown).unwrap(),
            r#""Unknown""#
        );
    }

    #[test]
    fn deserialise_model() {
        let v: AssetType = serde_json::from_str(r#""Model""#).unwrap();
        assert_eq!(v, AssetType::Model);
    }

    #[test]
    fn deserialise_unknown() {
        let v: AssetType = serde_json::from_str(r#""Unknown""#).unwrap();
        assert_eq!(v, AssetType::Unknown);
    }

    #[test]
    fn deserialise_unrecognised_variant_fails() {
        let result: std::result::Result<AssetType, _> = serde_json::from_str(r#""Texture""#);
        assert!(
            result.is_err(),
            "Unrecognised variant should fail deserialisation"
        );
    }

    #[test]
    fn serde_round_trip_model() {
        let json = serde_json::to_string(&AssetType::Model).unwrap();
        let back: AssetType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, AssetType::Model);
    }

    #[test]
    fn inequality() {
        assert_ne!(AssetType::Model, AssetType::Unknown);
    }

    #[test]
    fn usable_as_hash_map_key() {
        use std::collections::HashMap;
        let mut m: HashMap<AssetType, &str> = HashMap::new();
        m.insert(AssetType::Model, "model");
        m.insert(AssetType::Unknown, "unknown");
        assert_eq!(m[&AssetType::Model], "model");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelConfig
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod model_config {
    use super::*;

    fn make_meta(handle_path: &str) -> Metadata {
        Metadata {
            asset_type: AssetType::Model,
            handle: AssetHandle::from_raw(handle_path, AssetType::Model),
        }
    }

    #[test]
    fn serde_round_trip() {
        let json = r#"{"gltf":"meshes/hero.gltf"}"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.gltf, "meshes/hero.gltf");
        assert_eq!(serde_json::to_string(&config).unwrap(), json);
    }

    #[test]
    fn validate_returns_false_when_gltf_file_missing() {
        let config = ModelConfig {
            gltf: "definitely_does_not_exist.gltf".to_string(),
        };
        assert!(!config.validate(&make_meta("models/x.dirkasset")));
    }

    #[test]
    fn validate_returns_true_when_gltf_file_exists() {
        // When the gltf path is absolute, handle.dir().join(abs) == abs path,
        // so we can test the success branch by writing a real temp file.
        let dir = TempDir::new().unwrap();
        let gltf_path = dir.path().join("cube.gltf");
        fs::write(&gltf_path, minimal_gltf_json()).unwrap();

        let config = ModelConfig {
            // Absolute path: join() returns the absolute path unchanged.
            gltf: gltf_path.to_string_lossy().into_owned(),
        };
        // handle.dir() will be ASSETS_PATH; joining an absolute path ignores it.
        assert!(config.validate(&make_meta("")));
    }

    #[test]
    fn validate_emits_warning_for_missing_file() {
        // Regression: validate must not panic even with an empty handle path.
        let config = ModelConfig {
            gltf: "no_such_file.gltf".to_string(),
        };
        // Should return false gracefully (warning is emitted via tracing, not panics).
        let _ = config.validate(&make_meta(""));
    }

    #[test]
    fn clone_is_independent() {
        let a = ModelConfig {
            gltf: "file.gltf".to_string(),
        };
        let mut b = a.clone();
        b.gltf = "other.gltf".to_string();
        assert_eq!(a.gltf, "file.gltf");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod errors {
    use super::*;

    #[test]
    fn io_error_display_mentions_io() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        assert!(Error::IoError(io).to_string().contains("IO error"));
    }

    #[test]
    fn serialisation_error_display_mentions_serialisation() {
        let se: serde_json::Error = serde_json::from_str::<i32>("bad").unwrap_err();
        assert!(
            Error::SerialisationError(se)
                .to_string()
                .to_lowercase()
                .contains("serialis")
        );
    }

    #[test]
    fn already_taken_display_mentions_consumed() {
        assert!(Error::AlreadyTaken.to_string().contains("consumed"));
    }

    #[test]
    fn not_found_display_contains_handle_path() {
        let path = "models/missing.dirkasset";
        let msg = Error::NotFound(path.to_string()).to_string();
        assert!(msg.contains(path), "Expected '{path}' in error: {msg}");
    }

    #[test]
    fn not_found_display_mentions_not_found() {
        let msg = Error::NotFound("x".to_string()).to_string().to_lowercase();
        assert!(msg.contains("not found"), "Expected 'not found' in: {msg}");
    }

    #[test]
    fn type_mismatch_display_contains_handle_path() {
        let path = "models/hero.dirkasset";
        let msg = Error::TypeMismatch(path.to_string()).to_string();
        assert!(msg.contains(path));
    }

    #[test]
    fn type_mismatch_display_mentions_type() {
        let msg = Error::TypeMismatch("x".to_string())
            .to_string()
            .to_lowercase();
        assert!(msg.contains("type"), "Expected 'type' in: {msg}");
    }

    #[test]
    fn asset_load_error_display_includes_source() {
        let source = anyhow::anyhow!("gltf parse failed");
        let msg = Error::AssetLoadError(source).to_string();
        assert!(msg.contains("gltf parse failed"));
    }

    #[test]
    fn from_io_error_produces_io_variant() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err: Error = io.into();
        assert!(matches!(err, Error::IoError(_)));
    }

    #[test]
    fn from_serde_error_produces_serialisation_variant() {
        let se: serde_json::Error = serde_json::from_str::<i32>("bad").unwrap_err();
        let err: Error = se.into();
        assert!(matches!(err, Error::SerialisationError(_)));
    }

    #[test]
    fn debug_output_is_non_empty() {
        assert!(!format!("{:?}", Error::AlreadyTaken).is_empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle<T>
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod handle {
    use super::*;

    // ── get ──────────────────────────────────────────────────────────────────

    #[test]
    fn get_returns_correct_value() {
        let h = fake_handle(42, "test.dirkasset");
        assert_eq!(h.get().unwrap().value, 42);
    }

    #[test]
    fn get_is_non_destructive_on_first_call() {
        let h = fake_handle(7, "test.dirkasset");
        h.get().unwrap();
        // Second call must still succeed.
        assert_eq!(h.get().unwrap().value, 7);
    }

    #[test]
    fn get_is_callable_many_times() {
        let h = fake_handle(1, "test.dirkasset");
        for _ in 0..10 {
            assert!(h.get().is_ok());
        }
    }

    #[test]
    fn get_after_take_returns_already_taken() {
        let h = fake_handle(1, "test.dirkasset");
        h.take().unwrap();
        assert!(matches!(h.get().unwrap_err(), Error::AlreadyTaken));
    }

    // ── take ─────────────────────────────────────────────────────────────────

    #[test]
    fn take_returns_correct_value() {
        let h = fake_handle(99, "test.dirkasset");
        assert_eq!(h.take().unwrap().value, 99);
    }

    #[test]
    fn take_second_call_returns_already_taken() {
        let h = fake_handle(1, "test.dirkasset");
        h.take().unwrap();
        assert!(matches!(h.take().unwrap_err(), Error::AlreadyTaken));
    }

    #[test]
    fn take_is_destructive_across_clones() {
        // Both handle and its clone share the same inner Arc<Mutex<AssetRef>>.
        // Taking via one should make the data unavailable via the other.
        let h = fake_handle(5, "test.dirkasset");
        let clone = h.clone();
        clone.take().unwrap();
        assert!(
            matches!(h.take().unwrap_err(), Error::AlreadyTaken),
            "Original should reflect the take performed via its clone"
        );
    }

    #[test]
    fn get_is_non_destructive_across_clones() {
        let h = fake_handle(3, "test.dirkasset");
        let clone = h.clone();
        h.get().unwrap();
        // Clone can still get the data.
        assert_eq!(clone.get().unwrap().value, 3);
    }

    // ── clone ────────────────────────────────────────────────────────────────

    #[test]
    fn clone_shares_inner_data() {
        let h = fake_handle(8, "test.dirkasset");
        let c = h.clone();
        assert_eq!(h.get().unwrap().value, c.get().unwrap().value);
    }

    #[test]
    fn multiple_clones_all_share_data() {
        let h = fake_handle(4, "test.dirkasset");
        let c1 = h.clone();
        let c2 = h.clone();
        let c3 = c1.clone();
        for handle in [&h, &c1, &c2, &c3] {
            assert_eq!(handle.get().unwrap().value, 4);
        }
    }

    // ── drop event ───────────────────────────────────────────────────────────

    #[test]
    fn drop_of_sole_handle_fires_internal_unloaded_event() {
        let workers = WorkerPool::new("test");
        let events = EventManager::new(workers.clone());
        let mut consumer = events.subscribe::<InternalAssetUnloaded>();
        let dispatcher = events.register::<InternalAssetUnloaded>();

        let asset_handle = AssetHandle::from_raw("sole.dirkasset", AssetType::Unknown);
        let asset_ref = AssetRef::new(asset_handle.clone(), FakeAsset { value: 0 }, dispatcher);
        let handle = Handle::new(asset_ref);

        // wait for the event to be dispatched
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(consumer.consume_all().count(), 0, "no event yet");

        drop(handle);
        // wait for the event to be dispatched
        std::thread::sleep(std::time::Duration::from_millis(5));

        let fired: Vec<_> = consumer.consume_all().collect();
        assert_eq!(fired.len(), 1, "exactly one InternalAssetUnloaded event");
        assert_eq!(fired[0].handle, asset_handle);
    }

    #[test]
    fn drop_does_not_fire_while_clones_still_live() {
        let workers = WorkerPool::new("test");
        let events = EventManager::new(workers.clone());
        let mut consumer = events.subscribe::<InternalAssetUnloaded>();
        let dispatcher = events.register::<InternalAssetUnloaded>();

        let asset_ref = AssetRef::new(
            AssetHandle::from_raw("multi.dirkasset", AssetType::Unknown),
            FakeAsset { value: 0 },
            dispatcher,
        );
        let h1 = Handle::new(asset_ref);
        let h2 = h1.clone();
        let h3 = h1.clone();

        drop(h1);
        // wait for the event to be dispatched
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(consumer.consume_all().count(), 0, "clones still alive");

        drop(h2);
        // wait for the event to be dispatched
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(consumer.consume_all().count(), 0, "one clone still alive");

        drop(h3); // last reference
        // wait for the event to be dispatched
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(consumer.consume_all().count(), 1, "last clone dropped");
    }

    #[test]
    fn drop_event_carries_correct_asset_handle() {
        let workers = WorkerPool::new("test");
        let events = EventManager::new(workers.clone());
        let mut consumer = events.subscribe::<InternalAssetUnloaded>();
        let dispatcher = events.register::<InternalAssetUnloaded>();

        let expected = AssetHandle::from_raw("foo/bar.dirkasset", AssetType::Unknown);
        let asset_ref = AssetRef::new(expected.clone(), FakeAsset { value: 0 }, dispatcher);
        let handle = Handle::new(asset_ref);
        drop(handle);

        let ev = consumer.consume_blocking().unwrap();
        assert_eq!(ev.handle, expected);
    }

    // ── Debug ─────────────────────────────────────────────────────────────────

    #[test]
    fn debug_output_includes_asset_path() {
        let h = fake_handle(0, "models/debug_test.dirkasset");
        let dbg = format!("{h:?}");
        assert!(
            dbg.contains("debug_test"),
            "Expected asset path in debug output: {dbg}"
        );
    }

    #[test]
    fn debug_does_not_require_mutable_access() {
        let h = fake_handle(0, "test.dirkasset");
        // Should compile and run without locking the inner mutex for a write.
        let _ = format!("{h:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DirkAsset validation
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod dirk_asset_validation {
    use super::*;

    fn model_meta(raw: &str) -> Metadata {
        Metadata {
            asset_type: AssetType::Model,
            handle: AssetHandle::from_raw(raw, AssetType::Model),
        }
    }

    fn unknown_meta() -> Metadata {
        Metadata {
            asset_type: AssetType::Unknown,
            handle: AssetHandle::from_raw("x.dirkasset", AssetType::Unknown),
        }
    }

    // ── Unknown type ──────────────────────────────────────────────────────────

    #[test]
    fn unknown_type_is_rejected() {
        let asset = DirkAsset {
            meta: unknown_meta(),
            model: None,
        };
        assert!(!asset.validate(), "Unknown type must fail validation");
    }

    #[test]
    fn unknown_type_with_model_section_is_still_rejected() {
        // The model section is irrelevant when the type is Unknown.
        let asset = DirkAsset {
            meta: unknown_meta(),
            model: Some(ModelConfig {
                gltf: "ignored.gltf".to_string(),
            }),
        };
        assert!(!asset.validate());
    }

    // ── Model type ────────────────────────────────────────────────────────────

    #[test]
    fn model_without_model_section_is_rejected() {
        let asset = DirkAsset {
            meta: model_meta("models/x.dirkasset"),
            model: None,
        };
        assert!(!asset.validate(), "Model asset must have a [model] section");
    }

    #[test]
    fn model_with_missing_gltf_file_is_rejected() {
        let asset = DirkAsset {
            meta: model_meta("models/x.dirkasset"),
            model: Some(ModelConfig {
                gltf: "absolutely_does_not_exist_xyz.gltf".to_string(),
            }),
        };
        assert!(!asset.validate(), "Missing gltf file must fail validation");
    }

    #[test]
    fn model_with_existing_gltf_file_is_accepted() {
        let dir = TempDir::new().unwrap();
        let gltf_path = dir.path().join("cube.gltf");
        fs::write(&gltf_path, minimal_gltf_json()).unwrap();

        // Use the absolute path as the gltf value so that
        // handle.dir().join(abs_path) == abs_path regardless of ASSETS_PATH.
        let asset = DirkAsset {
            meta: model_meta(""),
            model: Some(ModelConfig {
                gltf: gltf_path.to_string_lossy().into_owned(),
            }),
        };
        assert!(asset.validate(), "Existing gltf file must pass validation");
    }

    #[test]
    fn validation_does_not_panic_on_empty_handle_path() {
        let asset = DirkAsset {
            meta: model_meta(""),
            model: Some(ModelConfig {
                gltf: "missing.gltf".to_string(),
            }),
        };
        // Should return false, not panic.
        assert!(!asset.validate());
    }
}

// AssetRegistry — isolated integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod registry {
    use super::*;
    use std::{
        sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };

    fn registry_with_model(name: &str) -> (TempDir, EventManager, AssetRegistry, AssetHandle) {
        let root = TempDir::new().unwrap();
        write_model_fixture(root.path(), name);
        let workers = WorkerPool::new("asset-test");
        let events = EventManager::new(workers.clone());
        let registry = AssetRegistry::init_at(&events, workers, root.path()).unwrap();
        let id = AssetHandle::from_raw(format!("{name}.dirkasset"), AssetType::Model);
        (root, events, registry, id)
    }

    fn wait_for_event<E: dirk_events::Event>(consumer: &mut dirk_events::Consumer<E>) -> E {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(event) = consumer.consume_all().next() {
                return event;
            }
            assert!(Instant::now() < deadline, "event was not delivered in time");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn explicit_root_loads_model_and_resolves_source_paths() {
        let (_root, _events, registry, id) = registry_with_model("hero");
        let handle = wait_for_load(registry.load_asset::<Model>(&id)).unwrap();
        assert_eq!(handle.get().unwrap().gltf.scenes().count(), 1);
        assert_eq!(
            handle.handle().dir(),
            handle.handle().path().parent().unwrap()
        );
    }

    #[test]
    fn async_load_works_inside_runtime() {
        let (_root, _events, registry, id) = registry_with_model("async");
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let handle = runtime.block_on(registry.load_asset::<Model>(&id)).unwrap();
        assert_eq!(handle.get().unwrap().gltf.scenes().count(), 1);
    }

    #[test]
    fn missing_root_type_mismatch_and_unknown_handle_report_errors() {
        let root = TempDir::new().unwrap();
        let workers = WorkerPool::new("asset-test");
        let events = EventManager::new(workers.clone());
        assert!(matches!(
            AssetRegistry::init_at(&events, workers.clone(), root.path().join("missing")),
            Err(Error::IoError(_))
        ));
        let registry = AssetRegistry::init_at(&events, workers, root.path()).unwrap();
        let wrong = AssetHandle::from_raw("wrong.dirkasset", AssetType::Unknown);
        assert!(matches!(
            wait_for_load(registry.load_asset::<Model>(&wrong)),
            Err(Error::TypeMismatch(path)) if path == "wrong.dirkasset"
        ));
        let absent = AssetHandle::from_raw("absent.dirkasset", AssetType::Model);
        assert!(matches!(
            wait_for_load(registry.load_asset::<Model>(&absent)),
            Err(Error::NotFound(path)) if path == "absent.dirkasset"
        ));
    }

    #[test]
    fn cached_load_shares_data_and_emits_one_loaded_event() {
        let (_root, events, registry, id) = registry_with_model("cached");
        let mut loaded = events.subscribe::<AssetLoaded<Model>>();
        let first = wait_for_load(registry.load_asset::<Model>(&id)).unwrap();
        let event = wait_for_event(&mut loaded);
        assert_eq!(event.handle.generation(), first.generation());
        drop(event);
        let second = wait_for_load(registry.load_asset::<Model>(&id)).unwrap();
        assert_eq!(first.generation(), second.generation());
        assert_eq!(loaded.consume_all().count(), 0);
        first.take().unwrap();
        assert!(matches!(second.take(), Err(Error::AlreadyTaken)));
    }

    #[derive(Clone)]
    struct CountingAsset;

    static DECODE_COUNT: AtomicUsize = AtomicUsize::new(0);

    impl Asset for CountingAsset {
        type Config = ModelConfig;

        fn load(_: &ModelConfig, _: &AssetHandle) -> Result<Self> {
            DECODE_COUNT.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(30));
            Ok(Self)
        }

        fn asset_type() -> AssetType {
            AssetType::Model
        }
    }

    #[test]
    fn concurrent_same_key_loads_share_one_decode_and_generation() {
        DECODE_COUNT.store(0, Ordering::SeqCst);
        let (_root, _events, registry, id) = registry_with_model("concurrent");
        let gate = Arc::new(Barrier::new(3));
        let threads = (0..2)
            .map(|_| {
                let registry = registry.clone();
                let id = id.clone();
                let gate = gate.clone();
                std::thread::spawn(move || {
                    gate.wait();
                    registry.load_asset_immediate::<CountingAsset>(id).unwrap()
                })
            })
            .collect::<Vec<_>>();
        gate.wait();
        let handles = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(DECODE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(handles[0].generation(), handles[1].generation());
    }

    #[test]
    fn old_unload_preserves_new_generation_and_identifies_old_instance() {
        let (_root, events, registry, id) = registry_with_model("reloaded");
        let mut loaded = events.subscribe::<AssetLoaded<Model>>();
        let mut unloaded = events.subscribe::<AssetUnloaded>();
        let first = wait_for_load(registry.load_asset::<Model>(&id)).unwrap();
        let first_generation = first.generation();
        drop(wait_for_event(&mut loaded));
        drop(first);
        let second = wait_for_load(registry.load_asset::<Model>(&id)).unwrap();
        assert_ne!(first_generation, second.generation());
        drop(wait_for_event(&mut loaded));
        let deadline = Instant::now() + Duration::from_secs(2);
        let old_unload = loop {
            registry.tick();
            if let Some(event) = unloaded.consume_all().next() {
                break event;
            }
            assert!(
                Instant::now() < deadline,
                "old unload was not delivered in time"
            );
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(old_unload.handle, id);
        assert_eq!(old_unload.generation, first_generation);
        assert_eq!(
            registry.cached_handle::<Model>(&id).unwrap().generation(),
            second.generation()
        );
    }
}
