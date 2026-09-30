# assets

Asset loading, lifetime management, and registry.

This crate is the central hub for all on-disk assets consumed by the
engine. It provides:

- **[`AssetHandle`]** — a lightweight identifier (path + type tag) that
  locates an asset inside the asset tree.
- **[`Handle<T>`]** — a ref-counted, typed wrapper around loaded asset
  data; the last [drop] triggers an [`AssetUnloaded`] event.
- **[`AssetRegistry`]** — scans the asset directory at startup, validates
  every `.dirkasset` descriptor, and vends [`Handle<T>`]s on demand.
- **[`AssetsPlugin`]** — installs the registry as an engine subsystem and
  publishes an `AssetRegistry` resource for other subsystem factories.

# Asset file format

Every asset on disk is described by a JSON file with the `.dirkasset`
extension. The file lives alongside (or near) the source data it
references. Example for a model:

```json
{
  "meta": { "asset_type": "Model" },
  "model": { "gltf": "hero.gltf" }
}
```

The `meta.asset_type` field determines which config section is used. Each
asset type has its own optional config object (currently only `"model"`).

# Engine integration

Most applications should register `AssetsPlugin` with `EngineBuilder`. The
plugin initializes `AssetRegistry`, publishes it as a resource, and ticks it
once per engine frame.

```rust
# fn main() -> anyhow::Result<()> {
let mut builder = dirk_engine::Engine::builder();
builder.with_plugin(dirk_assets::AssetsPlugin)?;
# Ok(()) }
```

# Manual registry usage

```rust
use dirk_assets::{AssetRegistry, AssetHandle, AssetType, AssetLoaded, AssetUnloaded, Model};
use dirk_events::EventManager;
use dirk_threads::WorkerPool;

# fn test() -> anyhow::Result<()> {
// 1. Initialise — scans ASSETS_PATH and validates all .dirkasset files.
let workers = WorkerPool::new("test");
let events = EventManager::new(workers.clone());
let registry = AssetRegistry::init(&events, workers)?;

// 2. Load an asset by its handle string (path relative to ASSETS_PATH).
let mut load = registry.load_asset::<Model>(
    &AssetHandle::from_raw("models/hero.dirkasset", AssetType::Model));

// 3. Subscribe to receive load events for future loads.
let mut loaded_consumer = events.subscribe::<AssetLoaded<Model>>();
let mut unloaded_consumer = events.subscribe::<AssetUnloaded>();

// 4. Game loop — call once per frame.
loop {
    registry.tick();

    if let Some(result) = load.try_poll() {
        let _handle = result?;
    }

    for event in loaded_consumer.consume_all() {
        let data = event.handle.take()?;
        // ... upload data to GPU
    }
    for AssetUnloaded { handle, generation } in unloaded_consumer.consume_all() {
        // ... remove data from the GPU only if this generation is still uploaded
    }
}
# Ok(()) }
```
