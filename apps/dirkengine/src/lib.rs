#![doc = include_str!("../../../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[doc(inline)]
pub use dirk_assets as assets;
#[cfg(feature = "editor")]
#[doc(inline)]
pub use dirk_editor as editor;
#[doc(inline)]
pub use dirk_engine as engine;
#[doc(inline)]
pub use dirk_events as events;
#[doc(inline)]
pub use dirk_input as input;
#[doc(inline)]
pub use dirk_platform as platform;
#[doc(inline)]
pub use dirk_player as player;
#[doc(inline)]
pub use dirk_renderer as renderer;
#[doc(inline)]
pub use dirk_threads as threads;
#[doc(inline)]
pub use dirk_universe as universe;
#[doc(inline)]
pub use dirk_utils as utils;
#[doc(inline)]
pub use dirk_world as world;

pub mod demo;

#[cfg(feature = "cli")]
pub mod cli;

/// Registers all the engine's default plugins.
pub struct DefaultPlugins;

impl dirk_engine::EnginePlugin for DefaultPlugins {
    fn name(&self) -> &'static str {
        "default_plugins"
    }

    fn build(&self, builder: &mut dirk_engine::EngineBuilder) -> anyhow::Result<()> {
        #[cfg(feature = "editor")]
        builder.with_plugin(editor::EditorPlugin)?;
        builder.with_plugin(assets::AssetsPlugin)?;
        builder.with_plugin(platform::PlatformPlugin)?;
        builder.with_plugin(player::PlayerPlugin)?;
        builder.with_plugin(world::WorldPlugin)?;
        builder.with_plugin(renderer::RendererPlugin)?;
        Ok(())
    }
}
