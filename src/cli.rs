//! This crate contains the CLI entrypoint & management for the enhanced clap
//! engine CLI.

use anyhow::Context;
use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(name = "dirkengine")]
#[command(about = "DirkEngine CLI")]
struct Cli {
    /// Increase logging verbosity. Use -v for debug and -vv for trace.
    #[arg(short = 'v', long = "verbose", global = true, action = ArgAction::Count)]
    verbose: u8,

    /// Disable the demo world.
    #[arg(long = "no-demo", global = true)]
    no_demo: bool,
}

/// Runs the enhanced [`clap`] CLI.
///
/// # Errors
///
/// Returns any error that occurs while running the engine.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut builder = dirk_engine::Engine::builder();
    builder.with_os_signals(true);

    match cli.verbose {
        0 => {}
        1 => {
            builder.with_log_level(piquel_log::LogLevel::Debug);
        }
        _ => {
            builder.with_log_level(piquel_log::LogLevel::Trace);
        }
    }

    #[cfg(feature = "editor")]
    builder.with_plugin(dirk_editor::EditorPlugin)?;
    builder.with_plugin(dirk_assets::AssetsPlugin)?;
    builder.with_plugin(dirk_platform::PlatformPlugin)?;
    builder.with_plugin(dirk_player::PlayerPlugin)?;
    builder.with_plugin(dirk_world::WorldPlugin)?;
    builder.with_plugin(dirk_renderer::RendererPlugin)?;

    if !cli.no_demo {
        builder.with_plugin(crate::demo::DemoPlugin)?;
    }

    let engine = builder.build().context("build new engine")?;

    engine.run().context("run new engine")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbosity_counts_repeated_flags() {
        assert_eq!(Cli::parse_from(["dirkengine"]).verbose, 0);
        assert_eq!(Cli::parse_from(["dirkengine", "-v"]).verbose, 1);
        assert_eq!(Cli::parse_from(["dirkengine", "-vv"]).verbose, 2);
        assert_eq!(Cli::parse_from(["dirkengine", "-v", "-v"]).verbose, 2);
    }
}
