//! Entrypoint for exercising the new plugin/subsystem engine runtime.

#[cfg(feature = "cli")]
use dirkengine::cli::run;

#[cfg(not(feature = "cli"))]
fn run() -> anyhow::Result<()> {
    use anyhow::Context;

    println!(
        "you are running the base cli. to run the advanced cli, enable the \"cli\" cargo feature"
    );
    let mut builder = dirk_engine::Engine::builder();
    builder.with_os_signals(true);
    builder.with_plugin(dirkengine::DefaultPlugins)?;
    builder.with_plugin(dirkengine::demo::DemoPlugin)?;
    let engine = builder.build().context("build new engine")?;

    engine.run().context("run new engine")?;
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            tracing::error!("{err:#}");
            panic!("Error: {err:#}");
        }
    }
}
