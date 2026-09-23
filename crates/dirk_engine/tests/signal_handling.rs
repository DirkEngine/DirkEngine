#![cfg(unix)]
//! Subprocess coverage for Unix signal shutdown behavior.

use std::{
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};

use dirk_engine::{Engine, EngineBuilder, EngineHandle, EnginePlugin, Subsystem};

const CHILD_ENV: &str = "DIRK_ENGINE_SIGNAL_TEST_CHILD";
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;

#[test]
fn default_engine_does_not_change_host_signal_disposition() -> anyhow::Result<()> {
    let child = Command::new(std::env::current_exe()?)
        .env(CHILD_ENV, "1")
        .arg("--exact")
        .arg("signal_test_child_drops_default_engine")
        .spawn()?;

    thread::sleep(Duration::from_secs(1));
    send_signal(&child, "TERM")?;
    let status = wait_for_exit(child, Duration::from_secs(5))?;
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(SIGTERM));
    Ok(())
}

#[test]
fn signal_test_child_drops_default_engine() -> anyhow::Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }

    drop(Engine::new()?);
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

struct BlockingShutdownPlugin;

impl EnginePlugin for BlockingShutdownPlugin {
    fn name(&self) -> &'static str {
        "blocking-shutdown"
    }

    fn build(&self, builder: &mut EngineBuilder) -> anyhow::Result<()> {
        builder.add_subsystem(|_ctx| Ok(BlockingShutdownSubsystem));
        Ok(())
    }
}

struct BlockingShutdownSubsystem;

impl Subsystem for BlockingShutdownSubsystem {
    fn name(&self) -> &'static str {
        "blocking-shutdown"
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }
}

#[test]
fn second_sigint_terminates_process_with_default_signal_handler() -> anyhow::Result<()> {
    let mut child = Command::new(std::env::current_exe()?)
        .env(CHILD_ENV, "1")
        .arg("--exact")
        .arg("signal_test_child_blocks_during_shutdown")
        .arg("--nocapture")
        .spawn()?;

    thread::sleep(Duration::from_secs(1));
    send_signal(&child, "INT")?;

    thread::sleep(Duration::from_millis(500));
    assert!(
        child.try_wait()?.is_none(),
        "child exited after first SIGINT before blocking shutdown"
    );

    send_signal(&child, "INT")?;
    let status = wait_for_exit(child, Duration::from_secs(5))?;

    assert!(
        status_was_sigint(status),
        "expected child to terminate from SIGINT or exit 130, got {status:?}",
    );

    Ok(())
}

#[test]
fn signal_test_child_blocks_during_shutdown() -> anyhow::Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }

    let mut builder = Engine::builder();
    builder.with_os_signals(true);
    builder.with_log_level(piquel_log::LogLevel::Error);
    builder.with_plugin(BlockingShutdownPlugin)?;

    builder.build()?.run()?;
    Ok(())
}

fn send_signal(child: &Child, signal: &str) -> anyhow::Result<()> {
    let status = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(child.id().to_string())
        .status()?;

    anyhow::ensure!(status.success(), "kill command failed with {status:?}");
    Ok(())
}

fn wait_for_exit(mut child: Child, timeout: Duration) -> anyhow::Result<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("child did not exit within {timeout:?}");
        }

        thread::sleep(Duration::from_millis(50));
    }
}

fn status_was_sigint(status: ExitStatus) -> bool {
    use std::os::unix::process::ExitStatusExt;

    status.signal() == Some(SIGINT) || status.code() == Some(128 + SIGINT)
}
