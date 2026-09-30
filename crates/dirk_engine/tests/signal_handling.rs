//! Subprocess coverage for Unix signal shutdown behavior.

use std::{
    io::{BufRead, BufReader},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use anyhow::Context as _;
use dirk_engine::{Engine, EngineBuilder, EngineHandle, EnginePlugin, Subsystem};

const CHILD_ENV: &str = "DIRK_ENGINE_SIGNAL_TEST_CHILD";
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const CHILD_READY: &str = "DIRK_ENGINE_SIGNAL_TEST_CHILD_READY";
const CHILD_SHUTTING_DOWN: &str = "DIRK_ENGINE_SIGNAL_TEST_CHILD_SHUTTING_DOWN";

#[test]
fn default_engine_does_not_change_host_signal_disposition() -> anyhow::Result<()> {
    use std::os::unix::process::ExitStatusExt;

    let mut child = SignalTestChild::spawn("signal_test_child_drops_default_engine")?;

    // Only signal once the engine has been created and dropped; otherwise the
    // test would pass without exercising the engine at all.
    child.wait_for_message(CHILD_READY)?;
    child.send_signal("TERM")?;
    let status = child.wait_for_exit()?;
    assert_eq!(status.signal(), Some(SIGTERM));
    Ok(())
}

#[test]
fn signal_test_child_drops_default_engine() -> anyhow::Result<()> {
    if std::env::var_os(CHILD_ENV).is_none() {
        return Ok(());
    }

    drop(Engine::new()?);
    println!("{CHILD_READY}");
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

    fn start(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        println!("{CHILD_READY}");
        Ok(())
    }

    fn shutdown(&mut self, _handle: &EngineHandle) -> anyhow::Result<()> {
        println!("{CHILD_SHUTTING_DOWN}");
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }
}

#[test]
fn second_sigint_terminates_process_with_default_signal_handler() -> anyhow::Result<()> {
    let mut child = SignalTestChild::spawn("signal_test_child_blocks_during_shutdown")?;

    child.wait_for_message(CHILD_READY)?;
    child.send_signal("INT")?;

    // Only interrupt again once the first signal has reached graceful shutdown.
    child.wait_for_message(CHILD_SHUTTING_DOWN)?;

    child.send_signal("INT")?;
    let status = child.wait_for_exit()?;

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

struct SignalTestChild {
    process: Child,
    output: mpsc::Receiver<std::io::Result<String>>,
}

impl SignalTestChild {
    fn spawn(test: &str) -> anyhow::Result<Self> {
        let process = Command::new(std::env::current_exe()?)
            .env(CHILD_ENV, "1")
            .args(["--exact", test, "--nocapture"])
            .stdout(Stdio::piped())
            .spawn()?;
        let (sender, output) = mpsc::channel();
        let mut child = Self { process, output };
        let stdout = child
            .process
            .stdout
            .take()
            .context("child stdout was not piped")?;
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(child)
    }

    fn send_signal(&self, signal: &str) -> anyhow::Result<()> {
        let status = Command::new("kill")
            .arg(format!("-{signal}"))
            .arg(self.process.id().to_string())
            .status()?;
        anyhow::ensure!(status.success(), "kill command failed with {status:?}");
        Ok(())
    }

    fn wait_for_message(&self, expected: &str) -> anyhow::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let line = self
                .output
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .with_context(|| format!("waiting for child message `{expected}`"))?
                .context("reading child stdout")?;
            if line == expected {
                return Ok(());
            }
        }
    }

    fn wait_for_exit(&mut self) -> anyhow::Result<ExitStatus> {
        let timeout = Duration::from_secs(5);
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.process.try_wait()? {
                return Ok(status);
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "child did not exit within {timeout:?}"
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for SignalTestChild {
    fn drop(&mut self) {
        // Reap the subprocess even when an assertion or readiness check fails.
        if !matches!(self.process.try_wait(), Ok(Some(_))) {
            let _ = self.process.kill();
            let _ = self.process.wait();
        }
    }
}

fn status_was_sigint(status: ExitStatus) -> bool {
    use std::os::unix::process::ExitStatusExt;

    status.signal() == Some(SIGINT) || status.code() == Some(128 + SIGINT)
}
