//! Operating system signal integration.
//!
//! The signal handler installed by this module does not touch engine state from
//! inside the OS signal context. On Unix, `signal-hook` writes notifications
//! into a pipe and a background thread forwards them to the engine. On Windows,
//! signal-safe atomic flags are polled during the normal tick flow.
//!
//! This is intentionally scoped to terminal and service-manager workflows. On
//! Unix-like systems this covers common termination signals such as `SIGINT`,
//! `SIGTERM`, `SIGHUP`, and `SIGQUIT`. On Windows, `signal-hook` is limited to
//! CRT signal emulation, so this covers `SIGINT` and `SIGBREAK` only. Console
//! close, logoff, shutdown events, and normal game-window close events are
//! handled elsewhere by platform/window integration. Signal handling is opt-in
//! because `signal-hook` cannot restore the host process's previous disposition
//! after an engine is dropped.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
};

#[cfg(not(windows))]
use std::thread::{self, JoinHandle};

#[cfg(not(windows))]
use signal_hook::iterator::{Handle as SignalIteratorHandle, Signals};
#[cfg(not(windows))]
use signal_hook::low_level::emulate_default_handler;
use signal_hook::low_level::signal_name;
#[cfg(windows)]
use signal_hook::{SigId, flag, low_level::unregister};
use tracing::{debug, error, info, warn};

#[cfg(windows)]
use signal_hook::consts::signal::{SIGBREAK, SIGINT};
#[cfg(not(windows))]
use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OperatingSystemSignal {
    number: i32,
}

impl OperatingSystemSignal {
    fn new(number: i32) -> Self {
        Self { number }
    }

    fn name(self) -> &'static str {
        signal_name(self.number).unwrap_or("unknown signal")
    }
}

#[cfg(not(windows))]
struct SignalListener {
    handle: SignalIteratorHandle,
    thread: Option<JoinHandle<()>>,
}

#[cfg(not(windows))]
impl SignalListener {
    fn install(sender: mpsc::Sender<OperatingSystemSignal>) -> anyhow::Result<Self> {
        let mut signals = Signals::new(handled_signals())?;
        let handle = signals.handle();
        let shutdown_requested = AtomicBool::new(false);
        let thread = thread::Builder::new()
            .name("dirk-os-signals".to_owned())
            .spawn(move || {
                for signal in signals.forever() {
                    if shutdown_requested.swap(true, Ordering::SeqCst) {
                        warn!(
                            signal,
                            name = OperatingSystemSignal::new(signal).name(),
                            "additional operating system signal received; using default handler",
                        );
                        if let Err(err) = emulate_default_handler(signal) {
                            error!(
                                signal,
                                error = %err,
                                "failed to emulate default signal handler; exiting process",
                            );
                        }
                        std::process::exit(128 + signal);
                    }

                    if sender.send(OperatingSystemSignal::new(signal)).is_err() {
                        break;
                    }
                }
            })?;

        Ok(Self {
            handle,
            thread: Some(thread),
        })
    }

    fn shutdown(&mut self) {
        self.handle.close();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            error!("operating system signal listener thread panicked");
        }
    }
}

#[cfg(not(windows))]
impl Drop for SignalListener {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(windows)]
struct SignalListener {
    registrations: Vec<(i32, std::sync::Arc<AtomicBool>, SigId)>,
}

#[cfg(windows)]
impl SignalListener {
    fn install(_sender: mpsc::Sender<OperatingSystemSignal>) -> anyhow::Result<Self> {
        let mut listener = Self {
            registrations: Vec::new(),
        };
        for &signal in handled_signals() {
            let requested = std::sync::Arc::new(AtomicBool::new(false));
            let id = flag::register(signal, std::sync::Arc::clone(&requested))?;
            listener.registrations.push((signal, requested, id));
        }
        Ok(listener)
    }

    fn received_signal(&self) -> Option<OperatingSystemSignal> {
        self.registrations
            .iter()
            .find(|(_, requested, _)| requested.swap(false, Ordering::SeqCst))
            .map(|(signal, _, _)| OperatingSystemSignal::new(*signal))
    }

    fn shutdown(&mut self) {
        for (_, _, id) in self.registrations.drain(..) {
            unregister(id);
        }
    }
}

#[cfg(windows)]
impl Drop for SignalListener {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub(crate) struct OperatingSystemSignals {
    receiver: Receiver<OperatingSystemSignal>,
    listener: Option<SignalListener>,
}

impl OperatingSystemSignals {
    pub(crate) fn install() -> anyhow::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let listener = SignalListener::install(sender)?;

        info!(
            signals = ?handled_signal_names(),
            "installed operating system signal handlers",
        );
        Ok(Self {
            receiver,
            listener: Some(listener),
        })
    }

    pub(crate) fn disabled() -> Self {
        let (_sender, receiver) = mpsc::channel();
        Self {
            receiver,
            listener: None,
        }
    }

    #[cfg(test)]
    fn from_receiver(receiver: Receiver<OperatingSystemSignal>) -> Self {
        Self {
            receiver,
            listener: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn empty_for_tests() -> Self {
        Self::disabled()
    }

    #[cfg(test)]
    pub(crate) fn with_signal_for_tests(signal: i32) -> anyhow::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        sender.send(OperatingSystemSignal::new(signal))?;
        Ok(Self::from_receiver(receiver))
    }

    pub(crate) fn exit_requested(&mut self) -> bool {
        let received = self.receiver.try_recv().ok();
        #[cfg(windows)]
        let received = received.or_else(|| self.listener.as_ref()?.received_signal());

        if let Some(signal) = received {
            warn!(
                signal = signal.number,
                name = signal.name(),
                "operating system signal received; requesting engine shutdown",
            );
            return true;
        }

        false
    }

    pub(crate) fn shutdown(&mut self) {
        if let Some(mut listener) = self.listener.take() {
            listener.shutdown();
            debug!("shut down operating system signal listener");
        }
    }
}

#[cfg(not(windows))]
fn handled_signals() -> &'static [i32] {
    &[SIGINT, SIGTERM, SIGHUP, SIGQUIT]
}

#[cfg(windows)]
fn handled_signals() -> &'static [i32] {
    &[SIGINT, SIGBREAK]
}

fn handled_signal_names() -> Vec<&'static str> {
    handled_signals()
        .iter()
        .copied()
        .map(signal_display_name)
        .collect()
}

fn signal_display_name(signal: i32) -> &'static str {
    if let Some(name) = signal_name(signal) {
        return name;
    }

    #[cfg(windows)]
    if signal == SIGBREAK {
        return "SIGBREAK";
    }

    "unknown signal"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn received_signal_requests_exit() -> anyhow::Result<()> {
        let mut signals = OperatingSystemSignals::with_signal_for_tests(SIGINT)?;

        assert!(signals.exit_requested());
        Ok(())
    }

    #[test]
    fn startup_signal_names_are_human_readable() {
        let signal_names = handled_signal_names();

        assert!(signal_names.contains(&"SIGINT"));
        assert!(!signal_names.contains(&"unknown signal"));
    }

    #[test]
    fn abort_signal_is_not_handled() {
        assert!(!handled_signals().contains(&signal_hook::consts::SIGABRT));
    }
}
