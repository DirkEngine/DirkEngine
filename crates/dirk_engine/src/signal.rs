//! Operating system signal integration.
//!
//! The signal handler installed by this module does not touch engine state from
//! inside the OS signal context. `signal-hook` writes notifications into a pipe
//! and a background thread forwards them to the engine.
//!
//! The first handled signal requests a graceful engine shutdown. A second
//! handled signal re-raises it with the default disposition, so a stuck shutdown
//! can still be interrupted.
//!
//! This is scoped to Unix terminal and service-manager workflows, covering
//! `SIGINT`, `SIGTERM`, `SIGHUP`, and `SIGQUIT`. Game-window close events are
//! handled separately by platform/window integration. Signal handling is opt-in
//! because `signal-hook` cannot restore the host process's previous disposition
//! after an engine is dropped.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
};

use signal_hook::{
    consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM},
    iterator::{Handle as SignalIteratorHandle, Signals},
    low_level::{emulate_default_handler, signal_name},
};
use tracing::{debug, error, info, warn};

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

struct SignalListener {
    handle: SignalIteratorHandle,
    thread: Option<JoinHandle<()>>,
}

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
        if let Ok(signal) = self.receiver.try_recv() {
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

fn handled_signals() -> &'static [i32] {
    &[SIGINT, SIGTERM, SIGHUP, SIGQUIT]
}

fn handled_signal_names() -> Vec<&'static str> {
    handled_signals()
        .iter()
        .copied()
        .map(|signal| OperatingSystemSignal::new(signal).name())
        .collect()
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
