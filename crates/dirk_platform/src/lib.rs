#![doc = include_str!("../README.md")]

use std::time::Duration;

use dirk_universe::Universe;
use tracing::info;
use winit::event_loop::{
    EventLoop,
    pump_events::{EventLoopExtPumpEvents, PumpStatus},
};

mod errors;
mod event;
mod handler;
mod window;

pub use winit::{
    cursor::CursorIcon,
    keyboard::ModifiersState,
    window::{Theme, WindowId},
};
pub use {
    errors::Error,
    event::*,
    window::{ImeArea, MainWindow, PlatformWindows, Window, WindowSurfaceTarget, Windows},
};

use errors::Result;
use handler::PlatformHandler;

/// Registers the platform layer as an engine subsystem.
pub struct PlatformPlugin;

impl dirk_engine::EnginePlugin for PlatformPlugin {
    fn name(&self) -> &'static str {
        "platform"
    }

    fn build(&self, builder: &mut dirk_engine::EngineBuilder) -> anyhow::Result<()> {
        builder.add_subsystem(|ctx| {
            let platform = Platform::init(ctx.events())?;
            ctx.add_resource(platform.platform_windows())?;
            Ok(platform)
        });
        Ok(())
    }
}

/// The main Platform struct that is initialized by the engine.
/// This is an engine [`Subsystem`].
///
/// [`Subsystem`]: dirk_engine::Subsystem
struct Platform {
    handler: PlatformHandler,
    event_loop: EventLoop,
    windows: PlatformWindows,

    window_consumer: dirk_events::Consumer<WindowEvent>,
}

impl Platform {
    /// Initialise the platform wrapper. Essentially just creates
    /// the handler for [winit] platform events.
    ///
    /// # Errors
    ///
    /// Returns an error if the [winit] App exited while trying
    /// to start it.
    fn init(events: &dirk_events::EventManager) -> Result<Self> {
        let windows = PlatformWindows::default();
        let mut platform = Self {
            handler: PlatformHandler::new(events, windows.clone()),
            event_loop: EventLoop::new()?,
            windows,
            window_consumer: events.subscribe(),
        };

        // Pump until `can_create_surfaces` creates the main window or fails.
        // Each call returns quickly; the OS dispatches the startup events
        // within the first few iterations.
        loop {
            let status = platform
                .event_loop
                .pump_app_events(Some(Duration::ZERO), &mut platform.handler);
            if let Some(result) = platform.handler.take_initialization() {
                result.map_err(Error::WindowCreation)?;
                break;
            }
            match status {
                PumpStatus::Exit(code) => return Err(Error::AppExited(code)),
                PumpStatus::Continue => {}
            }
        }

        info!("initialized platform");
        Ok(platform)
    }

    /// Returns the shared platform window resource.
    #[must_use]
    pub fn platform_windows(&self) -> PlatformWindows {
        self.windows.clone()
    }
}

impl dirk_engine::Subsystem for Platform {
    fn name(&self) -> &'static str {
        "platform"
    }

    fn tick(
        &mut self,
        _delta_time: f64,
        handle: &dirk_engine::EngineHandle,
        _universe: &Universe,
    ) -> anyhow::Result<()> {
        match self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut self.handler)
        {
            PumpStatus::Exit(code) => {
                // Treat a forced OS exit like a window close.
                info!("event loop exited with code {code}");
                handle.exit();
                return Ok(());
            }
            PumpStatus::Continue => {}
        }

        if self.windows.is_empty() {
            info!("all platform windows have been closed");
            handle.exit();
            return Ok(());
        }

        self.window_consumer.consume_all().for_each(|event| {
            self.windows.handle_window_event(&event);
        });

        Ok(())
    }
}

impl Drop for Platform {
    fn drop(&mut self) {
        info!("Shutting down platform");

        // AppKit sends window-destruction callbacks while closing native
        // windows. Ask winit to close them while its handler is still
        // registered, then release the Rust window wrappers.
        #[cfg(target_os = "macos")]
        {
            self.handler.request_shutdown();
            self.event_loop
                .pump_app_events(Some(Duration::ZERO), &mut self.handler);
        }

        self.handler.shutdown();
        #[cfg(not(target_os = "macos"))]
        {
            // On other platforms, process the window destruction events
            // before tearing down the event loop.
            self.event_loop
                .pump_app_events(Some(Duration::ZERO), &mut self.handler);
        }
    }
}
