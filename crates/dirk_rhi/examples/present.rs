//! Presents to two windows: clears each to a changing color for a few
//! seconds, resizing the first halfway to recreate its swapchain, with
//! validation enabled. Each window is cleared by its own submission, and
//! both are presented at once. Presentation needs a display, so this is how
//! to check it by hand:
//!
//! ```sh
//! cargo run -p dirk_rhi --example present
//! ```
//!
//! Exits with an error if anything fails or validation reports an error.

use std::{cell::RefCell, rc::Rc, sync::Arc};

use dirk_rhi::{
    ColorAttachment, Error, Format, Frame, Graphics, ImageUsage, LoadOp, PresentMode,
    RenderPassDesc, Result, Rhi, RhiCreateInfo, StoreOp, Subresource, Swapchain, SwapchainDesc,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
    },
    window::{WindowAttributes, WindowId},
};

/// Frames presented before exiting.
const FRAMES: u32 = 240;

/// Windows presented to.
const WINDOWS: usize = 2;

/// A window, shared with its surface like `dirk_platform`'s.
struct Window(Box<dyn winit::window::Window>);

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.0.window_handle()
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.0.display_handle()
    }
}

/// What the example renders with. Fields drop in order: the swapchains,
/// then the RHI, which waits for the GPU, then the windows.
struct Presenting {
    swapchains: Vec<Swapchain>,
    rhi: Rhi,
    windows: Vec<Arc<Window>>,
    frames: u32,
}

/// The example, and the error that ended it, if any.
struct App {
    presenting: Option<Presenting>,
    outcome: Rc<RefCell<Result<(), String>>>,
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.presenting.is_none() {
            let windows = (0..WINDOWS)
                .map(|window| {
                    let attributes = WindowAttributes::default()
                        .with_title(format!("dirk_rhi presentation {window}"))
                        .with_surface_size(PhysicalSize::new(640, 480));
                    event_loop.create_window(attributes)
                })
                .collect::<Result<Vec<_>, _>>();
            match windows {
                Ok(windows) => self.run(event_loop, |app| app.start(windows)),
                Err(error) => self.fail(event_loop, &error),
            }
        }
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.finish(event_loop),
            WindowEvent::SurfaceResized(size) => {
                if let Some(presenting) = &mut self.presenting
                    && let Some(swapchain) = presenting.swapchain(id)
                {
                    swapchain.resize(size.width, size.height);
                }
            }
            // The first window paces the frames, which draw every window.
            WindowEvent::RedrawRequested
                if self
                    .presenting
                    .as_ref()
                    .is_some_and(|presenting| presenting.windows[0].0.id() == id) =>
            {
                self.run(event_loop, App::draw);
            }
            _ => {}
        }
    }
}

impl Presenting {
    /// The swapchain presenting to window `id`.
    fn swapchain(&mut self, id: WindowId) -> Option<&mut Swapchain> {
        let index = self.windows.iter().position(|window| window.0.id() == id)?;
        self.swapchains.get_mut(index)
    }
}

impl App {
    /// Creates the RHI, and a surface and swapchain for each of `windows`.
    fn start(&mut self, windows: Vec<Box<dyn winit::window::Window>>) -> Result<()> {
        let windows: Vec<_> = windows
            .into_iter()
            .map(|window| Arc::new(Window(window)))
            .collect();
        let display = windows[0].display_handle().map_err(|error| Error::Native {
            operation: "getting the display",
            source: error.into(),
        })?;
        let rhi = Rhi::new(&RhiCreateInfo {
            application_name: "dirk_rhi presentation",
            validation: true,
            display: Some(display),
        })?;
        let swapchains = windows
            .iter()
            .map(|window| {
                let size = window.0.surface_size();
                rhi.create_swapchain(
                    rhi.create_surface(window.clone())?,
                    &SwapchainDesc {
                        label: "window",
                        extent: (size.width, size.height),
                        formats: &[Format::Bgra8Srgb, Format::Rgba8Srgb],
                        present_mode: PresentMode::Fifo,
                        usage: ImageUsage::COLOR_ATTACHMENT,
                    },
                )
            })
            .collect::<Result<_>>()?;
        windows[0].0.request_redraw();
        self.presenting = Some(Presenting {
            swapchains,
            rhi,
            windows,
            frames: 0,
        });
        Ok(())
    }

    /// Presents one frame to every window, resizing the first halfway, and
    /// asks for the next.
    fn draw(&mut self) -> Result<()> {
        let Some(presenting) = &mut self.presenting else {
            return Ok(());
        };
        presenting.frames += 1;
        if presenting.frames == FRAMES / 2 {
            // Applied now, later with a resize event, or not at all.
            let requested = PhysicalSize::new(320, 240).into();
            if let Some(size) = presenting.windows[0].0.request_surface_size(requested) {
                presenting.swapchains[0].resize(size.width, size.height);
            }
        }
        let frame = presenting.rhi.begin_frame()?;
        let shade = f32::from(u16::try_from(presenting.frames % 120).unwrap_or(0)) / 120.0;
        Self::clear(&frame, &mut presenting.swapchains, shade)?;
        frame.finish();
        presenting.windows[0].0.request_redraw();
        Ok(())
    }

    /// Clears the next image of each of `swapchains` to a color of `shade`,
    /// each in its own submission, and presents them all at once.
    fn clear(frame: &Frame<'_>, swapchains: &mut [Swapchain], shade: f32) -> Result<()> {
        let mut acquired = Vec::new();
        for swapchain in swapchains {
            match frame.acquire(swapchain) {
                Ok(image) => acquired.push(image),
                // Minimized: nothing to present to this frame.
                Err(Error::SwapchainOutOfDate) => {}
                Err(error) => return Err(error),
            }
        }
        for (window, image) in acquired.iter().enumerate() {
            let color = if window == 0 {
                [shade, 0.2, 1.0 - shade, 1.0]
            } else {
                [1.0 - shade, shade, 0.2, 1.0]
            };
            // The submission waits for the acquisitions, and the pass moves
            // the image from presentation to the attachment and back.
            let mut encoder = frame.encoder::<Graphics>("clear")?;
            encoder
                .render_pass(&RenderPassDesc {
                    label: "clear",
                    colors: &[ColorAttachment {
                        image: image.image(),
                        subresource: Subresource::default(),
                        load: LoadOp::Clear(color),
                        store: StoreOp::Store,
                        resolve: None,
                    }],
                    depth: None,
                })?
                .end();
            frame.submit(encoder)?;
        }
        frame.present(acquired)
    }

    /// Runs `step`, ending the example on errors and once every frame is
    /// presented.
    fn run(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        step: impl FnOnce(&mut Self) -> Result<()>,
    ) {
        if let Err(error) = step(self) {
            self.fail(event_loop, &error);
        } else if self
            .presenting
            .as_ref()
            .is_some_and(|presenting| presenting.frames >= FRAMES)
        {
            self.finish(event_loop);
        }
    }

    /// Ends the example, failing if validation reported errors.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop) {
        if let Some(mut presenting) = self.presenting.take() {
            let waited = presenting.rhi.wait_idle();
            let errors = presenting.rhi.validation_error_count();
            *self.outcome.borrow_mut() = match waited {
                Err(error) => Err(error.to_string()),
                Ok(()) if errors > 0 => Err(format!("validation reported {errors} errors")),
                Ok(()) => Ok(()),
            };
        }
        event_loop.exit();
    }

    /// Ends the example with `error`.
    fn fail(&mut self, event_loop: &dyn ActiveEventLoop, error: &dyn std::error::Error) {
        *self.outcome.borrow_mut() = Err(error.to_string());
        self.presenting = None;
        event_loop.exit();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();
    let outcome = Rc::new(RefCell::new(Ok(())));
    EventLoop::new()?.run_app(App {
        presenting: None,
        outcome: outcome.clone(),
    })?;
    outcome.replace(Ok(())).map_err(Into::into)
}
