//! Presents to a window: clears it to a changing color for a few seconds,
//! resizing it halfway to recreate the swapchain, with validation enabled.
//! Presentation needs a display, so this is how to check it by hand:
//!
//! ```sh
//! cargo run -p dirk_rhi --example present
//! ```
//!
//! Exits with an error if anything fails or validation reports an error.

use std::{cell::RefCell, rc::Rc, sync::Arc};

use dirk_rhi::{
    ColorAttachment, Error, Format, Frame, Graphics, ImageAccess, ImageBarrier, ImageRange,
    ImageUsage, LoadOp, PresentMode, RenderPassDesc, Result, Rhi, RhiCreateInfo, StoreOp,
    Subresource, Swapchain, SwapchainDesc,
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

/// The window, shared with its surface like `dirk_platform`'s.
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

/// What the example renders with. Fields drop in order: the swapchain,
/// then the RHI, which waits for the GPU, then the window.
struct Presenting {
    swapchain: Swapchain,
    rhi: Rhi,
    window: Arc<Window>,
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
            let attributes = WindowAttributes::default()
                .with_title("dirk_rhi presentation")
                .with_surface_size(PhysicalSize::new(640, 480));
            match event_loop.create_window(attributes) {
                Ok(window) => self.run(event_loop, |app| app.start(window)),
                Err(error) => self.fail(event_loop, &error),
            }
        }
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.finish(event_loop),
            WindowEvent::SurfaceResized(size) => {
                if let Some(presenting) = &mut self.presenting {
                    presenting.swapchain.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.run(event_loop, App::draw),
            _ => {}
        }
    }
}

impl App {
    /// Creates the RHI, surface and swapchain for `window`.
    fn start(&mut self, window: Box<dyn winit::window::Window>) -> Result<()> {
        let window = Arc::new(Window(window));
        let size = window.0.surface_size();
        let display = window.display_handle().map_err(|error| Error::Native {
            operation: "getting the display",
            source: error.into(),
        })?;
        let rhi = Rhi::new(&RhiCreateInfo {
            application_name: "dirk_rhi presentation",
            validation: true,
            display: Some(display),
        })?;
        let surface = rhi.create_surface(window.clone())?;
        let swapchain = rhi.create_swapchain(
            surface,
            &SwapchainDesc {
                label: "window",
                extent: (size.width, size.height),
                formats: &[Format::Bgra8Srgb, Format::Rgba8Srgb],
                present_mode: PresentMode::Fifo,
                usage: ImageUsage::COLOR_ATTACHMENT,
            },
        )?;
        window.0.request_redraw();
        self.presenting = Some(Presenting {
            swapchain,
            rhi,
            window,
            frames: 0,
        });
        Ok(())
    }

    /// Presents one frame, resizing halfway, and asks for the next.
    fn draw(&mut self) -> Result<()> {
        let Some(presenting) = &mut self.presenting else {
            return Ok(());
        };
        presenting.frames += 1;
        if presenting.frames == FRAMES / 2 {
            // Applied now, later with a resize event, or not at all.
            let requested = PhysicalSize::new(320, 240).into();
            if let Some(size) = presenting.window.0.request_surface_size(requested) {
                presenting.swapchain.resize(size.width, size.height);
            }
        }
        let frame = presenting.rhi.begin_frame()?;
        let shade = f32::from(u16::try_from(presenting.frames % 120).unwrap_or(0)) / 120.0;
        match Self::clear(
            &frame,
            &mut presenting.swapchain,
            [shade, 0.2, 1.0 - shade, 1.0],
        ) {
            // Minimized: nothing to present to this frame.
            Err(Error::SwapchainOutOfDate) => {}
            result => result?,
        }
        frame.finish();
        presenting.window.0.request_redraw();
        Ok(())
    }

    /// Clears the next image of `swapchain` to `color` and presents it.
    fn clear(frame: &Frame<'_>, swapchain: &mut Swapchain, color: [f32; 4]) -> Result<()> {
        let acquired = frame.acquire(swapchain)?;
        let mut encoder = frame.encoder::<Graphics>("clear")?;
        let barrier = |before, after| ImageBarrier {
            image: acquired.image(),
            range: ImageRange::ALL,
            before,
            after,
            transfer: None,
        };
        // SAFETY: the image was just acquired, and is cleared in between.
        unsafe {
            encoder.barrier(
                &[],
                &[barrier(ImageAccess::Present, ImageAccess::ColorAttachment)],
            )?;
        }
        encoder
            .render_pass(&RenderPassDesc {
                label: "clear",
                colors: &[ColorAttachment {
                    image: acquired.image(),
                    subresource: Subresource::default(),
                    load: LoadOp::Clear(color),
                    store: StoreOp::Store,
                    resolve: None,
                }],
                depth: None,
            })?
            .end();
        // SAFETY: the pass cleared it.
        unsafe {
            encoder.barrier(
                &[],
                &[barrier(ImageAccess::ColorAttachment, ImageAccess::Present)],
            )?;
        }
        let presentable = frame.submit_with(encoder, acquired)?;
        frame.present(presentable)
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
