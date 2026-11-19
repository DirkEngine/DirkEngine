//! Rendering utilities built on the RHI.
//!
//! [`RenderGraph`] orders a frame's passes: it derives every barrier
//! between them from what each pass declares it accesses, and creates their
//! transient images and buffers, reused across frames through
//! [`Transients`].
//!
//! [`Uploads`] stage host data in frame memory and copy it into new
//! device-local buffers and images, handing them to graphics work from the
//! copy queue when the device has a dedicated one.

mod graph;
mod upload;

pub use graph::{Commands, Handle, PassBuilder, PassContext, RenderGraph, Rendering, Transients};
pub use upload::Uploads;
