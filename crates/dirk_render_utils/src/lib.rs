//! Rendering utilities built on the RHI.
//!
//! [`Uploads`] stage host data in frame memory and copy it into new
//! device-local buffers and images, handing them to graphics work from the
//! copy queue when the device has a dedicated one.

mod upload;

pub use upload::Uploads;
