//! Metal 3 backend, for Apple targets.
//!
//! Synchronization relies on Metal's hazard tracking: every resource is
//! tracked, and render passes declare the heap's resources with
//! `useResources` (which also makes them resident), so Metal orders every
//! conflicting access on its own. Barriers therefore record nothing, and
//! queue ownership transfers have no Metal equivalent. Each queue is a
//! command queue with an `MTLSharedEvent` timeline; Metal's tracking stops at
//! the queue, so work on another queue is ordered by waiting for its
//! completion, as on Vulkan.

mod blit;
mod buffer;
mod command;
mod convert;
mod device;
mod heap;
mod image;
mod pass;
mod pipeline;
mod presentation;
mod queue;
mod sampler;

pub(crate) use buffer::{Buffer, BufferId};
pub(crate) use command::{CommandBuffer, CommandPool};
pub(crate) use device::Device;
pub(crate) use image::{Image, ImageId};
pub(crate) use pipeline::{GraphicsPipeline, PipelineId, ShaderModule};
pub(crate) use presentation::{Presentation, Semaphore, Surface, Swapchain};
pub(crate) use sampler::Sampler;

use std::ops::Deref;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSError;

use crate::{Error, Result};

/// A Metal object shared between threads, such as a resource or a pipeline.
///
/// Metal documents every object but command buffers and encoders as safe to
/// use from any thread; `objc2-metal` only marks some protocols `Send` and
/// `Sync`, so this wrapper does it for the others. Command buffers and
/// encoders are never wrapped.
pub(super) struct Object<P: ?Sized>(Retained<ProtocolObject<P>>);

// SAFETY: only thread-safe Metal objects are wrapped; see `Object`.
unsafe impl<P: ?Sized> Send for Object<P> {}
// SAFETY: as above.
unsafe impl<P: ?Sized> Sync for Object<P> {}

impl<P: ?Sized> Deref for Object<P> {
    type Target = ProtocolObject<P>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<P: ?Sized> Clone for Object<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: ?Sized> From<Retained<ProtocolObject<P>>> for Object<P> {
    fn from(object: Retained<ProtocolObject<P>>) -> Self {
        Self(object)
    }
}

device_children!(Buffer, Image, ShaderModule, GraphicsPipeline, Swapchain);

/// `value` as a `usize`, which is 64 bits wide on every Apple target.
#[expect(clippy::cast_possible_truncation, reason = "Apple targets are 64-bit")]
const fn usize(value: u64) -> usize {
    value as usize
}

/// Converts native results into RHI results that name the failed operation.
trait Check<T> {
    /// Maps a native failure of `operation` to an error.
    fn check(self, operation: &'static str) -> Result<T>;
}

/// Metal returns `nil` for objects it cannot create, without a reason.
impl<T> Check<T> for Option<T> {
    fn check(self, operation: &'static str) -> Result<T> {
        self.ok_or_else(|| Error::native(operation, "Metal returned no object"))
    }
}

impl<T> Check<T> for Result<T, Retained<NSError>> {
    fn check(self, operation: &'static str) -> Result<T> {
        self.map_err(|error| Error::native(operation, error.localizedDescription().to_string()))
    }
}
