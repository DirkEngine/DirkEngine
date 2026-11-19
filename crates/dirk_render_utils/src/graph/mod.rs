//! The render graph: passes declare how they access images and buffers, and
//! the graph derives the barriers between them.

mod access;
mod pass;
mod resources;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicU64, Ordering};

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferDesc, CommandEncoder, Frame, Graphics, Image,
    ImageAccess, ImageBarrier, ImageDesc, ImageRange, Result, StoreOp, Subresource,
};

pub use pass::{Commands, PassBuilder, PassContext, Rendering};
pub use resources::{Handle, ImageSlice, Transients};

use access::Subresources;
use pass::Pass;
use resources::{Resources, Source, Table};

/// One frame's passes, in order, and the images and buffers they access.
///
/// Passes declare each access, and the graph derives every barrier from the
/// declarations: read after write, write after read and write after write,
/// per mip level and array layer. Passes may declare an [`ImageSlice`] of an
/// image instead of the whole of it, such as a cube face, a shadow cascade
/// or a mip, and render to one mip of one layer. Reads that can share a state, such as sampling in more
/// shader stages, share one barrier. Passes record through the RHI's safe
/// commands when the graph executes, all into one encoder, with one barrier
/// call before each pass.
///
/// Images and buffers are either imported, owned elsewhere and declared in
/// the access they are in before the graph and the one they are left in
/// after it, or transient: created by the graph with the usage its passes
/// need, and reused across frames through [`Transients`].
///
/// ```no_run
/// # use dirk_render_utils::{RenderGraph, Transients};
/// # use dirk_rhi::{
/// #     Extent3d, Format, Frame, Graphics, GraphicsPipeline, ImageAccess, ImageDesc,
/// #     ImageDimension, ImageUsage, LoadOp, Result, SampleCount, ShaderStages, Swapchain,
/// # };
/// # fn frame(
/// #     frame: &Frame<'_>,
/// #     swapchain: &mut Swapchain,
/// #     transients: &mut Transients,
/// #     scene: &GraphicsPipeline<[f32; 4]>,
/// # ) -> Result<()> {
/// let acquired = frame.acquire(swapchain)?;
/// let mut encoder = frame.encoder::<Graphics>("frame")?;
/// let mut graph = RenderGraph::new();
/// let target = graph.import_image(acquired.image(), ImageAccess::Present, ImageAccess::Present)?;
/// let depth = graph.create_image(&ImageDesc {
///     label: "depth",
///     dimension: ImageDimension::D2,
///     extent: acquired.image().info().extent,
///     format: Format::Depth32Float,
///     usage: ImageUsage::empty(),
///     mip_levels: 1,
///     array_layers: 1,
///     samples: SampleCount::X1,
/// });
/// graph
///     .pass("scene")
///     .color(target, LoadOp::Clear([0.0, 0.0, 0.0, 1.0]))
///     .depth(depth, LoadOp::Clear(1.0))
///     .record(|pass| pass.draw(scene, &[1.0; 4], 0..3, 0..1))?;
/// graph.execute(frame, &mut encoder, transients)?;
/// for presentable in frame.submit_with(encoder, [acquired])? {
///     frame.present(presentable)?;
/// }
/// # Ok(())
/// # }
/// ```
pub struct RenderGraph<'a> {
    images: Resources<'a, Image>,
    buffers: Resources<'a, Buffer>,
    passes: Vec<Pass<'a>>,
}

/// The images and buffers of a graph while it executes.
pub(crate) struct Objects<'t> {
    images: Table<'t, Image>,
    buffers: Table<'t, Buffer>,
}

impl Default for RenderGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RenderGraph<'a> {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        // Distinguishes each graph's handles.
        static GRAPHS: AtomicU64 = AtomicU64::new(0);
        let graph = GRAPHS.fetch_add(1, Ordering::Relaxed);
        Self {
            images: Resources::new(graph),
            buffers: Resources::new(graph),
            passes: Vec::new(),
        }
    }

    /// Imports `image`, owned elsewhere, in `before`: how it was accessed
    /// since its last barrier, including its layout, as for a raw barrier.
    /// The graph leaves it in `after`. Importing it again with the same
    /// accesses returns the same handle.
    ///
    /// A swapchain image is imported from and to
    /// [`ImageAccess::Present`]; its first pass must render to it or copy
    /// into it, and executing a graph in which no pass does fails.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`](dirk_rhi::Error::Invalid) if `image` was
    /// imported with other accesses, `after` is
    /// [`ImageAccess::Undefined`], an access is a shader access without
    /// stages, or `image` lacks the usage an access needs.
    pub fn import_image(
        &mut self,
        image: &'a Image,
        before: ImageAccess,
        after: ImageAccess,
    ) -> Result<Handle<Image>> {
        let info = image.info();
        let all = Subresources::all(info.mip_levels, info.array_layers);
        self.images.import(image, image.label(), all, before, after)
    }

    /// Imports `image` like [`import_image`](Self::import_image), but its
    /// contents are not needed once the graph ends, such as a depth buffer
    /// cleared every frame: like a transient's, what passes render to it
    /// last is discarded. The graph still leaves it in `after`, from which
    /// later work imports it.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`import_image`](Self::import_image).
    pub fn import_scratch_image(
        &mut self,
        image: &'a Image,
        before: ImageAccess,
        after: ImageAccess,
    ) -> Result<Handle<Image>> {
        let handle = self.import_image(image, before, after)?;
        if let Source::Imported { scratch, .. } = &mut self.images.list[handle.index].source {
            *scratch = true;
        }
        Ok(handle)
    }

    /// Imports `buffer`, owned elsewhere, in `before`: how it was accessed
    /// since its last barrier. The graph leaves it in `after`. Importing it
    /// again with the same accesses returns the same handle.
    ///
    /// Buffers only the host writes each frame need no import; see
    /// [`PassBuilder::buffer`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`](dirk_rhi::Error::Invalid) if `buffer` was
    /// imported with other accesses, `after` is [`BufferAccess::None`], an
    /// access is a shader access without stages, or `buffer` lacks the usage
    /// an access needs.
    pub fn import_buffer(
        &mut self,
        buffer: &'a Buffer,
        before: BufferAccess,
        after: BufferAccess,
    ) -> Result<Handle<Buffer>> {
        self.buffers.import(
            buffer,
            buffer.label(),
            Subresources::all(1, 1),
            before,
            after,
        )
    }

    /// Adds a transient image described by `desc`, whose usage gains what
    /// the passes declaring it need. Its contents start undefined each
    /// frame. It is created, or reused from an earlier frame, when the graph
    /// executes, unless no pass declares it; the RHI then validates `desc`.
    pub fn create_image(&mut self, desc: &ImageDesc<'a>) -> Handle<Image> {
        let all = Subresources::all(desc.mip_levels, desc.array_layers);
        self.images.create(*desc, desc.label, all)
    }

    /// Adds a transient buffer described by `desc`, whose usage gains what
    /// the passes declaring it need. Its contents start undefined each
    /// frame. It is created, or reused from an earlier frame, when the graph
    /// executes, unless no pass declares it; the RHI then validates `desc`.
    pub fn create_buffer(&mut self, desc: &BufferDesc<'a>) -> Handle<Buffer> {
        self.buffers
            .create(*desc, desc.label, Subresources::all(1, 1))
    }

    /// Records every pass into `encoder`, which belongs to `frame`, then
    /// leaves each import in its `after` access. Transients come from
    /// `transients`, or are created, and return to it once recorded.
    ///
    /// # Errors
    ///
    /// Returns an error if a swapchain image is imported but no pass writes
    /// it, a transient cannot be created, a render pass is invalid, such as
    /// attachments of different extents, or a pass's callback fails. The encoder's recording is then incomplete and should
    /// be dropped.
    pub fn execute(
        mut self,
        frame: &Frame<'_>,
        encoder: &mut CommandEncoder<'_, Graphics>,
        transients: &mut Transients,
    ) -> Result<()> {
        let end = self.passes.len();
        self.images.finish(end)?;
        self.buffers.finish(end)?;
        let images = self.images.allocate(frame, &mut transients.images)?;
        let buffers = self.buffers.allocate(frame, &mut transients.buffers)?;
        let passes = std::mem::take(&mut self.passes);
        let recorded = self.record(
            passes,
            &Objects {
                images: self.images.objects(&images),
                buffers: self.buffers.objects(&buffers),
            },
            encoder,
        );
        transients
            .images
            .release(frame.number(), images.into_iter().flatten());
        transients
            .buffers
            .release(frame.number(), buffers.into_iter().flatten());
        recorded
    }

    /// Records `passes`, each after its barriers, then the boundary
    /// barriers.
    fn record(
        &self,
        passes: Vec<Pass<'a>>,
        objects: &Objects<'_>,
        encoder: &mut CommandEncoder<'_, Graphics>,
    ) -> Result<()> {
        let end = passes.len();
        for (index, pass) in passes.into_iter().enumerate() {
            self.barrier(index, objects, encoder)?;
            pass.record(index, &self.images, objects, encoder)?;
        }
        self.barrier(end, objects, encoder)
    }

    /// Records the barriers before pass `pass`, images and buffers in one
    /// call.
    fn barrier(
        &self,
        pass: usize,
        objects: &Objects<'_>,
        encoder: &mut CommandEncoder<'_, Graphics>,
    ) -> Result<()> {
        let images: Vec<_> = self
            .images
            .before(pass)
            .iter()
            .map(|barrier| ImageBarrier {
                image: objects.images.get(barrier.resource),
                range: ImageRange {
                    base_mip_level: barrier.subresources.mips.start,
                    mip_level_count: barrier.subresources.mips.end
                        - barrier.subresources.mips.start,
                    base_layer: barrier.subresources.layers.start,
                    layer_count: barrier.subresources.layers.end
                        - barrier.subresources.layers.start,
                },
                before: barrier.before,
                after: barrier.after,
                transfer: None,
            })
            .collect();
        let buffers: Vec<_> = self
            .buffers
            .before(pass)
            .iter()
            .map(|barrier| BufferBarrier {
                buffer: objects.buffers.get(barrier.resource),
                before: barrier.before,
                after: barrier.after,
                transfer: None,
            })
            .collect();
        if images.is_empty() && buffers.is_empty() {
            return Ok(());
        }
        // SAFETY: each `before` is the access the graph tracked since the
        // previous barrier on that range: the access an import was declared
        // in, the undefined contents of a transient, which no unfinished
        // frame still uses (see `Pool::take`), or the accesses passes
        // declared, which are the only way passes reach graph resources.
        unsafe { encoder.barrier(&buffers, &images) }
    }
}

impl Resources<'_, Image> {
    /// The one mip of one layer `range` of the image at `index` covers, as
    /// attachments render to, or why it covers several or none.
    fn attachment(&self, index: usize, range: ImageRange) -> Result<Subresource, String> {
        let Subresources { mips, layers } = self.list[index].resolve(range)?;
        if mips.len() != 1 || layers.len() != 1 {
            return Err(format!(
                "image `{}` is attached in mips {mips:?} of layers {layers:?}, not one mip of \
                 one layer",
                self.list[index].label
            ));
        }
        Ok(Subresource {
            mip_level: mips.start,
            layer: layers.start,
        })
    }

    /// What pass `pass` does with what it renders to attachment `image`:
    /// stores it unless the image is a transient or scratch import no later
    /// pass accesses.
    fn store(&self, image: usize, pass: usize) -> StoreOp {
        let tracked = &self.list[image];
        if matches!(tracked.source, Source::Imported { scratch: false, .. })
            || tracked.last_pass > Some(pass)
        {
            StoreOp::Store
        } else {
            StoreOp::Discard
        }
    }
}
