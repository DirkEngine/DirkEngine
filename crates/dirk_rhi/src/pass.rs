use std::ops::Range;

use dirk_shaders::indirect::{DrawArguments, DrawIndexedArguments};

use crate::{
    Buffer, BufferUsage, CommandEncoder, Error, Graphics, GraphicsPipeline, Image, ImageAccess,
    ImageRegion, ImageUsage, MAX_COLOR_TARGETS, Origin3d, PushData, Result, SampleCount,
    Subresource, command::Wrap, native, pipeline::Targets,
};

impl crate::Limits {
    /// Most color targets a pipeline or render pass may have: the device's
    /// limit, at most [`MAX_COLOR_TARGETS`].
    pub(crate) fn color_targets(&self) -> usize {
        (self.max_color_targets as usize).min(MAX_COLOR_TARGETS)
    }
}

/// What a render pass does with an attachment's contents when it begins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Keep the contents.
    Load,
    /// Clear to a value.
    Clear(T),
    /// Discard the contents, which become undefined.
    Discard,
}

/// What a render pass does with an attachment's contents when it ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Keep what the pass rendered.
    Store,
    /// Discard it, such as a multisampled target once resolved.
    Discard,
}

/// A color attachment of a render pass: one subresource of an image with
/// [`ImageUsage::COLOR_ATTACHMENT`], rendered to through the
/// [`ImageAccess::ColorAttachment`] access and the image's
/// [view of it](Image::subview).
#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    /// The image rendered to.
    pub image: &'a Image,
    /// The mip and layer rendered to; the default for single-mip,
    /// single-layer images.
    pub subresource: Subresource,
    /// What happens to its contents when the pass begins; clear colors are
    /// linear RGBA.
    pub load: LoadOp<[f32; 4]>,
    /// What happens to them when it ends.
    pub store: StoreOp,
    /// A single-mip, single-layer, single-sample image of the same format
    /// and extent that the multisampled `image` is averaged into when the
    /// pass ends, through the color attachment access too.
    pub resolve: Option<&'a Image>,
}

/// The depth attachment of a render pass: one subresource of a depth image
/// with [`ImageUsage::DEPTH_STENCIL_ATTACHMENT`], rendered to through the
/// [`ImageAccess::DepthStencilAttachment`] access and the image's
/// [view of it](Image::subview). Stencil, if the format has it, is loaded
/// and stored like depth and cleared to zero.
#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    /// The image rendered to.
    pub image: &'a Image,
    /// The mip and layer rendered to; the default for single-mip,
    /// single-layer images.
    pub subresource: Subresource,
    /// What happens to its depth when the pass begins; clear values are in
    /// `0.0..=1.0`.
    pub load: LoadOp<f32>,
    /// What happens to it when it ends.
    pub store: StoreOp,
}

/// Describes a render pass.
#[derive(Clone, Copy)]
pub struct RenderPassDesc<'a> {
    /// Debug label, shown by debugging tools and in errors.
    pub label: &'a str,
    /// Color attachments, in the order pipelines declare their targets.
    pub colors: &'a [ColorAttachment<'a>],
    /// The depth attachment, if any.
    pub depth: Option<DepthAttachment<'a>>,
}

impl<'a> RenderPassDesc<'a> {
    /// Every subresource the pass renders to, color attachments and their
    /// resolve targets, then depth, with the access it renders through.
    fn attachments(
        &self,
    ) -> impl Iterator<Item = (&'a Image, Subresource, ImageAccess)> + Clone + 'a {
        let colors = self
            .colors
            .iter()
            .flat_map(|color| {
                let resolve = color
                    .resolve
                    .map(|resolve| (resolve, Subresource::default()));
                [Some((color.image, color.subresource)), resolve]
            })
            .flatten()
            .map(|(image, subresource)| (image, subresource, ImageAccess::ColorAttachment));
        let depth = self.depth.map(|depth| {
            let access = ImageAccess::DepthStencilAttachment;
            (depth.image, depth.subresource, access)
        });
        colors.chain(depth)
    }
}

/// The region of the render area primitives are mapped to, and its depth
/// range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge, in pixels.
    pub x: f32,
    /// Top edge, in pixels.
    pub y: f32,
    /// Width, in pixels.
    pub width: f32,
    /// Height, in pixels.
    pub height: f32,
    /// Depth that clip-space depth 0 maps to, in `0.0..=1.0`.
    pub min_depth: f32,
    /// Depth that clip-space depth 1 maps to, in `0.0..=1.0`.
    pub max_depth: f32,
}

/// A rectangle of pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// Type of the indices in an index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    /// 16-bit indices.
    U16,
    /// 32-bit indices.
    U32,
}

impl IndexFormat {
    /// Size of one index in bytes.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

/// Indices read by an indexed draw.
#[derive(Clone)]
pub struct Indices<'a> {
    /// A buffer with [`BufferUsage::INDEX`], in the
    /// [`BufferAccess::Index`](crate::BufferAccess::Index) state.
    pub buffer: &'a Buffer,
    /// Type of its indices.
    pub format: IndexFormat,
    /// Offset of index zero, in bytes; a multiple of the index size.
    pub offset: u64,
    /// The indices drawn.
    pub range: Range<u32>,
    /// Added to each index before reading the vertex.
    pub base_vertex: i32,
}

/// The index buffer of indirect indexed draws, whose arguments choose the
/// indices drawn.
#[derive(Clone, Copy)]
pub struct IndexBuffer<'a> {
    /// A buffer with [`BufferUsage::INDEX`], in the
    /// [`BufferAccess::Index`](crate::BufferAccess::Index) state.
    pub buffer: &'a Buffer,
    /// Type of its indices.
    pub format: IndexFormat,
    /// Offset of index zero, in bytes; a multiple of the index size.
    pub offset: u64,
}

/// Arguments of indirect draws in a buffer, which the host or shaders
/// wrote: [`DrawArguments`] or [`DrawIndexedArguments`], back to back.
#[derive(Clone, Copy)]
pub struct IndirectDraws<'a> {
    /// A buffer with [`BufferUsage::INDIRECT`], in the
    /// [`BufferAccess::Indirect`](crate::BufferAccess::Indirect) state.
    pub buffer: &'a Buffer,
    /// Offset of the first draw's arguments, in bytes; a multiple of four.
    pub offset: u64,
    /// Number of draws.
    pub count: u32,
}

/// Records draws into attachments. Ends when dropped, and borrows its
/// encoder until then (or, if leaked, until the encoder records another
/// command or is submitted):
/// ```compile_fail
/// use dirk_rhi::{Buffer, BufferCopy, CommandEncoder, RenderPassDesc, Result};
/// fn copy_in_pass(
///     encoder: &mut CommandEncoder<'_>,
///     desc: &RenderPassDesc<'_>,
///     buffer: &Buffer,
///     copy: BufferCopy,
/// ) -> Result<()> {
///     let pass = encoder.render_pass(desc)?;
///     encoder.copy_buffer(buffer, buffer, &[copy])?;
///     pass.end();
///     Ok(())
/// }
/// ```
///
/// Only graphics encoders begin render passes:
/// ```compile_fail
/// use dirk_rhi::{CopyQueue, Frame, RenderPassDesc, Result};
/// fn copy_pass(frame: &Frame<'_>, desc: &RenderPassDesc<'_>) -> Result<()> {
///     let mut encoder = frame.encoder::<CopyQueue>("copies")?;
///     encoder.render_pass(desc)?.end();
///     Ok(())
/// }
/// ```
///
/// Draws only accept the pipeline's push data type:
/// ```compile_fail
/// use dirk_rhi::{GraphicsPipeline, RenderPass, Result};
/// fn draw(pass: &mut RenderPass<'_, '_>, pipeline: &GraphicsPipeline<[f32; 4]>) -> Result<()> {
///     pass.draw(pipeline, &[0_u32; 4], 0..3, 0..1)
/// }
/// ```
pub struct RenderPass<'e, 'f> {
    encoder: &'e mut CommandEncoder<'f, Graphics>,
    targets: Targets,
    /// Size of the render area, which every attachment covers.
    extent: (u32, u32),
    /// The pipeline bound last, to skip rebinding it.
    pipeline: Option<native::PipelineId>,
    /// The index buffer and its offset and format bound last.
    indices: Option<(native::BufferId, u64, IndexFormat)>,
}

impl<'f> CommandEncoder<'f, Graphics> {
    /// Begins a render pass. Viewport and scissor cover the whole render
    /// area, the extent shared by every attachment, until changed. The pass
    /// moves its attachments from their resting access into their attachment
    /// access, and back when it ends, like any safe command; see
    /// [`CommandEncoder`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pass has no attachment or too many,
    /// or an attachment is from another device, lacks the attachment usage,
    /// has no view of its subresource alone or a format that does not fit,
    /// a subresource is attached twice, as attachments or resolve targets,
    /// a clear depth is outside `0.0..=1.0`, or the attachments differ in
    /// extent or sample count, a resolve target does not match its
    /// attachment, or an attachment is not where the pass can reach it.
    pub fn render_pass<'e>(&'e mut self, desc: &RenderPassDesc<'_>) -> Result<RenderPass<'e, 'f>> {
        let (targets, extent) = self.check_pass(desc)?;
        let mut wrap = Wrap::default();
        for (image, subresource, access) in desc.attachments() {
            let region = ImageRegion {
                mip_level: subresource.mip_level,
                base_layer: subresource.layer,
                layer_count: 1,
                origin: Origin3d::default(),
                extent: image.info().extent.mip(subresource.mip_level),
            };
            self.enter(&mut wrap, image, [region], access)?;
        }
        self.end_pass();
        self.transition(&wrap.before);
        self.raw.begin_rendering(self.device(), desc, extent);
        self.rendering = true;
        self.after_pass = wrap.after;
        Ok(RenderPass {
            encoder: self,
            targets,
            extent,
            pipeline: None,
            indices: None,
        })
    }

    /// Validates a render pass and returns its targets and extent.
    fn check_pass(&self, desc: &RenderPassDesc<'_>) -> Result<(Targets, (u32, u32))> {
        let invalid = |reason: String| Err(Error::invalid("render pass", desc.label, reason));
        let limit = self.frame.rhi().capabilities().limits.color_targets();
        if desc.colors.len() > limit {
            return invalid(format!(
                "{} color attachments exceed the limit of {limit}",
                desc.colors.len()
            ));
        }
        // One layout per subresource: attaching one twice would need two.
        for (index, (image, subresource, _)) in desc.attachments().enumerate() {
            if desc
                .attachments()
                .skip(index + 1)
                .any(|(other, other_subresource, _)| {
                    (other.raw.id(), other_subresource) == (image.raw.id(), subresource)
                })
            {
                return invalid(format!(
                    "{subresource:?} of `{}` is attached twice",
                    image.label()
                ));
            }
        }
        // Each attachment's image and the extent and samples it renders.
        let mut attachments = desc
            .colors
            .iter()
            .map(|color| (color.image, color.subresource))
            .chain(desc.depth.map(|depth| (depth.image, depth.subresource)))
            .map(|(image, subresource)| {
                let info = image.info();
                let extent = info.extent.mip(subresource.mip_level);
                (image, (extent.width, extent.height), info.samples)
            });
        let Some((first, extent, samples)) = attachments.next() else {
            return invalid("a render pass needs an attachment".to_owned());
        };
        if let Some((image, other, other_samples)) = attachments
            .find(|&(_, other, other_samples)| (other, other_samples) != (extent, samples))
        {
            return invalid(format!(
                "`{}` renders {other:?} with {other_samples:?} samples, unlike `{}`",
                image.label(),
                first.label()
            ));
        }
        let mut targets = Targets {
            samples,
            ..Targets::default()
        };
        for (slot, color) in targets.colors.iter_mut().zip(desc.colors) {
            self.check_attachment(
                desc,
                color.image,
                color.subresource,
                ImageUsage::COLOR_ATTACHMENT,
            )?;
            if let Some(resolve) = color.resolve {
                self.check_attachment(
                    desc,
                    resolve,
                    Subresource::default(),
                    ImageUsage::COLOR_ATTACHMENT,
                )?;
                let (info, target) = (color.image.info(), resolve.info());
                if info.samples == SampleCount::X1
                    || target.samples != SampleCount::X1
                    || info.format != target.format
                    || info.extent != target.extent
                {
                    return invalid(format!(
                        "`{}` cannot be resolved into `{}`: resolves average a multisampled \
                         image into a single-sample one of the same format and extent",
                        color.image.label(),
                        resolve.label()
                    ));
                }
            }
            *slot = Some(color.image.info().format);
        }
        if let Some(depth) = desc.depth {
            self.check_attachment(
                desc,
                depth.image,
                depth.subresource,
                ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            )?;
            if let LoadOp::Clear(value) = depth.load
                && !(0.0..=1.0).contains(&value)
            {
                return invalid(format!("clear depth {value} is outside 0.0..=1.0"));
            }
            targets.depth = Some(depth.image.info().format);
        }
        Ok((targets, extent))
    }

    /// Checks that `subresource` of `image` can be an attachment with
    /// `usage`.
    fn check_attachment(
        &self,
        desc: &RenderPassDesc<'_>,
        image: &Image,
        subresource: Subresource,
        usage: ImageUsage,
    ) -> Result<()> {
        self.check_image(image, usage)?;
        let depth = usage == ImageUsage::DEPTH_STENCIL_ATTACHMENT;
        if image.view_of(subresource).is_none() || image.info().format.is_depth_stencil() != depth {
            return Err(Error::invalid(
                "render pass",
                desc.label,
                format!(
                    "`{}` is not a {} image with a view of {subresource:?} alone: a \
                     single-mip, single-layer 2D image, or a 2D or cube image with subviews",
                    image.label(),
                    if depth { "depth" } else { "color" }
                ),
            ));
        }
        Ok(())
    }
}

impl RenderPass<'_, '_> {
    /// Draws `instances` of the vertices in `vertices` with `pipeline`,
    /// whose shaders read `push`. Records nothing if a range is empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline belongs to another device
    /// or its targets differ from the pass's, or a range is reversed.
    pub fn draw<P: PushData>(
        &mut self,
        pipeline: &GraphicsPipeline<P>,
        push: &P,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<()> {
        let draws = Self::check_ranges(pipeline, &vertices, &instances)?;
        self.check_pipeline(pipeline)?;
        if draws {
            self.prepare(pipeline, push);
            self.encoder
                .raw
                .draw(self.encoder.device(), vertices, instances);
        }
        Ok(())
    }

    /// Makes the draws whose [`DrawArguments`] `draws` holds with
    /// `pipeline`, whose shaders read `push`. Each draw's arguments may be
    /// anything: shaders pull vertices through bounds-checked storage
    /// buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline belongs to another device
    /// or its targets differ from the pass's, or the arguments' buffer is
    /// from another device, lacks the indirect usage, or is too small for
    /// the draws or misaligned, or they exceed
    /// [`Limits::max_draw_indirect_count`](crate::Limits::max_draw_indirect_count).
    pub fn draw_indirect<P: PushData>(
        &mut self,
        pipeline: &GraphicsPipeline<P>,
        push: &P,
        draws: IndirectDraws<'_>,
    ) -> Result<()> {
        self.check_draws(draws, size_of::<DrawArguments>())?;
        self.check_pipeline(pipeline)?;
        self.prepare(pipeline, push);
        self.encoder.raw.draw_indirect(
            self.encoder.device(),
            &draws.buffer.raw,
            draws.offset,
            draws.count,
        );
        Ok(())
    }

    /// Makes the indexed draws whose [`DrawIndexedArguments`] `draws` holds,
    /// reading `indices`, with `pipeline`, whose shaders read `push`.
    ///
    /// # Safety
    ///
    /// The indices each draw reads lie inside the index buffer:
    /// `offset + (first_index + index_count) * index size` is at most its
    /// size. Arguments are only known on the GPU, and neither Vulkan
    /// without `robustBufferAccess2` nor Metal bound index reads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline belongs to another device
    /// or its targets differ from the pass's, a buffer is from another
    /// device or lacks its usage, the index offset is misaligned or past the
    /// index buffer, or the arguments' buffer is too small for the draws or
    /// misaligned, or they exceed
    /// [`Limits::max_draw_indirect_count`](crate::Limits::max_draw_indirect_count).
    pub unsafe fn draw_indexed_indirect<P: PushData>(
        &mut self,
        pipeline: &GraphicsPipeline<P>,
        push: &P,
        indices: IndexBuffer<'_>,
        draws: IndirectDraws<'_>,
    ) -> Result<()> {
        self.check_draws(draws, size_of::<DrawIndexedArguments>())?;
        let IndexBuffer {
            buffer,
            format,
            offset,
        } = indices;
        self.encoder.check_buffer(buffer, BufferUsage::INDEX)?;
        if !offset.is_multiple_of(format.size()) || offset >= buffer.info().size {
            return Err(Error::invalid(
                "indexed draw",
                buffer.label(),
                format!(
                    "{format:?} indices at offset {offset} are misaligned or past its {} bytes",
                    buffer.info().size
                ),
            ));
        }
        self.check_pipeline(pipeline)?;
        self.prepare(pipeline, push);
        self.bind_indices(indices);
        self.encoder.raw.draw_indexed_indirect(
            self.encoder.device(),
            &draws.buffer.raw,
            draws.offset,
            draws.count,
        );
        Ok(())
    }

    /// Draws `instances` of the vertices `indices` refer to with
    /// `pipeline`, whose shaders read `push`. Records nothing if a range is
    /// empty.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pipeline belongs to another device
    /// or its targets differ from the pass's, a range is reversed, or the
    /// index buffer is from another device, lacks the index usage, or is
    /// too small for the range or misaligned.
    pub fn draw_indexed<P: PushData>(
        &mut self,
        pipeline: &GraphicsPipeline<P>,
        push: &P,
        indices: Indices<'_>,
        instances: Range<u32>,
    ) -> Result<()> {
        let draws = Self::check_ranges(pipeline, &indices.range, &instances)?;
        self.check_pipeline(pipeline)?;
        let Indices {
            buffer,
            format,
            offset,
            ..
        } = indices;
        self.encoder.check_buffer(buffer, BufferUsage::INDEX)?;
        let end = u64::from(indices.range.end)
            .checked_mul(format.size())
            .and_then(|size| size.checked_add(offset));
        if !offset.is_multiple_of(format.size()) || end.is_none_or(|end| end > buffer.info().size) {
            return Err(Error::invalid(
                "indexed draw",
                buffer.label(),
                format!(
                    "{:?} indices {:?} at offset {offset} are misaligned or exceed its {} bytes",
                    format,
                    indices.range,
                    buffer.info().size
                ),
            ));
        }
        if !draws {
            return Ok(());
        }
        self.prepare(pipeline, push);
        self.bind_indices(IndexBuffer {
            buffer,
            format,
            offset,
        });
        self.encoder.raw.draw_indexed(
            self.encoder.device(),
            indices.range,
            indices.base_vertex,
            instances,
        );
        Ok(())
    }

    /// Maps primitives to `viewport` for the following draws.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the viewport is empty or not finite,
    /// its depth range is outside `0.0..=1.0`, or it exceeds the device's
    /// [viewport limits](crate::Limits::max_viewport_dimensions).
    pub fn set_viewport(&mut self, viewport: Viewport) -> Result<()> {
        let Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        } = viewport;
        let depths = 0.0..=1.0;
        if !(x.is_finite() && y.is_finite() && width.is_finite() && height.is_finite())
            || width <= 0.0
            || height <= 0.0
            || !depths.contains(&min_depth)
            || !depths.contains(&max_depth)
        {
            return Err(Error::invalid(
                "viewport",
                &format!("{viewport:?}"),
                "must be finite and not empty, with depths in 0.0..=1.0",
            ));
        }
        let limits = &self.encoder.frame.rhi().capabilities().limits;
        let [max_width, max_height] = limits.max_viewport_dimensions.map(f64::from);
        let [low, high] = limits.viewport_bounds.map(f64::from);
        let [x, y, width, height] = [x, y, width, height].map(f64::from);
        if width > max_width
            || height > max_height
            || x < low
            || y < low
            || x + width > high
            || y + height > high
        {
            return Err(Error::invalid(
                "viewport",
                &format!("{viewport:?}"),
                format!(
                    "exceeds the device's viewport limits: {max_width}x{max_height} pixels, \
                     edges in {low}..={high}"
                ),
            ));
        }
        self.encoder
            .raw
            .set_viewport(self.encoder.device(), viewport);
        Ok(())
    }

    /// Discards fragments outside `scissor` in the following draws.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the scissor extends past the render
    /// area.
    pub fn set_scissor(&mut self, scissor: Rect) -> Result<()> {
        let fits = |start: u32, length: u32, size: u32| {
            start.checked_add(length).is_some_and(|end| end <= size)
        };
        if !fits(scissor.x, scissor.width, self.extent.0)
            || !fits(scissor.y, scissor.height, self.extent.1)
        {
            return Err(Error::invalid(
                "scissor",
                &format!("{scissor:?}"),
                format!("extends past the {:?} render area", self.extent),
            ));
        }
        self.encoder.raw.set_scissor(self.encoder.device(), scissor);
        Ok(())
    }

    /// Ends the pass. Dropping it does the same.
    pub fn end(self) {}

    /// Rejects reversed ranges of a direct draw, and returns whether it
    /// draws anything.
    fn check_ranges<P: PushData>(
        pipeline: &GraphicsPipeline<P>,
        elements: &Range<u32>,
        instances: &Range<u32>,
    ) -> Result<bool> {
        if elements.start > elements.end || instances.start > instances.end {
            return Err(Error::invalid(
                "draw",
                pipeline.label(),
                format!("range {elements:?} or {instances:?} is reversed"),
            ));
        }
        Ok(!elements.is_empty() && !instances.is_empty())
    }

    /// Checks that `draws`, each `size` bytes of arguments, fit their
    /// buffer, which has the indirect usage, and the device's draw count.
    fn check_draws(&self, draws: IndirectDraws<'_>, size: usize) -> Result<()> {
        let IndirectDraws {
            buffer,
            offset,
            count,
        } = draws;
        self.encoder.check_buffer(buffer, BufferUsage::INDIRECT)?;
        let limit = self
            .encoder
            .frame
            .rhi()
            .capabilities()
            .limits
            .max_draw_indirect_count;
        let end = (size as u64)
            .checked_mul(u64::from(count))
            .and_then(|size| size.checked_add(offset));
        if !offset.is_multiple_of(4)
            || end.is_none_or(|end| end > buffer.info().size)
            || count > limit
        {
            return Err(Error::invalid(
                "indirect draw",
                buffer.label(),
                format!(
                    "{count} draws of {size} bytes at offset {offset} are misaligned, exceed its \
                     {} bytes or the device's {limit} draws",
                    buffer.info().size
                ),
            ));
        }
        Ok(())
    }

    /// Binds `indices` unless bound.
    fn bind_indices(&mut self, indices: IndexBuffer<'_>) {
        let IndexBuffer {
            buffer,
            format,
            offset,
        } = indices;
        let bound = Some((buffer.raw.id(), offset, format));
        if self.indices != bound {
            self.encoder
                .raw
                .bind_index_buffer(self.encoder.device(), &buffer.raw, offset, format);
            self.indices = bound;
        }
    }

    /// Checks that `pipeline` renders to this pass, on its device.
    fn check_pipeline<P: PushData>(&self, pipeline: &GraphicsPipeline<P>) -> Result<()> {
        if pipeline.targets != self.targets || !pipeline.raw.is_from(self.encoder.device()) {
            return Err(Error::invalid(
                "draw",
                pipeline.label(),
                format!(
                    "the pipeline renders to {:?} on its device, not this pass's {:?}",
                    pipeline.targets, self.targets
                ),
            ));
        }
        Ok(())
    }

    /// Binds `pipeline` unless bound, and pushes `push`.
    fn prepare<P: PushData>(&mut self, pipeline: &GraphicsPipeline<P>, push: &P) {
        let device = self.encoder.device();
        let id = pipeline.raw.id();
        if self.pipeline != Some(id) {
            self.encoder.raw.bind_pipeline(device, &pipeline.raw);
            self.pipeline = Some(id);
        }
        self.encoder.frame.reads(push);
        self.encoder.raw.push(device, bytemuck::bytes_of(push));
    }
}

impl Drop for RenderPass<'_, '_> {
    fn drop(&mut self) {
        self.encoder.end_pass();
    }
}
