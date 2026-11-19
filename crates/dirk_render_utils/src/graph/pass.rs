//! Passes: what they declare, and what they record with.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use dirk_rhi::{
    Buffer, BufferAccess, ColorAttachment, CommandEncoder, DepthAttachment, Error, Graphics, Image,
    ImageAccess, ImageRange, LoadOp, RenderPass, RenderPassDesc, Result, ShaderStages, Subresource,
};

use super::{
    Objects, RenderGraph,
    resources::{Handle, ImageSlice, Resource, Resources, Table, Use},
};

/// Marks a pass without attachments, which records into the frame's
/// [`CommandEncoder`].
pub struct Commands;

/// Marks a pass with attachments, which records into a [`RenderPass`].
pub struct Rendering;

/// Records a pass without attachments.
type RecordCommands<'a> =
    Box<dyn FnOnce(&mut PassContext<'_, CommandEncoder<'_, Graphics>>) -> Result<()> + 'a>;

/// Records a pass with attachments.
type RecordRendering<'a> =
    Box<dyn FnOnce(&mut PassContext<'_, RenderPass<'_, '_>>) -> Result<()> + 'a>;

/// Declares a pass's accesses, then adds it to the graph with its recording
/// callback by `record`. `K` is [`Rendering`] once it declares an
/// attachment, and [`Commands`] until then.
///
/// Passes without attachments record commands, not draws:
/// ```compile_fail
/// use dirk_render_utils::{Handle, RenderGraph};
/// use dirk_rhi::{GraphicsPipeline, Image, ImageAccess, Result};
/// fn draw<'a>(
///     graph: &mut RenderGraph<'a>,
///     image: Handle<Image>,
///     pipeline: &'a GraphicsPipeline<[f32; 4]>,
/// ) -> Result<()> {
///     graph
///         .pass("copy")
///         .image(image, ImageAccess::CopyDestination)
///         .record(|pass| pass.draw(pipeline, &[0.0; 4], 0..3, 0..1))
/// }
/// ```
#[must_use = "passes are added to the graph by `record`"]
pub struct PassBuilder<'g, 'a, K = Commands> {
    graph: &'g mut RenderGraph<'a>,
    declared: Declared,
    /// The first declaration the pass cannot make.
    error: Option<String>,
    kind: PhantomData<K>,
}

/// What a pass declared.
pub(crate) struct Declared {
    pub(crate) name: String,
    pub(crate) images: Vec<Use<ImageAccess>>,
    pub(crate) buffers: Vec<Use<BufferAccess>>,
    colors: Vec<Color>,
    depth: Vec<Depth>,
}

/// A color attachment, by image index.
struct Color {
    image: usize,
    target: Target,
    load: LoadOp<[f32; 4]>,
    resolve: Option<usize>,
}

/// A depth attachment, by image index.
struct Depth {
    image: usize,
    target: Target,
    load: LoadOp<f32>,
}

/// The subresource an attachment renders to.
struct Target {
    /// The range declared.
    range: ImageRange,
    /// The one subresource it covers, resolved once the pass is added.
    subresource: Subresource,
}

impl From<ImageRange> for Target {
    fn from(range: ImageRange) -> Self {
        Self {
            range,
            subresource: Subresource::default(),
        }
    }
}

/// A pass added to a graph.
pub(crate) struct Pass<'a> {
    declared: Declared,
    recorder: Recorder<'a>,
}

/// How a pass records.
enum Recorder<'a> {
    Commands(RecordCommands<'a>),
    Rendering(RecordRendering<'a>),
}

impl<'a> RenderGraph<'a> {
    /// Begins declaring a pass named `name`. Passes execute in the order
    /// they are added.
    pub fn pass(&mut self, name: &str) -> PassBuilder<'_, 'a> {
        PassBuilder {
            graph: self,
            declared: Declared {
                name: name.to_owned(),
                images: Vec::new(),
                buffers: Vec::new(),
                colors: Vec::new(),
                depth: Vec::new(),
            },
            error: None,
            kind: PhantomData,
        }
    }
}

impl<'g, 'a, K> PassBuilder<'g, 'a, K> {
    /// Samples `image`, or a slice of it, in `stages`.
    pub fn sampled(self, image: impl Into<ImageSlice>, stages: ShaderStages) -> Self {
        self.image(image, ImageAccess::Sampled(stages))
    }

    /// Accesses `image`, or a slice of it, through `access`. Attachments are
    /// declared with [`color`](Self::color) and [`depth`](Self::depth)
    /// instead.
    pub fn image(mut self, image: impl Into<ImageSlice>, access: ImageAccess) -> Self {
        if matches!(
            access,
            ImageAccess::ColorAttachment | ImageAccess::DepthStencilAttachment
        ) {
            self.fail(Err(format!(
                "declares attachment access {access:?} without an attachment; use `color` \
                 or `depth`"
            )));
            return self;
        }
        self.use_image(image.into(), access, !access.writes())
    }

    /// Accesses `buffer` through `access`.
    ///
    /// Buffers the host writes each frame, such as a
    /// [`PerFrame<Buffer>`](dirk_rhi::PerFrame) written with
    /// [`PerFrame::write`](dirk_rhi::PerFrame::write), are not declared:
    /// beginning the frame waited for the work that last read the slot's
    /// buffer, and submitting makes the host's writes visible.
    pub fn buffer(mut self, buffer: Handle<Buffer>, access: BufferAccess) -> Self {
        let result = self.graph.buffers.declare(
            &mut self.declared.buffers,
            buffer,
            ImageRange::ALL,
            access,
            !access.writes(),
        );
        self.fail(result);
        self
    }

    /// Renders to `image`, or one mip of one layer of it, as a color
    /// attachment, after `load`. Colors are in the order the pass's
    /// pipelines declare their targets. The graph stores what the pass
    /// renders unless nothing reads it later.
    pub fn color(
        self,
        image: impl Into<ImageSlice>,
        load: LoadOp<[f32; 4]>,
    ) -> PassBuilder<'g, 'a, Rendering> {
        self.attach_color(image.into(), load, None)
    }

    /// Renders to the multisampled `image` as a color attachment, after
    /// `load`, and averages it into the single-sample `resolve` when the
    /// pass ends.
    pub fn color_resolved(
        self,
        image: Handle<Image>,
        load: LoadOp<[f32; 4]>,
        resolve: Handle<Image>,
    ) -> PassBuilder<'g, 'a, Rendering> {
        self.attach_color(image.into(), load, Some(resolve))
            .use_image(resolve.into(), ImageAccess::ColorAttachment, false)
    }

    /// Renders to `image`, or one mip of one layer of it, as the depth
    /// attachment, after `load`. The graph stores what the pass renders
    /// unless nothing reads it later.
    pub fn depth(
        mut self,
        image: impl Into<ImageSlice>,
        load: LoadOp<f32>,
    ) -> PassBuilder<'g, 'a, Rendering> {
        let image = image.into();
        if !self.declared.depth.is_empty() {
            self.fail(Err("declares more than one depth attachment".to_owned()));
        }
        self.declared.depth.push(Depth {
            image: image.image.index,
            target: image.range.into(),
            load,
        });
        self.use_image(
            image,
            ImageAccess::DepthStencilAttachment,
            matches!(load, LoadOp::Load),
        )
        .into_kind()
    }

    /// Declares a color attachment.
    fn attach_color(
        mut self,
        image: ImageSlice,
        load: LoadOp<[f32; 4]>,
        resolve: Option<Handle<Image>>,
    ) -> PassBuilder<'g, 'a, Rendering> {
        self.declared.colors.push(Color {
            image: image.image.index,
            target: image.range.into(),
            load,
            resolve: resolve.map(|resolve| resolve.index),
        });
        self.use_image(
            image,
            ImageAccess::ColorAttachment,
            matches!(load, LoadOp::Load),
        )
        .into_kind()
    }

    /// Declares an access of `image`.
    fn use_image(mut self, image: ImageSlice, access: ImageAccess, reads: bool) -> Self {
        let result = self.graph.images.declare(
            &mut self.declared.images,
            image.image,
            image.range,
            access,
            reads,
        );
        self.fail(result);
        self
    }

    /// Keeps the first declaration error.
    fn fail(&mut self, result: Result<(), String>) {
        if let Err(reason) = result {
            self.error.get_or_insert(reason);
        }
    }

    /// The same declarations, as a pass of kind `L`.
    fn into_kind<L>(self) -> PassBuilder<'g, 'a, L> {
        PassBuilder {
            graph: self.graph,
            declared: self.declared,
            error: self.error,
            kind: PhantomData,
        }
    }

    /// Validates the declarations, derives the pass's barriers and adds it.
    /// Leaves the graph unchanged on error.
    fn add(self, recorder: Recorder<'a>) -> Result<()> {
        let Self {
            graph,
            mut declared,
            error,
            ..
        } = self;
        let invalid = |reason| Error::Invalid {
            object: "render graph pass",
            label: declared.name.clone(),
            reason,
        };
        if let Some(reason) = error {
            return Err(invalid(reason));
        }
        let targets = declared
            .colors
            .iter_mut()
            .map(|color| (color.image, &mut color.target))
            .chain(
                declared
                    .depth
                    .iter_mut()
                    .map(|depth| (depth.image, &mut depth.target)),
            );
        for (image, target) in targets {
            target.subresource = graph
                .images
                .attachment(image, target.range)
                .map_err(invalid)?;
        }
        let images = graph
            .images
            .prepare(std::mem::take(&mut declared.images))
            .map_err(invalid)?;
        let buffers = graph
            .buffers
            .prepare(std::mem::take(&mut declared.buffers))
            .map_err(invalid)?;
        let pass = graph.passes.len();
        graph.images.apply(pass, &images);
        graph.buffers.apply(pass, &buffers);
        declared.images = images;
        declared.buffers = buffers;
        graph.passes.push(Pass { declared, recorder });
        Ok(())
    }
}

impl<'a> PassBuilder<'_, 'a, Commands> {
    /// Adds the pass, which `record` records into the frame's command
    /// encoder, such as copies, when the graph executes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pass cannot make a declaration: it
    /// declares a resource of another graph, an access passes do not make,
    /// a shader access without stages, an access an import lacks the usage
    /// for, conflicting accesses of one resource, a read of contents nothing
    /// wrote, or a first access of a just-acquired swapchain image that is
    /// not a write as a color attachment or copy destination. The graph is
    /// left unchanged.
    pub fn record(
        self,
        record: impl FnOnce(&mut PassContext<'_, CommandEncoder<'_, Graphics>>) -> Result<()> + 'a,
    ) -> Result<()> {
        self.add(Recorder::Commands(Box::new(record)))
    }
}

impl<'a> PassBuilder<'_, 'a, Rendering> {
    /// Adds the pass, which `record` records into a render pass over its
    /// attachments when the graph executes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pass cannot make a declaration, as
    /// for [passes without attachments](PassBuilder::record), or declares
    /// more than one depth attachment. The graph is left unchanged.
    pub fn record(
        self,
        record: impl FnOnce(&mut PassContext<'_, RenderPass<'_, '_>>) -> Result<()> + 'a,
    ) -> Result<()> {
        self.add(Recorder::Rendering(Box::new(record)))
    }
}

/// What a pass records with: a [`RenderPass`] over its attachments, or the
/// frame's [`CommandEncoder`] for passes without. Dereferences to it, and
/// resolves the resources the pass declared.
pub struct PassContext<'p, R> {
    recorder: &'p mut R,
    declared: &'p Declared,
    objects: &'p Objects<'p>,
}

impl<'p, R> PassContext<'p, R> {
    /// The image behind `image`, which the pass declared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pass did not declare `image`.
    pub fn image(&self, image: Handle<Image>) -> Result<&'p Image> {
        self.resolve(&self.declared.images, &self.objects.images, image)
    }

    /// The buffer behind `buffer`, which the pass declared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the pass did not declare `buffer`.
    pub fn buffer(&self, buffer: Handle<Buffer>) -> Result<&'p Buffer> {
        self.resolve(&self.declared.buffers, &self.objects.buffers, buffer)
    }

    /// The resource behind `handle` in `table`, if `uses` declare it.
    fn resolve<T: Resource>(
        &self,
        uses: &[Use<T::Access>],
        table: &Table<'p, T>,
        handle: Handle<T>,
    ) -> Result<&'p T> {
        if table.owns(handle) && uses.iter().any(|usage| usage.resource == handle.index) {
            Ok(table.get(handle.index))
        } else {
            Err(Error::Invalid {
                object: "render graph pass",
                label: self.declared.name.clone(),
                reason: format!("did not declare {} {handle:?}", T::KIND),
            })
        }
    }
}

impl<R> Deref for PassContext<'_, R> {
    type Target = R;

    fn deref(&self) -> &R {
        self.recorder
    }
}

impl<R> DerefMut for PassContext<'_, R> {
    fn deref_mut(&mut self) -> &mut R {
        self.recorder
    }
}

impl Pass<'_> {
    /// Records the pass, the `index`th of a graph whose images are
    /// `images`, into `encoder`.
    pub(crate) fn record(
        self,
        index: usize,
        images: &Resources<'_, Image>,
        objects: &Objects<'_>,
        encoder: &mut CommandEncoder<'_, Graphics>,
    ) -> Result<()> {
        let Self { declared, recorder } = self;
        let record = match recorder {
            Recorder::Commands(record) => {
                return record(&mut PassContext {
                    recorder: encoder,
                    declared: &declared,
                    objects,
                });
            }
            Recorder::Rendering(record) => record,
        };
        let image = |index| objects.images.get(index);
        let colors: Vec<_> = declared
            .colors
            .iter()
            .map(|color| ColorAttachment {
                image: image(color.image),
                subresource: color.target.subresource,
                load: color.load,
                store: images.store(color.image, index),
                resolve: color.resolve.map(image),
            })
            .collect();
        let depth = declared.depth.first().map(|depth| DepthAttachment {
            image: image(depth.image),
            subresource: depth.target.subresource,
            load: depth.load,
            store: images.store(depth.image, index),
        });
        let mut pass = encoder.render_pass(&RenderPassDesc {
            label: &declared.name,
            colors: &colors,
            depth,
        })?;
        record(&mut PassContext {
            recorder: &mut pass,
            declared: &declared,
            objects,
        })
    }
}
