//! Backend-neutral render graph executed through the renderer RHI.

use crate::{
    Result,
    resources::{ActiveImage, ActiveImageView, command_pool::CommandBuffer, device::RenderDevice},
};
use dirk_rhi::{
    Backend as _, Color, CommandBuffer as _, DependencyInfo, Extent3d, ImageAspects, ImageBarrier,
    ImageDesc, ImageState, ImageUsages, ImageViewDesc, ImageViewType, LoadOp, RenderingInfo,
    SampleCount, ShaderStages, StoreOp, TextureFormat,
};

/// Opaque index into the graph texture table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(u32);

impl TextureHandle {
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Selected subresources of a graph texture.
///
/// Counts use the backend's whole-remainder convention: `u32::MAX` means "all
/// remaining levels/layers". The compiler tracks state per mip range, so
/// passes may declare disjoint mip accesses without forcing transitions of
/// untouched subresources. Array layers are carried through declarations into
/// emitted barriers untracked until a consumer needs per-layer states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceRange {
    /// First mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    pub mip_level_count: u32,
    /// First array layer.
    pub base_array_layer: u32,
    /// Array layer count.
    pub array_layer_count: u32,
}

impl SubresourceRange {
    /// Every mip level and array layer.
    pub const WHOLE: Self = Self {
        base_mip_level: 0,
        mip_level_count: u32::MAX,
        base_array_layer: 0,
        array_layer_count: u32::MAX,
    };

    fn overlaps_mips(self, other: Self) -> bool {
        let (this, that) = (self.mip_span(), other.mip_span());
        this.start < that.end && that.start < this.end
    }

    fn mip_intersect(self, other: Self) -> (u32, u32) {
        let (this, that) = (self.mip_span(), other.mip_span());
        let start = this.start.max(that.start);
        let end = this.end.min(that.end).max(start);
        (
            u32::try_from(start).unwrap_or(u32::MAX),
            u32::try_from(end - start).unwrap_or(u32::MAX),
        )
    }

    /// Splits `self`'s mip span around `minus`, largest piece first.
    fn mip_difference(self, minus: Self) -> [(u32, u32); 2] {
        let (this, that) = (self.mip_span(), minus.mip_span());
        let lower = this.start..that.start.clamp(this.start, this.end);
        let upper = that.end.clamp(this.start, this.end)..this.end;
        [
            (
                u32::try_from(lower.start).unwrap_or(u32::MAX),
                u32::try_from(lower.end - lower.start).unwrap_or(u32::MAX),
            ),
            (
                u32::try_from(upper.start).unwrap_or(u32::MAX),
                u32::try_from(upper.end - upper.start).unwrap_or(u32::MAX),
            ),
        ]
    }

    fn with_mips(self, base_mip_level: u32, mip_level_count: u32) -> Self {
        Self {
            base_mip_level,
            mip_level_count,
            ..self
        }
    }

    fn mip_span(&self) -> std::ops::Range<u64> {
        let start = u64::from(self.base_mip_level);
        let count = u64::from(self.mip_level_count);
        if count >= u64::from(u32::MAX) {
            start..u64::from(u32::MAX)
        } else {
            start..start + count
        }
    }
}

/// Texture allocation and import metadata used while building a graph.
///
/// Usage capabilities are intentionally absent: the compiler derives them from
/// declared accesses. Imported images keep whatever capabilities they were
/// created with.
pub struct TextureDesc {
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// Texel format.
    pub format: TextureFormat,
    /// Sample count.
    pub samples: SampleCount,
    /// `Some` for externally-owned images such as swapchain images.
    /// `None` for transient images the graph creates and destroys itself.
    pub imported: Option<ImportedTexture>,
}

/// Externally owned image and its graph-boundary states.
#[derive(Clone)]
pub struct ImportedTexture {
    pub image: ActiveImage,
    pub view: ActiveImageView,
    pub aspects: ImageAspects,
    pub initial_state: ImageState,
    pub final_state: ImageState,
}

/// Backend image resources resolved for a graph texture.
pub struct ResolvedImage {
    pub image: ActiveImage,
    pub view: ActiveImageView,
    pub aspects: ImageAspects,
}

/// Read-only use of a texture declared by a pass.
///
/// Declarations are data-shaped on purpose: stages and ranges travel with the
/// access so later compiler refinements need no call-site changes.
#[derive(Clone, Copy, Debug)]
pub enum TextureRead {
    /// Sampled by shaders in the given stages.
    Sampled {
        /// Stages that sample the texture.
        stages: ShaderStages,
    },
    /// Source of a copy or blit.
    CopySource,
}

/// Writable use of a texture declared by a pass.
#[derive(Clone, Copy)]
pub enum TextureWrite {
    /// Render target, optionally resolving into another texture.
    ColorAttachment {
        /// Load/store behavior and clear value.
        info: AttachmentInfo,
        /// Multisample resolve target written at the end of the pass.
        resolve: Option<TextureHandle>,
    },
    /// Depth/stencil render target.
    DepthStencilAttachment(AttachmentInfo),
    /// Shader storage image access.
    Storage {
        /// Stages accessing the image; honored once states carry stages.
        #[allow(dead_code)]
        stages: ShaderStages,
    },
    /// Destination of a copy or blit.
    CopyDestination,
}

impl TextureWrite {
    fn state(self) -> ImageState {
        match self {
            TextureWrite::ColorAttachment { .. } => ImageState::ColorAttachment,
            TextureWrite::DepthStencilAttachment(_) => ImageState::DepthStencilAttachment,
            TextureWrite::Storage { .. } => ImageState::ShaderWrite,
            TextureWrite::CopyDestination => ImageState::CopyDestination,
        }
    }

    fn attachment(self) -> Option<AttachmentInfo> {
        match self {
            TextureWrite::ColorAttachment { info, .. }
            | TextureWrite::DepthStencilAttachment(info) => Some(info),
            _ => None,
        }
    }
}

impl TextureRead {
    fn state(self) -> ImageState {
        match self {
            TextureRead::Sampled { .. } => ImageState::ShaderRead,
            TextureRead::CopySource => ImageState::CopySource,
        }
    }

    fn stages(self) -> ShaderStages {
        match self {
            TextureRead::Sampled { stages } => stages,
            TextureRead::CopySource => ShaderStages::NONE,
        }
    }
}

#[derive(Clone, Copy)]
enum AttachmentClear {
    Color(Color),
    DepthStencil { depth: f32, stencil: u32 },
}

/// Attachment load/store behavior.
#[derive(Clone, Copy)]
pub struct AttachmentInfo {
    clear: Option<AttachmentClear>,
    load: bool,
    store: StoreOp,
}

impl AttachmentInfo {
    #[must_use]
    pub fn clear_color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            clear: Some(AttachmentClear::Color(Color { r, g, b, a })),
            load: false,
            store: StoreOp::Store,
        }
    }

    #[allow(unused)]
    #[must_use]
    pub fn load_store() -> Self {
        Self {
            clear: None,
            load: true,
            store: StoreOp::Store,
        }
    }

    #[allow(unused)]
    #[must_use]
    pub fn clear_depth(depth: f32, stencil: u32) -> Self {
        Self {
            clear: Some(AttachmentClear::DepthStencil { depth, stencil }),
            load: false,
            store: StoreOp::Store,
        }
    }

    #[must_use]
    pub fn clear_discard_depth(depth: f32, stencil: u32) -> Self {
        Self {
            clear: Some(AttachmentClear::DepthStencil { depth, stencil }),
            load: false,
            store: StoreOp::DontCare,
        }
    }

    fn color_load(self) -> LoadOp<Color> {
        match self.clear {
            Some(AttachmentClear::Color(color)) => LoadOp::Clear(color),
            _ if self.load => LoadOp::Load,
            _ => LoadOp::DontCare,
        }
    }

    fn depth_load(self) -> LoadOp<f32> {
        match self.clear {
            Some(AttachmentClear::DepthStencil { depth, .. }) => LoadOp::Clear(depth),
            _ if self.load => LoadOp::Load,
            _ => LoadOp::DontCare,
        }
    }

    fn stencil_load(self) -> LoadOp<u32> {
        match self.clear {
            Some(AttachmentClear::DepthStencil { stencil, .. }) => LoadOp::Clear(stencil),
            _ if self.load => LoadOp::Load,
            _ => LoadOp::DontCare,
        }
    }
}

/// One declared texture access inside a pass.
#[derive(Clone, Copy)]
struct AccessDecl {
    handle: TextureHandle,
    range: SubresourceRange,
    state: ImageState,
    /// Declared shader stages; folded into backend state mappings today and
    /// honored by the compiler once states carry stage granularity.
    #[allow(dead_code)]
    stages: ShaderStages,
    attachment: Option<AttachmentInfo>,
}

pub type PassCallback<'a> =
    Box<dyn FnOnce(&mut CommandBuffer, &PassContext<'_>) -> Result<()> + 'a>;

struct PassNode<'a> {
    name: String,
    reads: Vec<AccessDecl>,
    writes: Vec<AccessDecl>,
    color_resolves: Vec<(TextureHandle, TextureHandle)>,
    callback: Option<PassCallback<'a>>,
}

pub struct PassBuilder<'graph, 'a> {
    pass: &'graph mut PassNode<'a>,
}

impl<'a> PassBuilder<'_, 'a> {
    /// Declares a read access with an explicit subresource range.
    pub fn read_range(
        &mut self,
        handle: TextureHandle,
        read: TextureRead,
        range: SubresourceRange,
    ) -> &mut Self {
        self.pass.reads.push(AccessDecl {
            handle,
            range,
            state: read.state(),
            stages: read.stages(),
            attachment: None,
        });
        self
    }

    /// Declares a read access covering every subresource.
    pub fn read(&mut self, handle: TextureHandle, read: TextureRead) -> &mut Self {
        self.read_range(handle, read, SubresourceRange::WHOLE)
    }

    /// Declares a sampled read covering every subresource.
    #[allow(unused)]
    pub fn read_sampled(&mut self, handle: TextureHandle, stages: ShaderStages) -> &mut Self {
        self.read(handle, TextureRead::Sampled { stages })
    }

    /// Declares a shader-storage write covering every subresource.
    ///
    /// Compute dispatch support is tracked in .agents/plans/01.
    #[allow(unused)]
    pub fn write_storage(&mut self, handle: TextureHandle, stages: ShaderStages) -> &mut Self {
        self.write(handle, TextureWrite::Storage { stages })
    }

    /// Declares a copy/blit source covering every subresource.
    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn read_transfer_src(&mut self, handle: TextureHandle) -> &mut Self {
        self.read(handle, TextureRead::CopySource)
    }

    /// Declares a write access with an explicit subresource range.
    pub fn write_range(
        &mut self,
        handle: TextureHandle,
        write: TextureWrite,
        range: SubresourceRange,
    ) -> &mut Self {
        self.pass.writes.push(AccessDecl {
            handle,
            range,
            state: write.state(),
            stages: ShaderStages::NONE,
            attachment: write.attachment(),
        });
        if let TextureWrite::ColorAttachment {
            resolve: Some(resolve_handle),
            ..
        } = write
        {
            self.pass.writes.push(AccessDecl {
                handle: resolve_handle,
                range,
                state: ImageState::ColorAttachment,
                stages: ShaderStages::NONE,
                attachment: None,
            });
            self.pass.color_resolves.push((handle, resolve_handle));
        }
        self
    }

    /// Declares a write access covering every subresource.
    pub fn write(&mut self, handle: TextureHandle, write: TextureWrite) -> &mut Self {
        self.write_range(handle, write, SubresourceRange::WHOLE)
    }

    /// Declares a colour attachment write covering every subresource.
    pub fn write_color_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.write(
            handle,
            TextureWrite::ColorAttachment {
                info,
                resolve: None,
            },
        )
    }

    /// Declares a multisampled colour attachment resolving into
    /// `resolve_handle`.
    pub fn write_color_attachment_with_resolve(
        &mut self,
        handle: TextureHandle,
        resolve_handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.write(
            handle,
            TextureWrite::ColorAttachment {
                info,
                resolve: Some(resolve_handle),
            },
        )
    }

    /// Declares a depth/stencil attachment write covering every subresource.
    pub fn write_depth_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.write(handle, TextureWrite::DepthStencilAttachment(info))
    }

    /// Declares a copy/blit destination covering every subresource.
    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn write_transfer_dst(&mut self, handle: TextureHandle) -> &mut Self {
        self.write(handle, TextureWrite::CopyDestination)
    }

    /// Provides the command-recording callback for this pass.
    pub fn execute(&mut self, callback: PassCallback<'a>) {
        self.pass.callback = Some(callback);
    }
}

pub struct RenderGraph<'a> {
    textures: Vec<TextureDesc>,
    passes: Vec<PassNode<'a>>,
}

impl<'a> RenderGraph<'a> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            passes: Vec::new(),
        }
    }

    pub fn create_texture(&mut self, desc: TextureDesc) -> TextureHandle {
        assert!(
            desc.imported.is_none(),
            "use import_texture for external images"
        );
        self.push_texture(desc)
    }

    pub fn import_texture(&mut self, desc: TextureDesc) -> TextureHandle {
        assert!(
            desc.imported.is_some(),
            "import_texture requires an imported image"
        );
        self.push_texture(desc)
    }

    fn push_texture(&mut self, desc: TextureDesc) -> TextureHandle {
        let handle = texture_handle(self.textures.len());
        self.textures.push(desc);
        handle
    }

    pub fn add_pass(&mut self, name: impl Into<String>) -> PassBuilder<'_, 'a> {
        self.passes.push(PassNode {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            color_resolves: Vec::new(),
            callback: None,
        });
        let index = self.passes.len() - 1;
        let pass = &mut self.passes[index];
        PassBuilder { pass }
    }

    pub fn run(self, device: &RenderDevice, cmd: &mut CommandBuffer) -> Result<()> {
        GraphExecutor::new(device, self.compile())?.execute(cmd)
    }

    fn compile(self) -> CompiledGraph<'a> {
        let initial_states = self.textures.iter().map(|desc| {
            vec![RangeState {
                mips: (0, u32::MAX),
                state: desc
                    .imported
                    .as_ref()
                    .map_or(ImageState::Undefined, |imported| imported.initial_state),
            }]
        });
        let mut states: Vec<Vec<RangeState>> = initial_states.collect();
        let mut derived_usages = vec![ImageUsages::NONE; self.textures.len()];
        let mut compiled_passes = Vec::with_capacity(self.passes.len());

        for pass in self.passes {
            let mut barriers = Vec::new();
            for usage in pass.reads.iter().chain(&pass.writes) {
                derived_usages[usage.handle.index()].insert(usage_bits(usage.state));
                barriers.extend(transition_barriers(
                    &mut states[usage.handle.index()],
                    usage,
                ));
            }

            let mut colors = Vec::new();
            let mut depth = None;
            let mut extent = None;
            for usage in &pass.writes {
                let Some(info) = usage.attachment else {
                    continue;
                };
                let desc = &self.textures[usage.handle.index()];
                extent.get_or_insert(Extent3d::new_2d(desc.width, desc.height));
                match usage.state {
                    ImageState::ColorAttachment => colors.push(CompiledColorAttachment {
                        handle: usage.handle,
                        info,
                        resolve: pass.color_resolves.iter().find_map(|(source, target)| {
                            (*source == usage.handle).then_some(*target)
                        }),
                    }),
                    ImageState::DepthStencilAttachment => depth = Some((usage.handle, info)),
                    _ => {}
                }
            }

            compiled_passes.push(CompiledPass {
                name: pass.name,
                barriers,
                colors,
                depth,
                extent,
                declared: pass
                    .reads
                    .iter()
                    .map(|access| (access.handle.0, false))
                    .chain(pass.writes.iter().map(|access| (access.handle.0, true)))
                    .collect(),
                callback: pass.callback,
            });
        }

        let final_barriers = self
            .textures
            .iter()
            .enumerate()
            .filter_map(|(index, desc)| {
                let imported = desc.imported.as_ref()?;
                let entries = &mut states[index];
                let mut barriers = Vec::new();
                for entry in entries.iter_mut() {
                    if entry.state != imported.final_state {
                        barriers.push(CompiledBarrier {
                            handle: texture_handle(index),
                            old_state: entry.state,
                            new_state: imported.final_state,
                            range: SubresourceRange::WHOLE.with_mips(entry.mips.0, entry.mips.1),
                        });
                        entry.state = imported.final_state;
                    }
                }
                (!barriers.is_empty()).then_some(barriers)
            })
            .flatten()
            .collect();

        CompiledGraph {
            textures: self.textures,
            usages: derived_usages,
            passes: compiled_passes,
            final_barriers,
        }
    }
}

impl Default for RenderGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Emits the barriers required to apply `usage` to a tracked texture and
/// updates its per-mip states.
fn transition_barriers(states: &mut Vec<RangeState>, usage: &AccessDecl) -> Vec<CompiledBarrier> {
    let mut barriers = Vec::new();
    for entry in states.iter_mut() {
        if !usage.range.overlaps_mips(entry.mip_range()) {
            continue;
        }
        if barrier_needed(entry.state, usage.state) {
            let (base_mip_level, mip_level_count) = entry.mip_range().mip_intersect(usage.range);
            barriers.push(CompiledBarrier {
                handle: usage.handle,
                old_state: entry.state,
                new_state: usage.state,
                range: usage.range.with_mips(base_mip_level, mip_level_count),
            });
        }
    }

    // Replace the covered mip span with the new state, keeping disjoint spans.
    let mut updated: Vec<RangeState> = Vec::with_capacity(states.len() + 1);
    for entry in states.drain(..) {
        if !entry.mip_range().overlaps_mips(usage.range) {
            updated.push(entry);
            continue;
        }
        updated.extend(
            entry
                .mip_range()
                .mip_difference(usage.range)
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .map(|(base, count)| RangeState {
                    mips: (base, count),
                    state: entry.state,
                }),
        );
    }
    let (base, count) = usage.range.mip_intersect(usage.range);
    updated.push(RangeState {
        mips: (base, count),
        state: usage.state,
    });
    *states = updated;
    barriers
}

fn barrier_needed(old: ImageState, new: ImageState) -> bool {
    old != new || is_write(old) || is_write(new)
}

fn is_write(state: ImageState) -> bool {
    matches!(
        state,
        ImageState::CopyDestination
            | ImageState::ShaderWrite
            | ImageState::ColorAttachment
            | ImageState::DepthStencilAttachment
    )
}

/// Derives allocation capabilities from a semantic state.
///
/// Exact because access kinds distinguish sampled reads from storage access.
/// Transient-memory inference is deliberately not done here; that belongs to
/// the planned transient allocator (.agents/plans/03).
fn usage_bits(state: ImageState) -> ImageUsages {
    match state {
        ImageState::CopySource => ImageUsages::COPY_SRC,
        ImageState::CopyDestination => ImageUsages::COPY_DST,
        ImageState::ShaderRead => ImageUsages::SAMPLED,
        ImageState::ShaderWrite => ImageUsages::STORAGE,
        ImageState::ColorAttachment => ImageUsages::COLOR_ATTACHMENT,
        ImageState::DepthStencilAttachment | ImageState::DepthStencilAttachmentReadOnly => {
            ImageUsages::DEPTH_STENCIL_ATTACHMENT
        }
        ImageState::Undefined | ImageState::Present => ImageUsages::NONE,
    }
}

/// Last-known synchronization state of one mip span.
struct RangeState {
    /// `(base mip level, mip level count)` with the whole-remainder
    /// convention for counts.
    mips: (u32, u32),
    state: ImageState,
}

impl RangeState {
    fn mip_range(&self) -> SubresourceRange {
        SubresourceRange::WHOLE.with_mips(self.mips.0, self.mips.1)
    }
}

struct CompiledBarrier {
    handle: TextureHandle,
    old_state: ImageState,
    new_state: ImageState,
    range: SubresourceRange,
}

struct CompiledColorAttachment {
    handle: TextureHandle,
    info: AttachmentInfo,
    resolve: Option<TextureHandle>,
}

/// Validated resource resolution handed to pass callbacks.
pub struct PassContext<'ctx> {
    device: &'ctx RenderDevice,
    /// Resolved graph textures indexed by [`TextureHandle`].
    #[cfg_attr(feature = "editor", allow(dead_code))]
    images: &'ctx [ResolvedImage],
    /// `(texture table index, is_write)` pairs declared by the running pass.
    #[cfg_attr(feature = "editor", allow(dead_code))]
    declared: &'ctx [(u32, bool)],
}

impl PassContext<'_> {
    /// Renderer device shared by all passes.
    #[cfg_attr(not(feature = "editor"), allow(unused))]
    pub fn device(&self) -> &RenderDevice {
        self.device
    }

    /// Resolves a texture this pass declared.
    ///
    /// # Errors
    ///
    /// Returns an error when `handle` was not declared by the running pass,
    /// since such an access would skip synchronization, or when `handle` is
    /// out of range.
    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn resolve(&self, handle: TextureHandle) -> Result<&ResolvedImage> {
        if !self.declared.iter().any(|(index, _)| *index == handle.0) {
            return Err(dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::Undeclared).into());
        }
        self.images
            .get(handle.index())
            .ok_or_else(|| dirk_rhi::Error::from(dirk_rhi::InvalidResourceKind::OutOfRange).into())
    }
}

struct CompiledPass<'a> {
    name: String,
    barriers: Vec<CompiledBarrier>,
    colors: Vec<CompiledColorAttachment>,
    depth: Option<(TextureHandle, AttachmentInfo)>,
    extent: Option<Extent3d>,
    declared: Vec<(u32, bool)>,
    callback: Option<PassCallback<'a>>,
}

struct CompiledGraph<'a> {
    textures: Vec<TextureDesc>,
    /// Capabilities derived from declared accesses, indexed like `textures`.
    usages: Vec<ImageUsages>,
    passes: Vec<CompiledPass<'a>>,
    final_barriers: Vec<CompiledBarrier>,
}

struct GraphExecutor<'a> {
    device: RenderDevice,
    images: Vec<ResolvedImage>,
    passes: Vec<CompiledPass<'a>>,
    final_barriers: Vec<CompiledBarrier>,
}

impl<'a> GraphExecutor<'a> {
    fn new(device: &RenderDevice, graph: CompiledGraph<'a>) -> Result<Self> {
        let CompiledGraph {
            textures,
            usages,
            passes,
            final_barriers,
        } = graph;
        let mut images = Vec::with_capacity(textures.len());
        for (desc, usage) in textures.into_iter().zip(usages) {
            let Some(imported) = desc.imported else {
                let aspects = format_aspects(desc.format);
                let image = device.rhi.create_image(&ImageDesc {
                    label: "render graph texture",
                    dimension: dirk_rhi::ImageDimension::TwoD,
                    extent: Extent3d::new_2d(desc.width, desc.height),
                    format: desc.format,
                    usage,
                    mip_levels: 1,
                    array_layers: 1,
                    samples: desc.samples,
                })?;
                let view = device.rhi.create_image_view(&ImageViewDesc {
                    label: "render graph texture view",
                    image: &image,
                    view_type: ImageViewType::TwoD,
                    aspects,
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                })?;
                images.push(ResolvedImage {
                    image,
                    view,
                    aspects,
                });
                continue;
            };
            images.push(ResolvedImage {
                image: imported.image,
                view: imported.view,
                aspects: imported.aspects,
            });
        }
        Ok(Self {
            device: device.clone(),
            images,
            passes,
            final_barriers,
        })
    }

    fn execute(mut self, cmd: &mut CommandBuffer) -> Result<()> {
        for pass in &mut self.passes {
            record_barriers(cmd, &self.images, &pass.barriers)?;
            let has_rendering = !pass.colors.is_empty() || pass.depth.is_some();

            if has_rendering {
                let colors = pass
                    .colors
                    .iter()
                    .map(|attachment| dirk_rhi::ColorAttachment {
                        view: &self.images[attachment.handle.index()].view,
                        resolve: attachment
                            .resolve
                            .map(|handle| &self.images[handle.index()].view),
                        load: attachment.info.color_load(),
                        store: attachment.info.store,
                    })
                    .collect::<Vec<_>>();
                let depth = pass.depth.map(|(handle, info)| dirk_rhi::DepthAttachment {
                    view: &self.images[handle.index()].view,
                    depth_load: info.depth_load(),
                    depth_store: info.store,
                    stencil_load: info.stencil_load(),
                    stencil_store: info.store,
                });
                let extent = pass.extent.unwrap_or_else(|| Extent3d::new_2d(1, 1));
                cmd.rhi_mut().begin_rendering(&RenderingInfo {
                    label: &pass.name,
                    width: extent.width,
                    height: extent.height,
                    layer_count: 1,
                    color_attachments: &colors,
                    depth_attachment: depth,
                })?;
            }

            if let Some(callback) = pass.callback.take() {
                let context = PassContext {
                    device: &self.device,
                    images: &self.images,
                    declared: &pass.declared,
                };
                callback(cmd, &context)?;
            }
            if has_rendering {
                cmd.rhi_mut().end_rendering()?;
            }
        }
        record_barriers(cmd, &self.images, &self.final_barriers)?;
        Ok(())
    }
}

fn record_barriers(
    cmd: &mut CommandBuffer,
    images: &[ResolvedImage],
    barriers: &[CompiledBarrier],
) -> Result<()> {
    if barriers.is_empty() {
        return Ok(());
    }
    let barriers = barriers
        .iter()
        .map(|barrier| {
            let image = &images[barrier.handle.index()];
            ImageBarrier {
                image: &image.image,
                old_state: barrier.old_state,
                new_state: barrier.new_state,
                aspects: image.aspects,
                base_mip_level: barrier.range.base_mip_level,
                mip_level_count: barrier.range.mip_level_count,
                base_array_layer: barrier.range.base_array_layer,
                array_layer_count: barrier.range.array_layer_count,
                queue_transfer: None,
            }
        })
        .collect::<Vec<_>>();
    cmd.rhi_mut().barrier(&DependencyInfo {
        memory_barriers: &[],
        buffer_barriers: &[],
        image_barriers: &barriers,
    })?;
    Ok(())
}

fn format_aspects(format: TextureFormat) -> ImageAspects {
    match format {
        TextureFormat::Depth16Unorm | TextureFormat::Depth32Float => ImageAspects::DEPTH,
        TextureFormat::Depth24UnormStencil8 | TextureFormat::Depth32FloatStencil8 => {
            ImageAspects::DEPTH | ImageAspects::STENCIL
        }
        _ => ImageAspects::COLOR,
    }
}

#[allow(clippy::cast_possible_truncation)]
fn texture_handle(index: usize) -> TextureHandle {
    assert!(u32::try_from(index).is_ok());
    TextureHandle(index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_desc() -> TextureDesc {
        TextureDesc {
            width: 64,
            height: 64,
            format: TextureFormat::Bgra8Unorm,
            samples: SampleCount::One,
            imported: None,
        }
    }

    #[test]
    fn compile_transitions_attachment_to_copy_source() {
        let mut graph = RenderGraph::new();
        let color = graph.create_texture(color_desc());
        graph
            .add_pass("render")
            .write_color_attachment(color, AttachmentInfo::clear_color(0.0, 0.0, 0.0, 1.0));
        graph.add_pass("copy").read_transfer_src(color);

        let compiled = graph.compile();
        assert_eq!(
            compiled.passes[1].barriers[0].old_state,
            ImageState::ColorAttachment
        );
        assert_eq!(
            compiled.passes[1].barriers[0].new_state,
            ImageState::CopySource
        );
        assert_eq!(
            compiled.passes[1].barriers[0].range,
            SubresourceRange::WHOLE
        );
    }

    #[test]
    fn repeated_attachment_writes_keep_a_dependency() {
        assert!(barrier_needed(
            ImageState::ColorAttachment,
            ImageState::ColorAttachment
        ));
    }

    #[test]
    fn read_after_read_in_the_same_state_needs_no_barrier() {
        assert!(!barrier_needed(
            ImageState::ShaderRead,
            ImageState::ShaderRead
        ));
    }

    #[test]
    fn disjoint_mip_reads_only_initialize_the_read_span() {
        let mut graph = RenderGraph::new();
        let texture = graph.create_texture(color_desc());
        graph.add_pass("write mip 0").write_range(
            texture,
            TextureWrite::CopyDestination,
            mip_range(0),
        );
        graph
            .add_pass("read mip 1")
            .read_range(texture, TextureRead::CopySource, mip_range(1));

        let compiled = graph.compile();
        // No hazard with the written mip; reading untouched contents only
        // initializes the read span out of Undefined.
        assert_eq!(compiled.passes[1].barriers.len(), 1);
        assert_eq!(
            compiled.passes[1].barriers[0].old_state,
            ImageState::Undefined
        );
    }

    #[test]
    fn overlapping_mip_access_barriers_only_the_overlap() {
        let mut graph = RenderGraph::new();
        let texture = graph.create_texture(color_desc());
        graph.add_pass("write mip 0").write_range(
            texture,
            TextureWrite::CopyDestination,
            mip_range(0),
        );
        graph
            .add_pass("read mip 0")
            .read_range(texture, TextureRead::CopySource, mip_range(0));

        let compiled = graph.compile();
        let barrier = &compiled.passes[1].barriers[0];
        assert_eq!(barrier.old_state, ImageState::CopyDestination);
        assert_eq!(barrier.new_state, ImageState::CopySource);
        assert_eq!(barrier.range.base_mip_level, 0);
        assert_eq!(barrier.range.mip_level_count, 1);
    }

    #[test]
    fn whole_read_after_partial_write_barriers_each_tracked_range() {
        let mut graph = RenderGraph::new();
        let texture = graph.create_texture(color_desc());
        graph.add_pass("write mip 0").write_range(
            texture,
            TextureWrite::CopyDestination,
            mip_range(0),
        );
        graph
            .add_pass("read all")
            .read(texture, TextureRead::CopySource);

        let compiled = graph.compile();
        let barriers = &compiled.passes[1].barriers;
        assert_eq!(
            barriers.len(),
            2,
            "mip 0 transitions, remainder leaves Undefined"
        );
        let covered: Vec<_> = barriers
            .iter()
            .map(|barrier| (barrier.range.base_mip_level, barrier.old_state))
            .collect();
        assert!(covered.contains(&(0, ImageState::CopyDestination)));
        assert!(covered.contains(&(1, ImageState::Undefined)));
    }

    #[test]
    fn usage_is_derived_from_declared_accesses() {
        let mut graph = RenderGraph::new();
        let resolved = graph.create_texture(color_desc());
        let storage = graph.create_texture(color_desc());
        let msaa = graph.create_texture(color_desc());
        graph
            .add_pass("resolve")
            .write_color_attachment_with_resolve(
                msaa,
                resolved,
                AttachmentInfo::clear_color(0., 0., 0., 1.),
            );
        graph.add_pass("storage").write(
            storage,
            TextureWrite::Storage {
                stages: ShaderStages::COMPUTE,
            },
        );

        let compiled = graph.compile();
        assert_eq!(
            compiled.usages[resolved.index()],
            ImageUsages::COLOR_ATTACHMENT
        );
        assert_eq!(compiled.usages[storage.index()], ImageUsages::STORAGE);
    }

    fn mip_range(base: u32) -> SubresourceRange {
        SubresourceRange {
            base_mip_level: base,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        }
    }
}
