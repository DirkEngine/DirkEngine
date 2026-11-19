//! A simple Vulkan render graph for ash + Vulkan 1.3 (dynamic rendering).

use ash::vk;

use crate::{
    Result,
    resources::{
        command_pool::CommandBuffer,
        device::RenderDevice,
        image::{Image, ImageCreateInfo},
    },
};

/// An opaque index into the graph's texture table.
/// These are "virtual" during graph construction – physical `VkImage`s are
/// assigned later at execution time (or immediately for imported resources).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(u32);

impl TextureHandle {
    /// Returns the texture table index represented by this handle.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Description used to create (or identify) a texture in the graph.
pub struct TextureDesc {
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// Vulkan image format.
    pub format: vk::Format,
    /// `ImageUsageFlags` covers all ways the texture will be used across the
    /// whole graph – the compiler needs this to allocate it correctly.
    pub usage: vk::ImageUsageFlags,
    /// Sample count used when creating the texture.
    pub samples: vk::SampleCountFlags,
    /// `Some` for externally-owned images such as swapchain images.
    /// `None` for transient images the graph creates and destroys itself.
    pub imported: Option<ImportedTexture>,
}

/// Carries the physical `VkImage`/`VkImageView` for resources that live
/// outside the graph (swapchain images being the canonical example).
#[derive(Clone, Copy)]
pub struct ImportedTexture {
    /// Externally-owned Vulkan image.
    pub image: vk::Image,
    /// Image view for [`Self::image`].
    pub view: vk::ImageView,
    /// Aspect mask covered by the imported image.
    pub aspect_flags: vk::ImageAspectFlags,
    /// Synchronization state the image is in before the first pass touches it.
    pub initial_state: TextureStateDesc,
    /// Synchronization state the image must be in after all passes execute.
    pub final_state: TextureStateDesc,
}

/// External texture synchronization state at a graph boundary.
#[derive(Clone, Copy)]
pub struct TextureStateDesc {
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

/// Resolved Vulkan handles for a graph texture.
///
/// This is what is to create attachments during rendering.
/// The `image` & `view` are **NOT** owned by this struct.
pub struct ResolvedImage {
    /// Vulkan image backing the graph texture.
    pub image: vk::Image,
    /// Image view for [`Self::image`].
    pub view: vk::ImageView,
    /// Aspect mask covered by the image.
    pub aspect_flags: vk::ImageAspectFlags,
}

/// Aggregates the load/store ops and clear value for a single attachment.
#[derive(Clone)]
pub struct AttachmentInfo {
    /// Load operation used when rendering begins.
    pub load_op: vk::AttachmentLoadOp,
    /// Store operation used when rendering ends.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear value used when [`Self::load_op`] is `CLEAR`.
    pub clear_value: vk::ClearValue,
}

impl AttachmentInfo {
    /// Clear to a solid colour, then store the result.
    #[must_use]
    pub fn clear_color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [r, g, b, a],
                },
            },
        }
    }

    /// Load the existing contents, then store the result (e.g. for additive
    /// blending passes after an initial clear pass).
    #[cfg_attr(
        not(feature = "editor"),
        expect(
            dead_code,
            reason = "Only the editor's overlay pass draws over earlier contents."
        )
    )]
    #[must_use]
    pub fn load_store() -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue::default(),
        }
    }

    /// Clear depth (and stencil) to the given values, then store.
    #[expect(
        dead_code,
        reason = "No pass clears depth yet; kept for the first depth pass."
    )]
    #[must_use]
    pub fn clear_depth(depth: f32, stencil: u32) -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }

    /// Discard the attachment at the end of the pass – saves bandwidth when
    /// the data is not needed afterwards (e.g. a depth buffer only used
    /// within one pass).
    #[must_use]
    pub fn clear_discard_depth(depth: f32, stencil: u32) -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

/// Internal description of how a single pass uses one texture.
/// The (layout, stage, access) triple is exactly what is needed to compute a
/// `VkImageMemoryBarrier2` from the previous state of the resource.
#[derive(Clone)]
struct TextureUsage {
    handle: TextureHandle,
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    /// Set when this usage is an attachment (load/store ops, clear value).
    attachment: Option<AttachmentInfo>,
}

/// Signature of the per-pass command-recording callback.
/// The callback receives:
///  - The ash `Device` for issuing Vulkan calls.
///  - The `CommandBuffer` to record into (already inside `vkCmdBeginRendering`
///    if the pass has attachments).
///  - `ResolvedResources` to look up `VkImage`/`VkImageView` for any handle.
pub type PassCallback<'a> =
    Box<dyn FnOnce(&RenderDevice, &CommandBuffer, &[ResolvedImage]) -> Result<()> + 'a>;

/// Internal graph node representing a single render pass.
struct PassNode<'a> {
    name: String,
    reads: Vec<TextureUsage>,
    writes: Vec<TextureUsage>,
    color_resolves: Vec<(TextureHandle, TextureHandle)>,
    callback: Option<PassCallback<'a>>,
}

/// Short-lived builder returned by `RenderGraph::add_pass`.
/// Borrows the pass node mutably so it can't outlive the graph.
pub struct PassBuilder<'graph, 'a> {
    pass: &'graph mut PassNode<'a>,
}

impl<'a> PassBuilder<'_, 'a> {
    /// Declare `handle` as a colour attachment written by this pass.
    pub fn write_color_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            attachment: Some(info),
        });
        self
    }

    /// Declare `handle` as a multisampled colour attachment which resolves into
    /// `resolve_handle` at the end of the pass.
    pub fn write_color_attachment_with_resolve(
        &mut self,
        handle: TextureHandle,
        resolve_handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.write_color_attachment(handle, info);
        self.pass.writes.push(TextureUsage {
            handle: resolve_handle,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            attachment: None,
        });
        self.pass.color_resolves.push((handle, resolve_handle));
        self
    }

    /// Declare `handle` as a depth attachment written by this pass.
    pub fn write_depth_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            // Both stages are required: early tests read depth for culling,
            // late tests write it after the fragment shader.
            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            attachment: Some(info),
        });
        self
    }

    /// Declare `handle` as sampled/read-only in a fragment shader.
    #[expect(
        dead_code,
        reason = "No pass samples an earlier pass's output in a fragment shader yet."
    )]
    pub fn read_texture_fragment(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.reads.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
            attachment: None,
        });
        self
    }

    /// Declare `handle` as sampled/read-only in a compute shader.
    #[expect(dead_code, reason = "No compute pass exists yet.")]
    pub fn read_texture_compute(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.reads.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ,
            attachment: None,
        });
        self
    }

    /// Declare `handle` as a transfer source.
    #[cfg_attr(
        all(feature = "editor", not(test)),
        expect(
            dead_code,
            reason = "The editor shows the viewport in egui instead of copying it to the swapchain."
        )
    )]
    pub fn read_transfer_src(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.reads.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_READ,
            attachment: None,
        });
        self
    }

    /// Declare `handle` as a transfer destination.
    #[cfg_attr(
        all(feature = "editor", not(test)),
        expect(
            dead_code,
            reason = "The editor shows the viewport in egui instead of copying it to the swapchain."
        )
    )]
    pub fn write_transfer_dst(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
            attachment: None,
        });
        self
    }

    /// Provide the command-recording callback for this pass.
    pub fn execute(&mut self, callback: PassCallback<'a>) {
        self.pass.callback = Some(callback);
    }
}

/// The render graph builder.  All methods are called in the *setup* phase
/// (before any Vulkan resources are touched).
pub struct RenderGraph<'a> {
    textures: Vec<TextureDesc>,
    passes: Vec<PassNode<'a>>,
}

impl<'a> RenderGraph<'a> {
    /// Creates an empty render graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// Register a *transient* texture that the graph owns and will
    /// allocate/destroy automatically.
    ///
    /// # Panics
    ///
    /// Panics if `desc` contains an imported texture. Use [`Self::import_texture`]
    /// for externally-owned images.
    pub fn create_texture(&mut self, desc: TextureDesc) -> TextureHandle {
        assert!(
            desc.imported.is_none(),
            "use import_texture for external images"
        );
        let handle = texture_handle(self.textures.len());
        self.textures.push(desc);
        handle
    }

    /// Register an *imported* texture (e.g. a swapchain image).
    /// The caller retains ownership; the graph only borrows it.
    ///
    /// # Panics
    ///
    /// Panics if `desc` does not contain an imported texture. Use
    /// [`Self::create_texture`] for transient images.
    pub fn import_texture(&mut self, desc: TextureDesc) -> TextureHandle {
        assert!(
            desc.imported.is_some(),
            "import_texture requires an ImportedTexture"
        );
        let handle = texture_handle(self.textures.len());
        self.textures.push(desc);
        handle
    }

    /// Begin building a new pass.  Returns a `PassBuilder` to declare
    /// resource usages and provide the callback.
    pub fn add_pass(&mut self, name: impl Into<String>) -> PassBuilder<'_, 'a> {
        let pass_index = self.passes.len();
        self.passes.push(PassNode {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            color_resolves: Vec::new(),
            callback: None,
        });
        PassBuilder {
            pass: &mut self.passes[pass_index],
        }
    }

    /// Compiles & runs the Graph
    ///
    /// # Errors
    ///
    /// Returns an error if transient image allocation fails or if a pass
    /// callback returns an error.
    pub fn run(self, device: &RenderDevice, cmd: &CommandBuffer) -> Result<()> {
        GraphExecutor::new(device, self.compile())?.execute(cmd)
    }

    /// Compile the graph: derive barriers and collect attachment metadata.
    /// Consumes `self`.
    fn compile(self) -> CompiledGraph<'a> {
        // Initialise per-resource state from the TextureDesc.
        let mut states: Vec<ResourceState> =
            self.textures.iter().map(ResourceState::from_desc).collect();

        let mut compiled_passes = Vec::with_capacity(self.passes.len());

        for pass in self.passes {
            // Iterate reads then writes.  For each usage, compare the desired
            // (layout, stage, access) against the current resource state and emit
            // a barrier if any transition is required.
            let mut pre_barriers = Vec::new();
            for usage in pass.reads.iter().chain(pass.writes.iter()) {
                let idx = usage.handle.0 as usize;
                let state = &states[idx];

                if barrier_needed(state, usage) {
                    pre_barriers.push(ImageBarrier {
                        handle: usage.handle,
                        old_layout: state.layout,
                        new_layout: usage.layout,
                        src_stage: state.stage,
                        dst_stage: usage.stage,
                        src_access: state.access,
                        dst_access: usage.access,
                    });
                }
            }

            // After the pass executes the resource is in its new state.
            for usage in pass.reads.iter().chain(pass.writes.iter()) {
                states[usage.handle.0 as usize] = ResourceState {
                    layout: usage.layout,
                    stage: usage.stage,
                    access: usage.access,
                };
            }

            let mut color_attachments: Vec<ColorAttachment> = Vec::new();
            let mut depth_attachment: Option<(TextureHandle, AttachmentInfo)> = None;
            let mut render_extent: Option<vk::Extent2D> = None;
            for usage in &pass.writes {
                if let Some(att) = &usage.attachment {
                    let desc = &self.textures[usage.handle.0 as usize];
                    match usage.layout {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
                            render_extent.get_or_insert(vk::Extent2D {
                                width: desc.width,
                                height: desc.height,
                            });
                            let resolve = pass
                                .color_resolves
                                .iter()
                                .find_map(|(src, dst)| (*src == usage.handle).then_some(*dst));
                            color_attachments.push(ColorAttachment {
                                handle: usage.handle,
                                info: att.clone(),
                                resolve,
                            });
                        }
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                        | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                            depth_attachment = Some((usage.handle, att.clone()));
                            // Use depth dimensions if there are no colour attachments.
                            render_extent.get_or_insert(vk::Extent2D {
                                width: desc.width,
                                height: desc.height,
                            });
                            // TODO: why no add to depth_attachments?
                        }
                        _ => {}
                    }
                }
            }

            compiled_passes.push(CompiledPass {
                name: pass.name,
                pre_barriers,
                color_attachments,
                depth_attachment,
                render_extent,
                callback: pass.callback,
            });
        }

        let mut final_barriers = Vec::new();
        for (idx, desc) in self.textures.iter().enumerate() {
            if let Some(imported) = &desc.imported {
                let state = &states[idx];
                let final_state = ResourceState::from_state_desc(imported.final_state);
                if state != &final_state {
                    final_barriers.push(ImageBarrier {
                        handle: texture_handle(idx),
                        old_layout: state.layout,
                        new_layout: final_state.layout,
                        src_stage: state.stage,
                        dst_stage: final_state.stage,
                        src_access: state.access,
                        dst_access: final_state.access,
                    });
                }
            }
        }

        CompiledGraph {
            textures: self.textures,
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

/// The last-known synchronisation state of a single texture.
/// Tracks exactly the three fields needed to fill out a
/// `VkImageMemoryBarrier2`.
#[derive(Clone, PartialEq, Eq)]
struct ResourceState {
    layout: vk::ImageLayout,
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

impl ResourceState {
    fn from_desc(desc: &TextureDesc) -> Self {
        desc.imported.as_ref().map_or(
            Self {
                layout: vk::ImageLayout::UNDEFINED,
                stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                access: vk::AccessFlags2::empty(),
            },
            |imported| Self::from_state_desc(imported.initial_state),
        )
    }

    fn from_state_desc(desc: TextureStateDesc) -> Self {
        Self {
            layout: desc.layout,
            stage: desc.stage,
            access: desc.access,
        }
    }
}

/// A fully-resolved image memory barrier, minus the physical `VkImage`
/// (that is substituted at execution time once physical resources exist).
struct ImageBarrier {
    handle: TextureHandle,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    dst_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
}

/// Compiled representation of a single pass: barriers to emit before it and
/// the attachment metadata needed to call `vkCmdBeginRendering`.
struct CompiledPass<'a> {
    #[expect(
        dead_code,
        reason = "Not read until the renderer builds graph metadata from pass names."
    )]
    name: String,
    /// Barriers to record immediately before this pass.
    pre_barriers: Vec<ImageBarrier>,
    /// Ordered list of colour attachments (in the order written to the pass).
    color_attachments: Vec<ColorAttachment>,
    /// Optional depth attachment.
    depth_attachment: Option<(TextureHandle, AttachmentInfo)>,
    /// Render area derived from the first colour attachment (or depth if none).
    render_extent: Option<vk::Extent2D>,
    /// The user-provided recording callback.
    callback: Option<PassCallback<'a>>,
}

struct ColorAttachment {
    handle: TextureHandle,
    info: AttachmentInfo,
    resolve: Option<TextureHandle>,
}

/// Output of the compilation phase.
struct CompiledGraph<'a> {
    textures: Vec<TextureDesc>,
    passes: Vec<CompiledPass<'a>>,
    /// Barriers emitted *after* the last pass – primarily used to transition
    /// imported textures to their required final state.
    final_barriers: Vec<ImageBarrier>,
}

/// Returns `true` when a `VkImageMemoryBarrier2` is required.
///
/// A barrier is needed when:
/// - The layout needs to change (always requires a barrier), OR
/// - The previous or upcoming access includes a write (WAW / RAW / WAR hazard).
///
/// Note: read-after-read with the same layout and no writes does *not* need a
/// barrier, which is why we gate on the write mask.
fn barrier_needed(state: &ResourceState, usage: &TextureUsage) -> bool {
    if state.layout != usage.layout {
        return true;
    }
    let write_mask = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags2::SHADER_WRITE
        | vk::AccessFlags2::TRANSFER_WRITE
        | vk::AccessFlags2::MEMORY_WRITE;
    state.access.intersects(write_mask) || usage.access.intersects(write_mask)
}

/// Owns the physical `VkImage` / `VkImageView` / `VkDeviceMemory` for each
/// resource in a compiled graph and drives command-buffer recording.
///
/// # Lifetime note
/// `GraphExecutor` is designed to be created per-frame (or per-graph) and
/// destroyed with `destroy()` once the GPU has finished using the resources.
/// A production implementation would replace the naive per-image allocations
/// here with a proper transient allocator (e.g. VMA's
/// `VMA_MEMORY_USAGE_GPU_ONLY` with aliasing hints from the compiled graph).
struct GraphExecutor<'a> {
    device: RenderDevice,
    transient_images: Vec<Image>,
    /// One entry per `TextureHandle` index.
    images: Vec<ResolvedImage>,
    passes: Vec<CompiledPass<'a>>,
    final_barriers: Vec<ImageBarrier>,
}

impl<'a> GraphExecutor<'a> {
    /// Allocate all transient resources and bind imported ones.
    fn new(device: &RenderDevice, graph: CompiledGraph<'a>) -> Result<Self> {
        let mut transient_images = Vec::new();
        let mut images = Vec::with_capacity(graph.textures.len());

        for desc in graph.textures {
            if let Some(imported) = desc.imported {
                images.push(ResolvedImage {
                    image: imported.image,
                    view: imported.view,
                    aspect_flags: imported.aspect_flags,
                });
            } else {
                // TODO: transient allocator
                let aspect_flags = if matches!(
                    desc.format,
                    vk::Format::D16_UNORM
                        | vk::Format::D32_SFLOAT
                        | vk::Format::D24_UNORM_S8_UINT
                        | vk::Format::D16_UNORM_S8_UINT
                        | vk::Format::D32_SFLOAT_S8_UINT
                ) {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };

                let info = ImageCreateInfo {
                    size: vk::Extent2D {
                        width: desc.width,
                        height: desc.height,
                    },
                    format: desc.format,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: desc.usage,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    mip_levels: 1,
                    num_samples: desc.samples,
                    aspect_flags,
                };
                let image = Image::create_image(device, &info)?;

                images.push(ResolvedImage {
                    image: image.image(),
                    view: image.view(),
                    aspect_flags,
                });
                transient_images.push(image);
            }
        }

        Ok(Self {
            device: device.clone(),
            transient_images,
            images,
            passes: graph.passes,
            final_barriers: graph.final_barriers,
        })
    }

    /// Record the entire graph into `cmd`.
    ///
    /// `cmd` must be in the recording state and must *not* already be inside
    /// a render pass or dynamic rendering scope.
    fn execute(mut self, cmd: &CommandBuffer) -> Result<()> {
        debug_assert!(self.transient_images.len() <= self.images.len());

        for pass in &mut self.passes {
            if !pass.pre_barriers.is_empty() {
                let image_barriers: Vec<vk::ImageMemoryBarrier2> = pass
                    .pre_barriers
                    .iter()
                    .map(|b| {
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(b.src_stage)
                            .src_access_mask(b.src_access)
                            .dst_stage_mask(b.dst_stage)
                            .dst_access_mask(b.dst_access)
                            .old_layout(b.old_layout)
                            .new_layout(b.new_layout)
                            .image(self.images[b.handle.index()].image)
                            .subresource_range(subresource_range_for(
                                self.images[b.handle.index()].aspect_flags,
                            ))
                    })
                    .collect();

                cmd.pipeline_barrier2(
                    &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
                );
            }

            // Replaces vkBeginRenderPass/vkEndRenderPass entirely.
            // Attachments are specified inline – no VkRenderPass or
            // VkFramebuffer objects are needed.
            let has_rendering =
                !pass.color_attachments.is_empty() || pass.depth_attachment.is_some();

            if has_rendering {
                let extent = pass.render_extent.unwrap_or(vk::Extent2D {
                    width: 1,
                    height: 1,
                });

                let color_infos: Vec<vk::RenderingAttachmentInfo> = pass
                    .color_attachments
                    .iter()
                    .map(|att| {
                        let mut info = vk::RenderingAttachmentInfo::default()
                            .image_view(self.images[att.handle.index()].view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(att.info.load_op)
                            .store_op(att.info.store_op)
                            .clear_value(att.info.clear_value);
                        if let Some(resolve) = att.resolve {
                            info = info
                                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                                .resolve_image_view(self.images[resolve.index()].view)
                                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                        }
                        info
                    })
                    .collect();

                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_infos);

                let depth_info_storage;
                if let Some((h, att)) = &pass.depth_attachment {
                    depth_info_storage = vk::RenderingAttachmentInfo::default()
                        .image_view(self.images[h.index()].view)
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(att.load_op)
                        .store_op(att.store_op)
                        .clear_value(att.clear_value);
                    rendering_info = rendering_info.depth_attachment(&depth_info_storage);
                }

                cmd.begin_rendering(&rendering_info);
            }

            if let Some(callback) = pass.callback.take() {
                callback(&self.device, cmd, &self.images)?;
            }

            if has_rendering {
                cmd.end_rendering();
            }
        }

        if !self.final_barriers.is_empty() {
            let image_barriers: Vec<vk::ImageMemoryBarrier2> = self
                .final_barriers
                .iter()
                .map(|b| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(b.src_stage)
                        .src_access_mask(b.src_access)
                        .dst_stage_mask(b.dst_stage)
                        .dst_access_mask(b.dst_access)
                        .old_layout(b.old_layout)
                        .new_layout(b.new_layout)
                        .image(self.images[b.handle.index()].image)
                        .subresource_range(subresource_range_for(
                            self.images[b.handle.index()].aspect_flags,
                        ))
                })
                .collect();

            cmd.pipeline_barrier2(
                &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
            );
        }

        Ok(())
    }
}

/// Build an `ImageSubresourceRange` that covers the whole image.
fn subresource_range_for(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The assert above checks that the index fits in a u32."
)]
fn texture_handle(index: usize) -> TextureHandle {
    assert!(u32::try_from(index).is_ok());
    TextureHandle(index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_desc(usage: vk::ImageUsageFlags, samples: vk::SampleCountFlags) -> TextureDesc {
        TextureDesc {
            width: 64,
            height: 64,
            format: vk::Format::B8G8R8A8_UNORM,
            usage,
            samples,
            imported: None,
        }
    }

    fn texture_state(
        layout: vk::ImageLayout,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> TextureStateDesc {
        TextureStateDesc {
            layout,
            stage,
            access,
        }
    }

    fn imported_color_desc(
        usage: vk::ImageUsageFlags,
        initial_state: TextureStateDesc,
        final_state: TextureStateDesc,
    ) -> TextureDesc {
        TextureDesc {
            width: 64,
            height: 64,
            format: vk::Format::B8G8R8A8_UNORM,
            usage,
            samples: vk::SampleCountFlags::TYPE_1,
            imported: Some(ImportedTexture {
                image: vk::Image::null(),
                view: vk::ImageView::null(),
                aspect_flags: vk::ImageAspectFlags::COLOR,
                initial_state,
                final_state,
            }),
        }
    }

    #[test]
    fn compile_transitions_color_attachment_to_transfer_src() {
        let mut graph = RenderGraph::new();
        let scene_color = graph.create_texture(color_desc(
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::SampleCountFlags::TYPE_1,
        ));

        graph
            .add_pass("scene")
            .write_color_attachment(scene_color, AttachmentInfo::clear_color(0., 0., 0., 1.));
        graph.add_pass("copy").read_transfer_src(scene_color);

        let compiled = graph.compile();
        let copy_pass = &compiled.passes[1];
        let barrier = copy_pass
            .pre_barriers
            .iter()
            .find(|barrier| barrier.handle == scene_color)
            .expect("copy pass should transition scene color");

        assert_eq!(
            barrier.old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(barrier.new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(barrier.dst_stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(barrier.dst_access, vk::AccessFlags2::TRANSFER_READ);
    }

    #[test]
    fn compile_transitions_imported_swapchain_to_transfer_dst_then_present() {
        let mut graph = RenderGraph::new();
        let swapchain = graph.import_texture(imported_color_desc(
            vk::ImageUsageFlags::TRANSFER_DST,
            texture_state(
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            texture_state(
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
        ));

        graph.add_pass("copy").write_transfer_dst(swapchain);

        let compiled = graph.compile();
        let copy_barrier = compiled.passes[0]
            .pre_barriers
            .iter()
            .find(|barrier| barrier.handle == swapchain)
            .expect("copy pass should transition swapchain");
        assert_eq!(copy_barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(
            copy_barrier.new_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
        assert_eq!(copy_barrier.dst_stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(copy_barrier.dst_access, vk::AccessFlags2::TRANSFER_WRITE);

        let final_barrier = compiled
            .final_barriers
            .iter()
            .find(|barrier| barrier.handle == swapchain)
            .expect("final barrier should transition swapchain for present");
        assert_eq!(
            final_barrier.old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
        assert_eq!(final_barrier.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
        assert_eq!(
            final_barrier.dst_stage,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE
        );
        assert_eq!(final_barrier.dst_access, vk::AccessFlags2::empty());
    }

    #[test]
    fn compile_transitions_imported_viewport_output_to_shader_read_final_state() {
        let mut graph = RenderGraph::new();
        let viewport = graph.import_texture(imported_color_desc(
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            texture_state(
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            ),
            texture_state(
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
        ));

        graph
            .add_pass("scene")
            .write_color_attachment(viewport, AttachmentInfo::clear_color(0., 0., 0., 1.));

        let compiled = graph.compile();
        let final_barrier = compiled
            .final_barriers
            .iter()
            .find(|barrier| barrier.handle == viewport)
            .expect("final barrier should make viewport shader-readable");
        assert_eq!(
            final_barrier.old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            final_barrier.new_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            final_barrier.dst_stage,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert_eq!(final_barrier.dst_access, vk::AccessFlags2::SHADER_READ);
    }

    #[test]
    fn compile_reused_imported_viewport_starts_from_tracked_shader_read_state() {
        let mut graph = RenderGraph::new();
        let viewport = graph.import_texture(imported_color_desc(
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            texture_state(
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            texture_state(
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
        ));

        graph
            .add_pass("scene")
            .write_color_attachment(viewport, AttachmentInfo::clear_color(0., 0., 0., 1.));

        let compiled = graph.compile();
        let scene_barrier = compiled.passes[0]
            .pre_barriers
            .iter()
            .find(|barrier| barrier.handle == viewport)
            .expect("scene pass should transition from tracked viewport state");
        assert_eq!(
            scene_barrier.old_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            scene_barrier.src_stage,
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert_eq!(scene_barrier.src_access, vk::AccessFlags2::SHADER_READ);
        assert_eq!(
            scene_barrier.new_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn compile_msaa_scene_resolves_to_regular_scene_image() {
        let mut graph = RenderGraph::new();
        let msaa_color = graph.create_texture(color_desc(
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::SampleCountFlags::TYPE_4,
        ));
        let scene_color = graph.create_texture(color_desc(
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::SampleCountFlags::TYPE_1,
        ));

        graph.add_pass("scene").write_color_attachment_with_resolve(
            msaa_color,
            scene_color,
            AttachmentInfo::clear_color(0., 0., 0., 1.),
        );

        let compiled = graph.compile();
        let attachment = compiled.passes[0]
            .color_attachments
            .iter()
            .find(|attachment| attachment.handle == msaa_color)
            .expect("scene pass should contain the MSAA color attachment");

        assert_eq!(attachment.resolve, Some(scene_color));
    }
}
