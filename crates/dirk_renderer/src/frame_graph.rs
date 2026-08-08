//! Backend-neutral render graph executed through the renderer RHI.

use dirk_rhi::{
    Color, CommandBuffer as _, DependencyInfo, Extent3d, Format, ImageAspects, ImageBarrier,
    ImageDesc, ImageState, ImageUsages, ImageViewDesc, ImageViewType, LoadOp, RenderingInfo,
    SampleCount, StoreOp,
};
use dirk_rhi_vulkan::{VulkanImage, VulkanImageView};

use crate::{
    Result,
    resources::{command_pool::CommandBuffer, device::RenderDevice},
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

/// Texture allocation and import metadata used while building a graph.
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub usage: ImageUsages,
    pub samples: SampleCount,
    pub imported: Option<ImportedTexture>,
}

/// Externally owned image and its graph-boundary states.
#[derive(Clone)]
pub struct ImportedTexture {
    pub image: VulkanImage,
    pub view: VulkanImageView,
    pub aspects: ImageAspects,
    pub initial_state: ImageState,
    pub final_state: ImageState,
}

/// Backend image resources resolved for a graph texture.
pub struct ResolvedImage {
    pub image: VulkanImage,
    pub view: VulkanImageView,
    pub aspects: ImageAspects,
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

#[derive(Clone)]
struct TextureUsage {
    handle: TextureHandle,
    state: ImageState,
    attachment: Option<AttachmentInfo>,
}

pub type PassCallback<'a> =
    Box<dyn FnOnce(&RenderDevice, &mut CommandBuffer, &[ResolvedImage]) -> Result<()> + 'a>;

struct PassNode<'a> {
    name: String,
    reads: Vec<TextureUsage>,
    writes: Vec<TextureUsage>,
    color_resolves: Vec<(TextureHandle, TextureHandle)>,
    callback: Option<PassCallback<'a>>,
}

pub struct PassBuilder<'graph, 'a> {
    pass: &'graph mut PassNode<'a>,
}

impl<'a> PassBuilder<'_, 'a> {
    pub fn write_color_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            state: ImageState::ColorAttachment,
            attachment: Some(info),
        });
        self
    }

    pub fn write_color_attachment_with_resolve(
        &mut self,
        handle: TextureHandle,
        resolve_handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.write_color_attachment(handle, info);
        self.pass.writes.push(TextureUsage {
            handle: resolve_handle,
            state: ImageState::ColorAttachment,
            attachment: None,
        });
        self.pass.color_resolves.push((handle, resolve_handle));
        self
    }

    pub fn write_depth_attachment(
        &mut self,
        handle: TextureHandle,
        info: AttachmentInfo,
    ) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            state: ImageState::DepthStencilAttachment,
            attachment: Some(info),
        });
        self
    }

    #[allow(unused)]
    pub fn read_texture_fragment(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.reads.push(TextureUsage {
            handle,
            state: ImageState::ShaderRead,
            attachment: None,
        });
        self
    }

    #[allow(unused)]
    pub fn read_texture_compute(&mut self, handle: TextureHandle) -> &mut Self {
        self.read_texture_fragment(handle)
    }

    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn read_transfer_src(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.reads.push(TextureUsage {
            handle,
            state: ImageState::CopySource,
            attachment: None,
        });
        self
    }

    #[cfg_attr(feature = "editor", allow(unused))]
    pub fn write_transfer_dst(&mut self, handle: TextureHandle) -> &mut Self {
        self.pass.writes.push(TextureUsage {
            handle,
            state: ImageState::CopyDestination,
            attachment: None,
        });
        self
    }

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
        let mut states = self
            .textures
            .iter()
            .map(|desc| {
                desc.imported
                    .as_ref()
                    .map_or(ImageState::Undefined, |imported| imported.initial_state)
            })
            .collect::<Vec<_>>();
        let mut compiled_passes = Vec::with_capacity(self.passes.len());

        for pass in self.passes {
            let mut barriers = Vec::new();
            for usage in pass.reads.iter().chain(&pass.writes) {
                let state = states[usage.handle.index()];
                if barrier_needed(state, usage.state) {
                    barriers.push(CompiledBarrier {
                        handle: usage.handle,
                        old_state: state,
                        new_state: usage.state,
                    });
                }
                states[usage.handle.index()] = usage.state;
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
                callback: pass.callback,
            });
        }

        let final_barriers = self
            .textures
            .iter()
            .enumerate()
            .filter_map(|(index, desc)| {
                let imported = desc.imported.as_ref()?;
                (states[index] != imported.final_state).then_some(CompiledBarrier {
                    handle: texture_handle(index),
                    old_state: states[index],
                    new_state: imported.final_state,
                })
            })
            .collect();

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

struct CompiledBarrier {
    handle: TextureHandle,
    old_state: ImageState,
    new_state: ImageState,
}

struct CompiledColorAttachment {
    handle: TextureHandle,
    info: AttachmentInfo,
    resolve: Option<TextureHandle>,
}

struct CompiledPass<'a> {
    name: String,
    barriers: Vec<CompiledBarrier>,
    colors: Vec<CompiledColorAttachment>,
    depth: Option<(TextureHandle, AttachmentInfo)>,
    extent: Option<Extent3d>,
    callback: Option<PassCallback<'a>>,
}

struct CompiledGraph<'a> {
    textures: Vec<TextureDesc>,
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
        let mut images = Vec::with_capacity(graph.textures.len());
        for desc in graph.textures {
            if let Some(imported) = desc.imported {
                images.push(ResolvedImage {
                    image: imported.image,
                    view: imported.view,
                    aspects: imported.aspects,
                });
                continue;
            }

            let aspects = format_aspects(desc.format);
            let image = device.rhi.create_image(&ImageDesc {
                label: "render graph texture",
                extent: Extent3d::new_2d(desc.width, desc.height),
                format: desc.format,
                usage: desc.usage,
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
        }
        Ok(Self {
            device: device.clone(),
            images,
            passes: graph.passes,
            final_barriers: graph.final_barriers,
        })
    }

    fn execute(mut self, cmd: &mut CommandBuffer) -> Result<()> {
        for pass in &mut self.passes {
            record_barriers(cmd, &self.images, &pass.barriers);
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
                cmd.rhi_mut().begin_rendering(&RenderingInfo {
                    label: &pass.name,
                    extent: pass.extent.unwrap_or_else(|| Extent3d::new_2d(1, 1)),
                    color_attachments: &colors,
                    depth_attachment: depth,
                })?;
            }

            if let Some(callback) = pass.callback.take() {
                callback(&self.device, cmd, &self.images)?;
            }
            if has_rendering {
                cmd.rhi_mut().end_rendering()?;
            }
        }
        record_barriers(cmd, &self.images, &self.final_barriers);
        Ok(())
    }
}

fn record_barriers(
    cmd: &mut CommandBuffer,
    images: &[ResolvedImage],
    barriers: &[CompiledBarrier],
) {
    if barriers.is_empty() {
        return;
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
                base_mip_level: 0,
                mip_level_count: u32::MAX,
                base_array_layer: 0,
                array_layer_count: u32::MAX,
            }
        })
        .collect::<Vec<_>>();
    cmd.rhi_mut().barrier(&DependencyInfo {
        image_barriers: &barriers,
    });
}

fn format_aspects(format: Format) -> ImageAspects {
    match format {
        Format::Depth16Unorm | Format::Depth32Float => ImageAspects::DEPTH,
        Format::Depth24UnormStencil8 | Format::Depth32FloatStencil8 => {
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
            format: Format::Bgra8Unorm,
            usage: ImageUsages::COLOR_ATTACHMENT | ImageUsages::COPY_SRC,
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
}
