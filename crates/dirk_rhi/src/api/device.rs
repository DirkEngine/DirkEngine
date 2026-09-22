use super::{
    Arc, BufferInfo, GpuBindGroup, GpuBindGroupLayout, GpuBuffer, GpuGraphicsPipeline, GpuImage,
    GpuImageView, GpuPipelineLayout, GpuSampler, GpuShader, GpuSurface, GpuSurfaceFrame,
    GpuSwapchain, GroupMetadata, Mutex, Object, PipelineMetadata, RecordedCommands, SamplerInfo,
    ViewMetadata, lock,
};
use crate::ImageViewType as V;
use crate::{
    Api, Backend, BindGroupDesc, BindingResource, BindingType, BufferDesc, GraphicsPipelineDesc,
    ImageDesc, ImageDimension, ImageInfo, ImageSubresourceRange, ImageViewDesc,
    InvalidResourceKind as Ir, NativeFence as _, Result, SamplerDesc, ShaderDesc, ShaderStage,
    SurfaceCreateInfo, TextureFormat,
};

/// Device owner with one graphics queue, one transfer queue, and deferred destruction.
pub struct Rhi<B: Backend> {
    pub(super) device: Arc<Device<B>>,
    pub(super) graphics: super::Queue<B, super::Graphics>,
    pub(super) transfer: super::Queue<B, super::CopyQueue>,
}
pub(super) struct Device<B: Backend> {
    pub(super) backend: Arc<B>,
    pub(super) gate: Arc<Mutex<()>>,
    pub(super) state: Mutex<DeviceState<B>>,
    pub(super) pools: Arc<Mutex<PoolCache<B>>>,
}
pub(super) struct DeviceState<B: Backend> {
    pub(super) pending: Vec<Arc<Work<B>>>,
    pub(super) lost: bool,
    pub(super) cycle: u64,
    pub(super) cycles: std::collections::VecDeque<Vec<Arc<Work<B>>>>,
}
impl<B: Backend> Api for Rhi<B> {
    type Buffer = GpuBuffer<B>;
    type Image = GpuImage<B>;
    type ImageView = GpuImageView<B>;
    type Sampler = GpuSampler<B>;
    type Shader = GpuShader<B>;
    type BindGroupLayout = GpuBindGroupLayout<B>;
    type BindGroup = GpuBindGroup<B>;
    type PipelineLayout = GpuPipelineLayout<B>;
    type GraphicsPipeline = GpuGraphicsPipeline<B>;
    type CommandPool = ();
    type CommandBuffer = RecordedCommands<B>;
    type Fence = Completion<B>;
    type TimelineSemaphore = ();
    type Surface = GpuSurface<B>;
    type Swapchain = GpuSwapchain<B>;
    type SurfaceFrame = GpuSurfaceFrame<B>;
}
impl<B: Backend> Rhi<B> {
    /// Creates the platform-selected native backend and its queues.
    pub fn new(info: &crate::RhiCreateInfo<'_>) -> Result<Self> {
        // SAFETY: creation only borrows valid platform providers and retains no borrowed handles.
        let backend = Arc::new(unsafe { B::new(info)? });
        let device = Arc::new(Device {
            backend,
            gate: Arc::new(Mutex::new(())),
            pools: Arc::new(Mutex::new(std::collections::HashMap::new())),
            state: Mutex::new(DeviceState {
                pending: Vec::new(),
                lost: false,
                cycle: 0,
                cycles: std::collections::VecDeque::new(),
            }),
        });
        Ok(Self {
            graphics: super::Queue::new(device.clone())?,
            transfer: super::Queue::new(device.clone())?,
            device,
        })
    }
    /// Portable limits and optional features.
    #[must_use]
    pub fn capabilities(&self) -> crate::Capabilities {
        self.device.backend.capabilities()
    }
    /// Per-format support, queried before allocation or command recording.
    #[must_use]
    pub fn format_capabilities(&self, format: TextureFormat) -> crate::FormatCapabilities {
        self.device.backend.format_capabilities(format)
    }
    /// Supported attachment sample counts.
    #[must_use]
    pub fn supported_sample_counts(
        &self,
        format: TextureFormat,
        usages: crate::ImageUsages,
    ) -> crate::SampleCounts {
        self.device.backend.supported_sample_counts(format, usages)
    }
    /// Backend's preferred supported depth formats.
    #[must_use]
    pub fn supported_depth_formats(&self) -> &[TextureFormat] {
        self.device.backend.supported_depth_formats()
    }
    pub(super) fn object<T, M>(raw: T, metadata: M) -> Object<B, T, M> {
        Object {
            raw,
            metadata,
            marker: std::marker::PhantomData,
        }
    }

    /// Allocates a buffer with fixed size, usage, and host domain.
    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<GpuBuffer<B>> {
        if desc.size == 0
            || desc.size > self.capabilities().limits.max_buffer_size
            || desc.usage.is_empty()
        {
            return Err(Ir::OutOfRange.into());
        }
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_buffer(desc)? },
            BufferInfo {
                size: desc.size,
                usage: desc.usage,
                memory: desc.memory,
            },
        ))
    }
    /// Allocates a supported image; metadata is retained for imports and checks.
    pub fn create_image(&self, desc: &ImageDesc<'_>) -> Result<GpuImage<B>> {
        let info = ImageInfo::from(desc);
        self.validate_image(info)?;
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_image(desc)? },
            info,
        ))
    }
    fn validate_image(&self, info: ImageInfo) -> Result<()> {
        let e = info.extent;
        let limits = self.capabilities().limits;
        let dimension_limit = if info.dimension == ImageDimension::ThreeD {
            limits.max_image_dimension_3d
        } else {
            limits.max_image_dimension_2d
        };
        if e.width == 0
            || e.height == 0
            || e.depth == 0
            || info.mip_levels == 0
            || info.array_layers == 0
        {
            return Err(Ir::Empty.into());
        }
        if e.width.max(e.height).max(e.depth) > dimension_limit
            || info.array_layers > limits.max_image_array_layers
            || info.mip_levels > 32 - e.width.max(e.height).max(e.depth).leading_zeros()
        {
            return Err(Ir::OutOfRange.into());
        }
        if (info.dimension == ImageDimension::ThreeD && info.array_layers != 1)
            || (info.dimension != ImageDimension::ThreeD && e.depth != 1)
            || (info.dimension == ImageDimension::Cube
                && (e.width != e.height || !info.array_layers.is_multiple_of(6)))
            || (info.samples != crate::SampleCount::One
                && (info.mip_levels != 1 || info.dimension != ImageDimension::TwoD))
        {
            return Err(Ir::Mismatch.into());
        }
        if info.usage.is_empty()
            || !self.format_capabilities(info.format).supports(info.usage)
            || !self
                .supported_sample_counts(info.format, info.usage)
                .supports(info.samples)
        {
            return Err(crate::UnsupportedOperation::TextureFormat(info.format).into());
        }
        Ok(())
    }
    /// Creates a non-owning view with a checked subresource selection.
    pub fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<GpuImageView<B>> {
        let info = desc.image.description();
        let range = ImageSubresourceRange {
            aspects: desc.aspects,
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
        }
        .resolve(&info)?;
        let compatible = match desc.view_type {
            V::TwoD => info.dimension != ImageDimension::ThreeD && range.array_layer_count == 1,
            V::TwoDArray => info.dimension != ImageDimension::ThreeD,
            V::ThreeD => info.dimension == ImageDimension::ThreeD,
            V::Cube => {
                info.dimension == ImageDimension::Cube
                    && range.array_layer_count == 6
                    && range.base_array_layer.is_multiple_of(6)
            }
            V::CubeArray => {
                info.dimension == ImageDimension::Cube
                    && range.array_layer_count.is_multiple_of(6)
                    && range.base_array_layer.is_multiple_of(6)
            }
        };
        if !compatible {
            return Err(Ir::Mismatch.into());
        }
        let raw = ImageViewDesc::<B> {
            label: desc.label,
            image: desc.image.raw(),
            view_type: desc.view_type,
            aspects: range.aspects,
            base_mip_level: range.base_mip_level,
            mip_level_count: range.mip_level_count,
            base_array_layer: range.base_array_layer,
            array_layer_count: range.array_layer_count,
        };
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_image_view(&raw)? },
            ViewMetadata { image: info, range },
        ))
    }
    /// Creates a default full view of an image.
    pub fn view(&self, image: &GpuImage<B>) -> Result<GpuImageView<B>> {
        let info = image.description();
        let view_type = match (info.dimension, info.array_layers) {
            (ImageDimension::ThreeD, _) => crate::ImageViewType::ThreeD,
            (ImageDimension::Cube, 6) => crate::ImageViewType::Cube,
            (ImageDimension::Cube, _) => crate::ImageViewType::CubeArray,
            (_, 1) => crate::ImageViewType::TwoD,
            _ => crate::ImageViewType::TwoDArray,
        };
        self.create_image_view(&ImageViewDesc {
            label: "full image view",
            image,
            view_type,
            aspects: info.format.aspects(),
            base_mip_level: 0,
            mip_level_count: info.mip_levels,
            base_array_layer: 0,
            array_layer_count: info.array_layers,
        })
    }
    /// Creates a sampler, rejecting unsupported required settings.
    pub fn create_sampler(&self, desc: &SamplerDesc<'_>) -> Result<GpuSampler<B>> {
        if desc.max_anisotropy == 0
            || desc.max_anisotropy > self.capabilities().max_sampler_anisotropy
            || !desc.lod_min.is_finite()
            || !desc.lod_max.is_finite()
            || desc.lod_min > desc.lod_max
            || desc.lod_min < 0.0
        {
            return Err(Ir::OutOfRange.into());
        }
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_sampler(desc)? },
            SamplerInfo {
                linear: [desc.mag_filter, desc.min_filter, desc.mip_filter]
                    .contains(&crate::FilterMode::Linear),
            },
        ))
    }
    /// Imports a shader generated by the trusted engine shader pipeline.
    ///
    /// # Safety
    /// Native shader code must be valid for its declared stage and entry point,
    /// agree with pipeline bindings, and respect resource bounds. Native binaries
    /// are not a sandbox for untrusted shader programs. Compilation/reflection lives
    /// in the shader build layer; this boundary makes that trust explicit.
    pub unsafe fn create_shader(&self, desc: &ShaderDesc<'_>) -> Result<GpuShader<B>> {
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_shader(desc)? },
            desc.stage,
        ))
    }
    /// Creates an immutable group layout with unique sorted bindings.
    pub fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDesc<'_>,
    ) -> Result<GpuBindGroupLayout<B>> {
        let mut entries = desc.entries.to_vec();
        entries.sort_by_key(|e| e.binding);
        if entries
            .windows(2)
            .any(|pair| pair[0].binding == pair[1].binding)
            || entries.iter().any(|e| e.visibility.is_empty())
        {
            return Err(Ir::Mismatch.into());
        }
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe {
                self.device
                    .backend
                    .create_bind_group_layout(&crate::BindGroupLayoutDesc {
                        label: desc.label,
                        entries: &entries,
                    })?
            },
            entries,
        ))
    }
    /// Creates non-owning bindings with checked types, ranges, and usages.
    pub fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<GpuBindGroup<B>> {
        if desc.entries.len() != desc.layout.info().len() {
            return Err(Ir::Mismatch.into());
        }

        let mut native = Vec::new();
        for layout in desc.layout.info() {
            let mut matches = desc.entries.iter().filter(|e| e.binding == layout.binding);
            let entry = matches.next().ok_or(Ir::Mismatch)?;
            if matches.next().is_some() {
                return Err(Ir::Mismatch.into());
            }
            let resource = self.prepare_binding(entry, layout.ty)?;
            native.push(crate::BindGroupEntry {
                binding: entry.binding,
                resource,
            });
        }
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe {
                self.device.backend.create_bind_group(&BindGroupDesc {
                    label: desc.label,
                    layout: desc.layout.raw(),
                    entries: &native,
                })?
            },
            GroupMetadata {
                layout: desc.layout.info().clone(),
            },
        ))
    }
    fn prepare_binding<'a>(
        &self,
        entry: &'a crate::BindGroupEntry<'_, Self>,
        ty: BindingType,
    ) -> Result<BindingResource<'a, B>> {
        let prepared = match (&entry.resource, ty) {
            (
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                },
                ty @ (BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. }),
            ) => {
                let range = crate::BufferRange {
                    offset: *offset,
                    size: *size,
                }
                .resolve(buffer.size())?;
                let uniform = matches!(ty, BindingType::UniformBuffer { .. });
                let caps = self.capabilities();
                let alignment = if uniform {
                    caps.min_uniform_buffer_offset_alignment
                } else {
                    caps.min_storage_buffer_offset_alignment
                }
                .max(1);
                let usage = if uniform {
                    crate::BufferUsages::UNIFORM
                } else {
                    crate::BufferUsages::STORAGE
                };
                if !range.offset.is_multiple_of(alignment) || !buffer.info().usage.contains(usage) {
                    return Err(Ir::Mismatch.into());
                }
                BindingResource::Buffer {
                    buffer: buffer.raw(),
                    offset: range.offset,
                    size: range.size,
                }
            }
            (BindingResource::SampledImage { view, sampler }, BindingType::SampledImage) => {
                let info = view.info().image;
                if !info.usage.contains(crate::ImageUsages::SAMPLED)
                    || info.samples != crate::SampleCount::One
                    || (sampler.info().linear && !self.format_capabilities(info.format).filterable)
                {
                    return Err(Ir::Mismatch.into());
                }
                BindingResource::SampledImage {
                    view: view.raw(),
                    sampler: sampler.raw(),
                }
            }
            (BindingResource::StorageImage(view), BindingType::StorageImage) => {
                if !view
                    .info()
                    .image
                    .usage
                    .contains(crate::ImageUsages::STORAGE)
                {
                    return Err(Ir::Mismatch.into());
                }
                BindingResource::StorageImage(view.raw())
            }
            _ => return Err(Ir::Mismatch.into()),
        };
        Ok(prepared)
    }
    /// Creates a pipeline layout retaining its group declarations.
    pub fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDesc<'_, Self>,
    ) -> Result<GpuPipelineLayout<B>> {
        if desc.bind_group_layouts.len() > self.capabilities().limits.max_bind_groups as usize {
            return Err(Ir::OutOfRange.into());
        }
        let groups: Vec<_> = desc
            .bind_group_layouts
            .iter()
            .map(|l| l.info().as_slice())
            .collect();
        for stage in [
            ShaderStage::Vertex,
            ShaderStage::Fragment,
            ShaderStage::Compute,
        ] {
            crate::BindingMap::new(&groups, stage)?.validate(self.capabilities().limits)?;
        }
        let native: Vec<_> = desc.bind_group_layouts.iter().map(|l| l.raw()).collect();
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe {
                self.device
                    .backend
                    .create_pipeline_layout(&crate::PipelineLayoutDesc {
                        label: desc.label,
                        bind_group_layouts: &native,
                    })?
            },
            desc.bind_group_layouts
                .iter()
                .map(|l| l.info().clone())
                .collect(),
        ))
    }
    /// Creates a graphics pipeline with checked stages, formats, and required features.
    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<GpuGraphicsPipeline<B>> {
        if *desc.vertex.info() != ShaderStage::Vertex {
            return Err(Ir::Mismatch.into());
        }
        if let Some(fragment) = desc.fragment
            && *fragment.info() != ShaderStage::Fragment
        {
            return Err(Ir::Mismatch.into());
        }
        if desc.primitive_restart.is_some()
            && desc.raster.topology != crate::PrimitiveTopology::TriangleStrip
        {
            return Err(Ir::Mismatch.into());
        }
        let caps = self.capabilities();
        if desc.color_targets.len() > caps.limits.max_color_attachments as usize
            || desc.vertex_buffers.len() > caps.limits.max_vertex_buffers as usize
        {
            return Err(Ir::OutOfRange.into());
        }
        if desc.depth_bias.clamp != 0.0 && !caps.depth_bias_clamp {
            return Err(crate::UnsupportedOperation::Capability("depth bias clamping").into());
        }
        for target in desc.color_targets {
            let support = self.format_capabilities(target.format);
            if !support.supports(crate::ImageUsages::COLOR_ATTACHMENT)
                || (target.blend.is_some() && !support.blendable)
                || !self
                    .supported_sample_counts(target.format, crate::ImageUsages::COLOR_ATTACHMENT)
                    .supports(desc.samples)
            {
                return Err(crate::UnsupportedOperation::TextureFormat(target.format).into());
            }
        }
        if let Some(depth) = desc.depth
            && !self
                .supported_sample_counts(depth.format, crate::ImageUsages::DEPTH_STENCIL_ATTACHMENT)
                .supports(desc.samples)
        {
            return Err(crate::UnsupportedOperation::TextureFormat(depth.format).into());
        }
        let raw = GraphicsPipelineDesc::<B> {
            label: desc.label,
            layout: desc.layout.raw(),
            vertex: desc.vertex.raw(),
            fragment: desc.fragment.map(Object::raw),
            vertex_buffers: desc.vertex_buffers,
            raster: desc.raster,
            color_targets: desc.color_targets,
            depth: desc.depth,
            depth_bias: desc.depth_bias,
            primitive_restart: desc.primitive_restart,
            alpha_to_coverage: desc.alpha_to_coverage,
            samples: desc.samples,
        };
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_graphics_pipeline(&raw)? },
            PipelineMetadata {
                primitive_restart: desc.primitive_restart,
                colors: desc.color_targets.to_vec(),
                depth: desc.depth,
                samples: desc.samples,
            },
        ))
    }
    /// Creates a retained presentation target.
    pub fn create_surface(&self, info: SurfaceCreateInfo) -> Result<GpuSurface<B>> {
        let _gate = lock(&self.device.gate)?;
        Ok(Self::object(
            unsafe { self.device.backend.create_surface(info)? },
            (),
        ))
    }
    /// Waits for this device's submitted work and releases completed ownership.
    pub fn wait_idle(&mut self) -> Result<()> {
        self.device.wait_idle()
    }
    /// Seals the current cycle, waits for submitted work, and drains sealed retirements.
    /// All recordings from this cycle must already be submitted or discarded.
    /// Use this when no future frames will advance the retirement queue.
    pub fn flush(&mut self) -> Result<()> {
        self.finish_cycle()?;
        self.wait_idle()
    }
    /// Seals one submission cycle, including all graphics and transfer work.
    /// Commands recorded in this cycle must be submitted before it ends.
    pub fn finish_cycle(&mut self) -> Result<()> {
        {
            let mut state = lock(&self.device.state)?;
            self.device.backend.seal_garbage();
            let work = std::mem::take(&mut state.pending);
            state.cycles.push_back(work);
            state.cycle += 1;
        }
        self.collect_garbage()
    }
    /// Collects old buckets only after all queues through that cycle have completed.
    pub fn collect_garbage(&mut self) -> Result<()> {
        let _gate = lock(&self.device.gate)?;
        let mut state = lock(&self.device.state)?;
        while state.cycles.len() >= 3 {
            for work in &state.cycles[0] {
                match work.wait(0) {
                    Ok(()) => {}
                    Err(crate::Error::Timeout) => return Ok(()),
                    Err(error) => return Err(error),
                }
            }
            state.cycles.pop_front();
            unsafe {
                self.device.backend.collect_garbage()?;
            }
        }
        Ok(())
    }
}

pub(super) struct Payload<B: Backend> {
    pub(super) commands: Vec<(B::CommandBuffer, B::CommandPool)>,
}
pub(super) type PoolCache<B> = std::collections::HashMap<
    crate::QueueType,
    Vec<(<B as Api>::CommandBuffer, <B as Api>::CommandPool)>,
>;
pub(super) struct Work<B: Backend> {
    pub(super) timeline: B::TimelineSemaphore,
    pub(super) value: u64,
    pub(super) pools: Arc<Mutex<PoolCache<B>>>,
    pub(super) queue: crate::QueueType,
    pub(super) fence: B::Fence,
    pub(super) payload: Mutex<Option<Payload<B>>>,
    pub(super) _backend: Arc<B>,
}
impl<B: Backend> Work<B> {
    pub(super) fn wait(&self, timeout: u64) -> Result<()> {
        let mut payload = lock(&self.payload)?;
        if payload.is_none() {
            return Ok(());
        }
        unsafe {
            self.fence.wait(timeout)?;
        }
        if let Some(done) = payload.take() {
            lock(&self.pools)?
                .entry(self.queue)
                .or_default()
                .extend(done.commands);
        }
        Ok(())
    }
}
/// Completion evidence for a submitted batch. Dropping it never abandons GPU ownership.
pub struct Completion<B: Backend>(pub(super) Arc<Work<B>>);
impl<B: Backend> Clone for Completion<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<B: Backend> Completion<B> {
    /// Waits for the submitted batch to finish.
    pub fn wait(&self, timeout_ns: u64) -> Result<()> {
        self.0.wait(timeout_ns)
    }
    /// Queries completion without blocking.
    pub fn is_complete(&self) -> Result<bool> {
        match self.wait(0) {
            Ok(()) => Ok(true),
            Err(crate::Error::Timeout) => Ok(false),
            Err(error) => Err(error),
        }
    }
}
impl<B: Backend> Drop for Device<B> {
    fn drop(&mut self) {
        // This owner is the last entry point for submissions. Native device loss is fatal.
        let _ = unsafe { self.backend.wait_idle() };
        if let Ok(state) = self.state.get_mut() {
            state.pending.clear();
            state.cycles.clear();
        }
        let _ = unsafe { self.backend.wait_idle() };
    }
}

impl<B: Backend> Device<B> {
    pub(super) fn wait_idle(&self) -> Result<()> {
        let _gate = lock(&self.gate)?;
        unsafe {
            self.backend.wait_idle()?;
        }
        let mut state = lock(&self.state)?;
        for work in &state.pending {
            work.wait(u64::MAX)?;
        }
        state.pending.clear();
        for _ in state.cycles.drain(..) {
            unsafe {
                self.backend.collect_garbage()?;
            }
        }
        // Reclaim command allocations released by dropping completed work above.
        unsafe {
            self.backend.wait_idle()?;
        }
        Ok(())
    }
}
