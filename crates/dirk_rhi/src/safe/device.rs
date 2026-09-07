use super::{
    Arc, AtomicU64, AtomicUsize, BufferInfo, GpuBindGroup, GpuBindGroupLayout, GpuBuffer,
    GpuGraphicsPipeline, GpuImage, GpuImageView, GpuPipelineLayout, GpuSampler, GpuShader,
    GpuSurface, GpuSurfaceFrame, GpuSwapchain, GpuTimeline, GroupMetadata, ImageMetadata, Mutex,
    Object, ObjectInner, Ordering, OwnedBinding, PipelineMetadata, RecordedCommands, SamplerInfo,
    ViewMetadata, identity, lock,
};
use crate::ImageViewType as V;
use crate::{
    Api, Backend, BindGroupDesc, BindingResource, BindingType, BufferDesc, Fence as _,
    GraphicsPipelineDesc, ImageDesc, ImageDimension, ImageInfo, ImageSubresourceRange,
    ImageViewDesc, InvalidResourceKind as Ir, Result, SamplerDesc, ShaderDesc, ShaderStage,
    SurfaceCreateInfo, TextureFormat,
};
use std::collections::HashMap;

/// Shared safe device. Clones share one submission and host-access domain.
pub struct Rhi<B: Backend>(pub(super) Arc<Device<B>>);
pub(super) struct Device<B: Backend> {
    pub(super) backend: Arc<B>,
    pub(super) gate: Arc<Mutex<()>>,
    pub(super) state: Mutex<DeviceState<B>>,
}
pub(super) struct DeviceState<B: Backend> {
    pub(super) pending: Vec<Arc<Work<B>>>,
    pub(super) image_lifetimes: HashMap<u64, std::sync::Weak<AtomicUsize>>,
    pub(super) images: HashMap<u64, Vec<crate::ResourceAccess>>,
    pub(super) lost: bool,
}
impl<B: Backend> Clone for Rhi<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
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
    type TimelineSemaphore = GpuTimeline<B>;
    type Surface = GpuSurface<B>;
    type Swapchain = GpuSwapchain<B>;
    type SurfaceFrame = GpuSurfaceFrame<B>;
}
impl<B: Backend> Rhi<B> {
    /// Borrows the backend for integration with an external native API.
    ///
    /// # Safety
    /// The caller must externally synchronize native operations against all RHI
    /// operations, retain their resources through completion, and preserve tracked
    /// resource states. External work is not covered by RHI completion tokens.
    #[must_use]
    pub unsafe fn native(&self) -> &B {
        &self.0.backend
    }

    /// Creates the selected native backend and a safe ownership domain.
    pub fn new(info: &crate::RhiCreateInfo<'_>) -> Result<Self> {
        // SAFETY: creation only borrows valid platform providers and retains no borrowed handles.
        let backend = Arc::new(unsafe { B::new(info)? });
        Ok(Self(Arc::new(Device {
            backend,
            gate: Arc::new(Mutex::new(())),
            state: Mutex::new(DeviceState {
                pending: Vec::new(),
                images: HashMap::new(),
                image_lifetimes: HashMap::new(),
                lost: false,
            }),
        })))
    }
    /// Portable limits and optional features. Native queues are serialized by this initial safe scheduler.
    #[must_use]
    pub fn capabilities(&self) -> crate::Capabilities {
        let mut caps = self.0.backend.capabilities();
        caps.dedicated_compute_queue = false;
        caps.dedicated_copy_queue = false;
        caps
    }
    /// Per-format support, queried before allocation or command recording.
    #[must_use]
    pub fn format_capabilities(&self, format: TextureFormat) -> crate::FormatCapabilities {
        self.0.backend.format_capabilities(format)
    }
    /// Supported attachment sample counts.
    #[must_use]
    pub fn supported_sample_counts(
        &self,
        format: TextureFormat,
        usages: crate::ImageUsages,
    ) -> crate::SampleCounts {
        self.0.backend.supported_sample_counts(format, usages)
    }
    /// Backend's preferred supported depth formats.
    #[must_use]
    pub fn supported_depth_formats(&self) -> &[TextureFormat] {
        self.0.backend.supported_depth_formats()
    }
    pub(super) fn object<T, M>(&self, raw: T, metadata: M) -> Object<B, T, M> {
        Object(Arc::new(ObjectInner {
            raw,
            metadata,
            backend: self.0.backend.clone(),
            gate: self.0.gate.clone(),
            id: identity(),
            busy: Arc::new(AtomicUsize::new(0)),
        }))
    }
    /// Allocates a buffer with fixed size, usage, and host domain.
    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Result<GpuBuffer<B>> {
        if desc.size == 0
            || desc.size > self.capabilities().limits.max_buffer_size
            || desc.usage.is_empty()
        {
            return Err(Ir::OutOfRange.into());
        }
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_buffer(desc)? },
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_image(desc)? },
            ImageMetadata {
                allocation: info,
                frame: None,
            },
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
    /// Creates a view retaining its image and resolved selection.
    pub fn create_image_view(&self, desc: &ImageViewDesc<'_, Self>) -> Result<GpuImageView<B>> {
        desc.image.require_device(&self.0.backend)?;
        desc.image.require_live()?;
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_image_view(&raw)? },
            ViewMetadata {
                image: desc.image.clone(),
                range,
            },
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_sampler(desc)? },
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(unsafe { self.0.backend.create_shader(desc)? }, desc.stage))
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe {
                self.0
                    .backend
                    .create_bind_group_layout(&crate::BindGroupLayoutDesc {
                        label: desc.label,
                        entries: &entries,
                    })?
            },
            entries,
        ))
    }
    /// Creates a group with checked ranges and retained resource ownership.
    pub fn create_bind_group(&self, desc: &BindGroupDesc<'_, Self>) -> Result<GpuBindGroup<B>> {
        desc.layout.require_device(&self.0.backend)?;
        if desc.entries.len() != desc.layout.info().len() {
            return Err(Ir::Mismatch.into());
        }
        let mut owned = Vec::new();
        let mut native = Vec::new();
        for layout in desc.layout.info() {
            let mut matches = desc.entries.iter().filter(|e| e.binding == layout.binding);
            let entry = matches.next().ok_or(Ir::Mismatch)?;
            if matches.next().is_some() {
                return Err(Ir::Mismatch.into());
            }
            let (resource, retained) = self.prepare_binding(entry, layout.ty)?;
            native.push(crate::BindGroupEntry {
                binding: entry.binding,
                resource,
            });
            owned.push((entry.binding, retained));
        }
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe {
                self.0.backend.create_bind_group(&BindGroupDesc {
                    label: desc.label,
                    layout: desc.layout.raw(),
                    entries: &native,
                })?
            },
            GroupMetadata {
                layout: desc.layout.clone(),
                entries: owned,
            },
        ))
    }
    fn prepare_binding<'a>(
        &self,
        entry: &'a crate::BindGroupEntry<'_, Self>,
        ty: BindingType,
    ) -> Result<(BindingResource<'a, B>, OwnedBinding<B>)> {
        let prepared = match (&entry.resource, ty) {
            (
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                },
                ty @ (BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. }),
            ) => {
                buffer.require_device(&self.0.backend)?;
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
                (
                    BindingResource::Buffer {
                        buffer: buffer.raw(),
                        offset: range.offset,
                        size: range.size,
                    },
                    OwnedBinding::Buffer((*buffer).clone(), range, ty),
                )
            }
            (BindingResource::SampledImage { view, sampler }, BindingType::SampledImage) => {
                view.require_device(&self.0.backend)?;
                sampler.require_device(&self.0.backend)?;
                view.info().image.require_live()?;
                let info = view.info().image.description();
                if !info.usage.contains(crate::ImageUsages::SAMPLED)
                    || info.samples != crate::SampleCount::One
                    || (sampler.info().linear && !self.format_capabilities(info.format).filterable)
                {
                    return Err(Ir::Mismatch.into());
                }
                (
                    BindingResource::SampledImage {
                        view: view.raw(),
                        sampler: sampler.raw(),
                    },
                    OwnedBinding::Sampled((*view).clone(), (*sampler).clone()),
                )
            }
            (BindingResource::StorageImage(view), BindingType::StorageImage) => {
                view.require_device(&self.0.backend)?;
                view.info().image.require_live()?;
                if !view
                    .info()
                    .image
                    .description()
                    .usage
                    .contains(crate::ImageUsages::STORAGE)
                {
                    return Err(Ir::Mismatch.into());
                }
                (
                    BindingResource::StorageImage(view.raw()),
                    OwnedBinding::Storage((*view).clone()),
                )
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
        for layout in desc.bind_group_layouts {
            layout.require_device(&self.0.backend)?;
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe {
                self.0
                    .backend
                    .create_pipeline_layout(&crate::PipelineLayoutDesc {
                        label: desc.label,
                        bind_group_layouts: &native,
                    })?
            },
            desc.bind_group_layouts
                .iter()
                .map(|l| (*l).clone())
                .collect(),
        ))
    }
    /// Creates a graphics pipeline with checked stages, formats, and required features.
    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc<'_, Self>,
    ) -> Result<GpuGraphicsPipeline<B>> {
        desc.layout.require_device(&self.0.backend)?;
        desc.vertex.require_device(&self.0.backend)?;
        if *desc.vertex.info() != ShaderStage::Vertex {
            return Err(Ir::Mismatch.into());
        }
        if let Some(fragment) = desc.fragment {
            fragment.require_device(&self.0.backend)?;
            if *fragment.info() != ShaderStage::Fragment {
                return Err(Ir::Mismatch.into());
            }
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
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_graphics_pipeline(&raw)? },
            PipelineMetadata {
                layout: desc.layout.clone(),
                vertex: desc.vertex.clone(),
                fragment: desc.fragment.cloned(),
                colors: desc.color_targets.to_vec(),
                depth: desc.depth,
                samples: desc.samples,
                vertices: desc
                    .vertex_buffers
                    .iter()
                    .map(|v| (u64::from(v.stride), v.step_mode, v.attributes.to_vec()))
                    .collect(),
            },
        ))
    }
    /// Creates a retained presentation target.
    pub fn create_surface(&self, info: SurfaceCreateInfo) -> Result<GpuSurface<B>> {
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(unsafe { self.0.backend.create_surface(info)? }, ()))
    }
    /// Creates a monotonic timeline. Signals are validated before submission.
    pub fn create_timeline_semaphore(&self, initial: u64) -> Result<GpuTimeline<B>> {
        let _gate = lock(&self.0.gate)?;
        Ok(self.object(
            unsafe { self.0.backend.create_timeline_semaphore(initial)? },
            AtomicU64::new(initial),
        ))
    }
    /// Waits for this device's submitted work and releases completed ownership.
    pub fn wait_idle(&self) -> Result<()> {
        let _gate = lock(&self.0.gate)?;
        unsafe {
            self.0.backend.wait_idle()?;
        }
        let mut state = lock(&self.0.state)?;
        for work in &state.pending {
            work.wait(u64::MAX)?;
        }
        state.pending.clear();
        Ok(())
    }
    /// Nonblocking collection based on actual completion, never frame counts.
    pub fn collect_garbage(&self) -> Result<()> {
        let _gate = lock(&self.0.gate)?;
        let mut state = lock(&self.0.state)?;
        let mut completed = Vec::new();
        for (index, work) in state.pending.iter().enumerate() {
            match work.wait(0) {
                Ok(()) => completed.push(index),
                Err(crate::Error::Timeout) => {}
                Err(error) => return Err(error),
            }
        }
        for index in completed.into_iter().rev() {
            state.pending.remove(index);
        }
        state
            .image_lifetimes
            .retain(|_, lifetime| lifetime.strong_count() != 0);
        let live: std::collections::HashSet<_> = state.image_lifetimes.keys().copied().collect();
        state.images.retain(|id, _| live.contains(id));
        unsafe { self.0.backend.collect_garbage() }
    }
}

pub(super) struct Retained {
    pub(super) _object: Arc<dyn Send + Sync>,
    pub(super) busy: Arc<AtomicUsize>,
}
pub(super) struct Payload<B: Backend> {
    pub(super) commands: Vec<(B::CommandBuffer, B::CommandPool)>,
    pub(super) retained: Vec<Retained>,
}
pub(super) struct Work<B: Backend> {
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
            for resource in &done.retained {
                resource.busy.fetch_sub(1, Ordering::Release);
            }
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
    /// Waits and makes retained buffers available for host access.
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
        // No safe handles can submit through this owner anymore. Keep the native
        // device and payloads alive if device loss prevents proving completion.
        if let Ok(state) = self.state.get_mut() {
            for work in state.pending.drain(..) {
                if work.wait(u64::MAX).is_err() {
                    std::mem::forget(work);
                }
            }
        }
    }
}
