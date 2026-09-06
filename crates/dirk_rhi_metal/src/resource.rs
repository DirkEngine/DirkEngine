use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use dirk_rhi::{
    BindGroupDesc, BindGroupLayoutDesc, BindingResource, BindingType, Buffer, BufferDesc, Fence,
    GraphicsPipelineDesc, ImageAspects, ImageDesc, ImageDimension, ImageUsages, ImageViewDesc,
    ImageViewType, InvalidResourceKind as Ir, MemoryDomain, PipelineLayoutDesc, Result,
    SamplerDesc, ShaderDesc, ShaderSource, ShaderStage, ShaderStages, TimelineSemaphore,
};
use metal::{
    CompileOptions, DepthStencilDescriptor, DepthStencilState, Function, MTLColorWriteMask,
    MTLResourceOptions, MTLSamplerMipFilter, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    RenderPipelineDescriptor, RenderPipelineState, SamplerDescriptor, SamplerState, SharedEvent,
    StencilDescriptor, Texture, TextureDescriptor, VertexDescriptor,
};

use crate::{backend::Context, backend_error, convert};

pub(crate) const VERTEX_BUFFER_BASE: u64 = 16;

/// Metal buffer and its RHI allocation metadata.
#[derive(Clone)]
pub struct MetalBuffer {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: metal::Buffer,
    pub(crate) size: u64,
    pub(crate) memory: MemoryDomain,
    host_access: Arc<parking_lot::Mutex<()>>,
}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalBuffer")
            .field("size", &self.size)
            .field("memory", &self.memory)
            .finish_non_exhaustive()
    }
}

impl MetalBuffer {
    pub(crate) fn create(context: &Arc<Context>, desc: &BufferDesc<'_>) -> Result<Self> {
        if desc.size == 0 {
            return Err(dirk_rhi::Error::from(Ir::Empty));
        }
        let options = match desc.memory {
            MemoryDomain::Device => MTLResourceOptions::StorageModePrivate,
            MemoryDomain::Upload | MemoryDomain::Readback => MTLResourceOptions::StorageModeShared,
        } | MTLResourceOptions::HazardTrackingModeTracked;
        let raw = context.device.new_buffer(desc.size, options);
        raw.set_label(desc.label);
        Ok(Self {
            context: context.clone(),
            raw,
            size: desc.size,
            memory: desc.memory,
            host_access: Arc::new(parking_lot::Mutex::new(())),
        })
    }
}

impl Buffer for MetalBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let _guard = self.host_access.lock();
        if self.memory == MemoryDomain::Device {
            return Err(dirk_rhi::InvalidResourceKind::NotHostAccessible.into());
        }
        let length =
            u64::try_from(data.len()).map_err(|error| dirk_rhi::Error::Backend(error.into()))?;
        if offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(dirk_rhi::InvalidResourceKind::OutOfRange.into());
        }
        let offset =
            usize::try_from(offset).map_err(|_| dirk_rhi::InvalidResourceKind::OutOfRange)?;
        // SAFETY: Bounds are checked above and shared Metal buffer memory is
        // host visible for the lifetime of `self`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.raw.contents().cast::<u8>().add(offset),
                data.len(),
            );
        }
        Ok(())
    }

    fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        let _guard = self.host_access.lock();
        if self.memory != MemoryDomain::Readback {
            return Err(Ir::NotHostAccessible.into());
        }
        let length = u64::try_from(data.len()).map_err(backend_error)?;
        if offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(Ir::OutOfRange.into());
        }
        let offset = usize::try_from(offset).map_err(|_| Ir::OutOfRange)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.contents().cast::<u8>().add(offset),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        Ok(())
    }
}

/// Metal texture.
#[derive(Clone)]
pub struct MetalImage {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: Texture,
    pub(crate) format: dirk_rhi::TextureFormat,
    pub(crate) mip_levels: u32,
    pub(crate) array_layers: u32,
}

impl std::fmt::Debug for MetalImage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalImage")
            .field("format", &self.format)
            .field("mip_levels", &self.mip_levels)
            .field("array_layers", &self.array_layers)
            .finish_non_exhaustive()
    }
}

impl MetalImage {
    pub(crate) fn create(context: &Arc<Context>, desc: &ImageDesc<'_>) -> Result<Self> {
        if desc.extent.width == 0
            || desc.extent.height == 0
            || desc.extent.depth == 0
            || desc.mip_levels == 0
            || desc.array_layers == 0
        {
            return Err(dirk_rhi::Error::from(Ir::Empty));
        }
        if desc.array_layers > 1 && desc.samples != dirk_rhi::SampleCount::One {
            return Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
                "Metal does not support multisampled texture arrays"
            )));
        }
        if desc.usage.contains(ImageUsages::TRANSIENT_ATTACHMENT)
            && (!desc.usage.contains(ImageUsages::COLOR_ATTACHMENT)
                && !desc.usage.contains(ImageUsages::DEPTH_STENCIL_ATTACHMENT)
                || desc.usage.contains(ImageUsages::COPY_SRC)
                || desc.usage.contains(ImageUsages::COPY_DST)
                || desc.usage.contains(ImageUsages::SAMPLED)
                || desc.usage.contains(ImageUsages::STORAGE))
        {
            return Err(Ir::Mismatch.into());
        }
        let (texture_type, array_length) = match desc.dimension {
            ImageDimension::TwoD if desc.extent.depth == 1 => {
                let ty = if desc.array_layers > 1 {
                    MTLTextureType::D2Array
                } else if desc.samples != dirk_rhi::SampleCount::One {
                    MTLTextureType::D2Multisample
                } else {
                    MTLTextureType::D2
                };
                (ty, desc.array_layers)
            }
            ImageDimension::ThreeD if desc.array_layers == 1 => (MTLTextureType::D3, 1),
            ImageDimension::Cube
                if desc.extent.depth == 1 && desc.array_layers.is_multiple_of(6) =>
            {
                let cubes = desc.array_layers / 6;
                let ty = if cubes == 1 {
                    MTLTextureType::Cube
                } else {
                    MTLTextureType::CubeArray
                };
                (ty, cubes)
            }
            _ => return Err(Ir::Mismatch.into()),
        };
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(texture_type);
        descriptor.set_pixel_format(convert::format(desc.format));
        descriptor.set_width(u64::from(desc.extent.width));
        descriptor.set_height(u64::from(desc.extent.height));
        descriptor.set_depth(u64::from(desc.extent.depth));
        descriptor.set_mipmap_level_count(u64::from(desc.mip_levels));
        descriptor.set_array_length(u64::from(array_length));
        descriptor.set_sample_count(convert::samples(desc.samples));
        descriptor.set_storage_mode(if desc.usage.contains(ImageUsages::TRANSIENT_ATTACHMENT) {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        });
        let mut usage = MTLTextureUsage::Unknown;
        if desc.usage.contains(ImageUsages::SAMPLED) {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if desc.usage.contains(ImageUsages::STORAGE) {
            usage |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.contains(ImageUsages::COLOR_ATTACHMENT)
            || desc.usage.contains(ImageUsages::DEPTH_STENCIL_ATTACHMENT)
        {
            usage |= MTLTextureUsage::RenderTarget;
        }
        descriptor.set_usage(usage);
        let raw = context.device.new_texture(&descriptor);
        raw.set_label(desc.label);
        Ok(Self {
            context: context.clone(),
            raw,
            format: desc.format,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
        })
    }

    pub(crate) fn surface(
        context: &Arc<Context>,
        raw: Texture,
        format: dirk_rhi::TextureFormat,
    ) -> Self {
        Self {
            context: context.clone(),
            raw,
            format,
            mip_levels: 1,
            array_layers: 1,
        }
    }
}

/// Metal texture view.
#[derive(Clone)]
pub struct MetalImageView {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: Texture,
    pub(crate) aspects: ImageAspects,
}

impl std::fmt::Debug for MetalImageView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalImageView")
            .field("aspects", &self.aspects)
            .finish_non_exhaustive()
    }
}

impl MetalImageView {
    /// Multisampled textures cannot create Metal texture views, so their only
    /// representable RHI view reuses the source texture.
    fn reuses_source_texture(
        source_type: MTLTextureType,
        view_type: ImageViewType,
        array_layer_count: u32,
    ) -> Result<bool> {
        match source_type {
            MTLTextureType::D2Multisample => match view_type {
                ImageViewType::TwoD if array_layer_count == 1 => Ok(true),
                ImageViewType::TwoD
                | ImageViewType::TwoDArray
                | ImageViewType::ThreeD
                | ImageViewType::Cube
                | ImageViewType::CubeArray => Err(dirk_rhi::Error::from(Ir::Mismatch)),
            },
            MTLTextureType::D2MultisampleArray | MTLTextureType::D3 => {
                Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
                    "the RHI cannot represent this Metal texture view type"
                )))
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &ImageViewDesc<'_, crate::MetalBackend>,
    ) -> Result<Self> {
        if !Arc::ptr_eq(context, &desc.image.context) {
            return Err(Ir::ForeignInstance.into());
        }
        if desc
            .base_mip_level
            .checked_add(desc.mip_level_count)
            .is_none_or(|end| end > desc.image.mip_levels)
            || desc
                .base_array_layer
                .checked_add(desc.array_layer_count)
                .is_none_or(|end| end > desc.image.array_layers)
        {
            return Err(Ir::OutOfRange.into());
        }
        let reuses_source = Self::reuses_source_texture(
            desc.image.raw.texture_type(),
            desc.view_type,
            desc.array_layer_count,
        )?;
        let raw = if reuses_source {
            desc.image.raw.clone()
        } else {
            let texture_type = match desc.view_type {
                ImageViewType::TwoD if desc.array_layer_count == 1 => MTLTextureType::D2,
                ImageViewType::TwoD | ImageViewType::TwoDArray => MTLTextureType::D2Array,
                ImageViewType::ThreeD => MTLTextureType::D3,
                ImageViewType::Cube if desc.array_layer_count == 6 => MTLTextureType::Cube,
                ImageViewType::Cube | ImageViewType::CubeArray
                    if desc.array_layer_count.is_multiple_of(6)
                        && desc.base_array_layer.is_multiple_of(6) =>
                {
                    MTLTextureType::CubeArray
                }
                ImageViewType::Cube | ImageViewType::CubeArray => {
                    return Err(Ir::Mismatch.into());
                }
            };
            desc.image.raw.new_texture_view_from_slice(
                convert::format(desc.image.format),
                texture_type,
                metal::NSRange::new(
                    u64::from(desc.base_mip_level),
                    u64::from(desc.mip_level_count),
                ),
                metal::NSRange::new(
                    u64::from(desc.base_array_layer),
                    u64::from(desc.array_layer_count),
                ),
            )
        };
        raw.set_label(desc.label);
        Ok(Self {
            context: context.clone(),
            raw,
            aspects: desc.aspects,
        })
    }

    pub(crate) fn surface(context: &Arc<Context>, raw: Texture) -> Self {
        Self {
            context: context.clone(),
            raw,
            aspects: ImageAspects::COLOR,
        }
    }
}

/// Metal sampler state.
#[derive(Clone)]
pub struct MetalSampler {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: SamplerState,
}

impl std::fmt::Debug for MetalSampler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("MetalSampler")
    }
}

impl MetalSampler {
    pub(crate) fn create(context: &Arc<Context>, desc: &SamplerDesc<'_>) -> Self {
        let descriptor = SamplerDescriptor::new();
        descriptor.set_label(desc.label);
        descriptor.set_min_filter(convert::min_mag_filter(desc.min_filter));
        descriptor.set_mag_filter(convert::min_mag_filter(desc.mag_filter));
        descriptor.set_mip_filter(if desc.lod_max <= desc.lod_min {
            MTLSamplerMipFilter::NotMipmapped
        } else {
            convert::mip_filter(desc.mip_filter)
        });
        descriptor.set_address_mode_s(convert::address_mode(desc.address_u));
        descriptor.set_address_mode_t(convert::address_mode(desc.address_v));
        descriptor.set_address_mode_r(convert::address_mode(desc.address_w));
        descriptor.set_max_anisotropy(u64::from(desc.max_anisotropy.clamp(1, 16)));
        descriptor.set_lod_min_clamp(desc.lod_min);
        descriptor.set_lod_max_clamp(desc.lod_max);
        Self {
            context: context.clone(),
            raw: context.device.new_sampler(&descriptor),
        }
    }
}

/// Compiled MSL shader function.
#[derive(Clone)]
pub struct MetalShader {
    pub(crate) context: Arc<Context>,
    pub(crate) function: Function,
    pub(crate) stage: ShaderStage,
}

impl MetalShader {
    pub(crate) fn create(context: &Arc<Context>, desc: &ShaderDesc<'_>) -> Result<Self> {
        let ShaderSource::Msl(source) = desc.source else {
            return Err(
                dirk_rhi::UnsupportedOperation::ShaderSource(desc.source.language()).into(),
            );
        };
        let library = context
            .device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(backend_error)?;
        library.set_label(desc.label);
        let function = library
            .get_function(desc.entry, None)
            .map_err(backend_error)?;
        Ok(Self {
            context: context.clone(),
            function,
            stage: desc.stage,
        })
    }
}

/// Metal bind-group layout metadata.
#[derive(Clone)]
pub struct MetalBindGroupLayout {
    pub(crate) context: Arc<Context>,
    pub(crate) entries: Arc<[dirk_rhi::BindGroupLayoutEntry]>,
}

impl MetalBindGroupLayout {
    pub(crate) fn create(context: &Arc<Context>, desc: &BindGroupLayoutDesc<'_>) -> Result<Self> {
        let mut entries = desc.entries.to_vec();
        entries.sort_unstable_by_key(|entry| entry.binding);
        if entries
            .windows(2)
            .any(|pair| pair[0].binding == pair[1].binding)
        {
            return Err(Ir::Mismatch.into());
        }
        Ok(Self {
            context: context.clone(),
            entries: entries.into(),
        })
    }
}

#[derive(Clone)]
pub(crate) enum OwnedBinding {
    Buffer {
        buffer: MetalBuffer,
        offset: u64,
    },
    SampledImage {
        view: MetalImageView,
        sampler: MetalSampler,
    },
    StorageImage(MetalImageView),
}

/// Metal resources associated with one bind group.
#[derive(Clone)]
pub struct MetalBindGroup {
    pub(crate) layout: MetalBindGroupLayout,
    pub(crate) entries: Arc<[(u32, OwnedBinding)]>,
}

impl MetalBindGroup {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &BindGroupDesc<'_, crate::MetalBackend>,
    ) -> Result<Self> {
        require_context(context, &desc.layout.context)?;
        let mut entries = Vec::with_capacity(desc.entries.len());
        for entry in desc.entries {
            let layout_entry = desc
                .layout
                .entries
                .iter()
                .find(|layout| layout.binding == entry.binding)
                .ok_or_else(|| dirk_rhi::Error::from(Ir::Mismatch))?;
            let resource = match (&entry.resource, layout_entry.ty) {
                (
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    },
                    BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. },
                ) => {
                    require_context(context, &buffer.context)?;
                    if offset
                        .checked_add(*size)
                        .is_none_or(|end| end > buffer.size)
                    {
                        return Err(dirk_rhi::Error::from(Ir::OutOfRange));
                    }
                    OwnedBinding::Buffer {
                        buffer: (*buffer).clone(),
                        offset: *offset,
                    }
                }
                (BindingResource::SampledImage { view, sampler }, BindingType::SampledImage) => {
                    require_context(context, &view.context)?;
                    require_context(context, &sampler.context)?;
                    OwnedBinding::SampledImage {
                        view: (*view).clone(),
                        sampler: (*sampler).clone(),
                    }
                }
                (BindingResource::StorageImage(view), BindingType::StorageImage) => {
                    require_context(context, &view.context)?;
                    OwnedBinding::StorageImage((*view).clone())
                }
                _ => {
                    return Err(dirk_rhi::Error::from(Ir::Mismatch));
                }
            };
            entries.push((entry.binding, resource));
        }
        if entries.len() != desc.layout.entries.len() {
            return Err(dirk_rhi::Error::from(Ir::Mismatch));
        }
        entries.sort_unstable_by_key(|entry| entry.0);
        Ok(Self {
            layout: desc.layout.clone(),
            entries: entries.into(),
        })
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GroupOffsets {
    pub buffers: u64,
    pub textures: u64,
    pub samplers: u64,
}

/// Metal pipeline binding layout.
#[derive(Clone)]
pub struct MetalPipelineLayout {
    pub(crate) context: Arc<Context>,
    pub(crate) layouts: Arc<[MetalBindGroupLayout]>,
    pub(crate) offsets: Arc<[GroupOffsets]>,
}

impl MetalPipelineLayout {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &PipelineLayoutDesc<'_, crate::MetalBackend>,
    ) -> Result<Self> {
        let mut offsets = Vec::with_capacity(desc.bind_group_layouts.len());
        let mut next = GroupOffsets::default();
        let mut layouts = Vec::with_capacity(desc.bind_group_layouts.len());
        for layout in desc.bind_group_layouts {
            require_context(context, &layout.context)?;
            offsets.push(next);
            for entry in layout.entries.iter() {
                let end = u64::from(entry.binding) + 1;
                match entry.ty {
                    BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. } => {
                        next.buffers = next
                            .buffers
                            .max(offsets.last().map_or(0, |base| base.buffers) + end);
                    }
                    BindingType::SampledImage => {
                        let base = *offsets.last().unwrap_or(&GroupOffsets::default());
                        next.textures = next.textures.max(base.textures + end);
                        next.samplers = next.samplers.max(base.samplers + end);
                    }
                    BindingType::StorageImage => {
                        next.textures = next
                            .textures
                            .max(offsets.last().map_or(0, |base| base.textures) + end);
                    }
                }
            }
            layouts.push((*layout).clone());
        }
        if next.buffers > VERTEX_BUFFER_BASE {
            return Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
                "Metal pipeline layouts support at most 16 shader buffer slots"
            )));
        }
        Ok(Self {
            context: context.clone(),
            layouts: layouts.into(),
            offsets: offsets.into(),
        })
    }
}

/// Metal render pipeline state.
#[derive(Clone)]
pub struct MetalGraphicsPipeline {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: RenderPipelineState,
    pub(crate) depth: Option<DepthStencilState>,
    pub(crate) depth_bias: dirk_rhi::DepthBiasState,
    pub(crate) topology: metal::MTLPrimitiveType,
    pub(crate) winding: metal::MTLWinding,
    pub(crate) cull: metal::MTLCullMode,
}

impl MetalGraphicsPipeline {
    #[allow(
        clippy::too_many_lines,
        reason = "pipeline translation keeps vertex layout, blending, and depth/stencil state together"
    )]
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &GraphicsPipelineDesc<'_, crate::MetalBackend>,
    ) -> Result<Self> {
        for object in [&desc.layout.context, &desc.vertex.context] {
            require_context(context, object)?;
        }
        if let Some(fragment) = desc.fragment {
            require_context(context, &fragment.context)?;
        }
        if desc.vertex.stage != ShaderStage::Vertex
            || desc
                .fragment
                .is_some_and(|fragment| fragment.stage != ShaderStage::Fragment)
        {
            return Err(dirk_rhi::Error::from(Ir::Mismatch));
        }
        let vertex_descriptor = VertexDescriptor::new();
        for (buffer_index, layout) in desc.vertex_buffers.iter().enumerate() {
            let metal_index =
                VERTEX_BUFFER_BASE + u64::try_from(buffer_index).map_err(|_| Ir::OutOfRange)?;
            let metal_layout = vertex_descriptor
                .layouts()
                .object_at(metal_index)
                .ok_or_else(|| dirk_rhi::Error::from(Ir::OutOfRange))?;
            metal_layout.set_stride(u64::from(layout.stride));
            metal_layout.set_step_function(convert::vertex_step(layout.step_mode));
            metal_layout.set_step_rate(1);
            for attribute in layout.attributes {
                let metal_attribute = vertex_descriptor
                    .attributes()
                    .object_at(u64::from(attribute.location))
                    .ok_or_else(|| dirk_rhi::Error::from(Ir::OutOfRange))?;
                metal_attribute.set_format(convert::vertex_format(attribute.format));
                metal_attribute.set_offset(u64::from(attribute.offset));
                metal_attribute.set_buffer_index(metal_index);
            }
        }
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_label(desc.label);
        descriptor.set_vertex_function(Some(&desc.vertex.function));
        if let Some(fragment) = desc.fragment {
            descriptor.set_fragment_function(Some(&fragment.function));
        }
        descriptor.set_vertex_descriptor(Some(vertex_descriptor));
        descriptor.set_sample_count(convert::samples(desc.samples));
        descriptor.set_alpha_to_coverage_enabled(desc.alpha_to_coverage);
        for (index, target) in desc.color_targets.iter().enumerate() {
            let state = descriptor
                .color_attachments()
                .object_at(u64::try_from(index).map_err(|_| Ir::OutOfRange)?)
                .ok_or_else(|| dirk_rhi::Error::from(Ir::OutOfRange))?;
            state.set_pixel_format(convert::format(target.format));
            let mut write_mask = MTLColorWriteMask::empty();
            for (rhi, metal) in [
                (dirk_rhi::ColorWrites::RED, MTLColorWriteMask::Red),
                (dirk_rhi::ColorWrites::GREEN, MTLColorWriteMask::Green),
                (dirk_rhi::ColorWrites::BLUE, MTLColorWriteMask::Blue),
                (dirk_rhi::ColorWrites::ALPHA, MTLColorWriteMask::Alpha),
            ] {
                if target.write_mask.contains(rhi) {
                    write_mask |= metal;
                }
            }
            state.set_write_mask(write_mask);
            if let Some(blend) = target.blend {
                state.set_blending_enabled(true);
                state.set_source_rgb_blend_factor(convert::blend_factor(blend.color.source));
                state.set_destination_rgb_blend_factor(convert::blend_factor(
                    blend.color.destination,
                ));
                state.set_rgb_blend_operation(convert::blend_op(blend.color.operation));
                state.set_source_alpha_blend_factor(convert::blend_factor(blend.alpha.source));
                state.set_destination_alpha_blend_factor(convert::blend_factor(
                    blend.alpha.destination,
                ));
                state.set_alpha_blend_operation(convert::blend_op(blend.alpha.operation));
            }
        }
        descriptor.set_alpha_to_coverage_enabled(desc.alpha_to_coverage);
        let depth = desc.depth.map(|depth| {
            descriptor.set_depth_attachment_pixel_format(convert::format(depth.format));
            if matches!(
                depth.format,
                dirk_rhi::TextureFormat::Depth24UnormStencil8
                    | dirk_rhi::TextureFormat::Depth32FloatStencil8
            ) {
                descriptor.set_stencil_attachment_pixel_format(convert::format(depth.format));
            }
            let state = DepthStencilDescriptor::new();
            state.set_depth_compare_function(convert::compare(depth.compare));
            state.set_depth_write_enabled(depth.write_enabled);
            if let Some(stencil) = depth.stencil {
                let face = |face: dirk_rhi::StencilFaceState| {
                    let info = StencilDescriptor::new();
                    info.set_stencil_failure_operation(convert::stencil_op(face.fail_op));
                    info.set_depth_failure_operation(convert::stencil_op(face.depth_fail_op));
                    info.set_depth_stencil_pass_operation(convert::stencil_op(face.pass_op));
                    info.set_stencil_compare_function(convert::compare(face.compare));
                    info.set_read_mask(stencil.read_mask);
                    info.set_write_mask(stencil.write_mask);
                    info
                };
                state.set_front_face_stencil(Some(&face(stencil.front)));
                state.set_back_face_stencil(Some(&face(stencil.back)));
            }
            context.device.new_depth_stencil_state(&state)
        });
        let bias_enabled =
            desc.depth_bias.constant_factor != 0.0 || desc.depth_bias.slope_factor != 0.0;
        let raw = context
            .device
            .new_render_pipeline_state(&descriptor)
            .map_err(backend_error)?;
        Ok(Self {
            context: context.clone(),
            raw,
            depth,
            depth_bias: if bias_enabled {
                desc.depth_bias
            } else {
                dirk_rhi::DepthBiasState::default()
            },
            topology: convert::topology(desc.raster.topology),
            winding: convert::winding(desc.raster.front_face),
            cull: convert::cull(desc.raster.cull_mode),
        })
    }
}

/// CPU-waitable Metal submission fence.
pub struct MetalFence {
    pub(crate) context: Arc<Context>,
    pub(crate) event: SharedEvent,
    pub(crate) target: AtomicU64,
}

impl MetalFence {
    pub(crate) fn create(context: &Arc<Context>, signaled: bool) -> Self {
        let event = context.device.new_shared_event();
        if signaled {
            event.set_signaled_value(1);
        }
        Self {
            context: context.clone(),
            event,
            target: AtomicU64::new(1),
        }
    }

    pub(crate) fn value(&self) -> u64 {
        self.target.load(Ordering::Acquire)
    }
}

impl Fence for MetalFence {
    fn wait(&self, timeout_ns: u64) -> Result<()> {
        wait_event(&self.event, self.value(), timeout_ns)
    }

    fn reset(&self) -> Result<()> {
        let value = self.value();
        if self.event.signaled_value() < value {
            return Err(Ir::BadState.into());
        }
        self.target.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

/// Metal shared event used as an RHI timeline semaphore.
#[derive(Clone)]
pub struct MetalTimelineSemaphore {
    pub(crate) context: Arc<Context>,
    pub(crate) event: SharedEvent,
}

impl TimelineSemaphore for MetalTimelineSemaphore {
    fn wait(&self, value: u64, timeout_ns: u64) -> Result<()> {
        wait_event(&self.event, value, timeout_ns)
    }

    fn value(&self) -> Result<u64> {
        Ok(self.event.signaled_value())
    }
}

fn wait_event(event: &metal::SharedEventRef, value: u64, timeout_ns: u64) -> Result<()> {
    let started = Instant::now();
    let timeout = Duration::from_nanos(timeout_ns);
    while event.signaled_value() < value {
        if timeout_ns != u64::MAX && started.elapsed() >= timeout {
            return Err(dirk_rhi::Error::Backend(anyhow::anyhow!(
                "timed out waiting for a Metal shared event"
            )));
        }
        std::thread::yield_now();
    }
    Ok(())
}

pub(crate) fn require_context(expected: &Arc<Context>, actual: &Arc<Context>) -> Result<()> {
    if Arc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(dirk_rhi::Error::from(Ir::ForeignInstance))
    }
}

pub(crate) fn binding_visibility(layout: &MetalBindGroupLayout, binding: u32) -> ShaderStages {
    layout
        .entries
        .iter()
        .find(|entry| entry.binding == binding)
        .map_or(ShaderStages::NONE, |entry| entry.visibility)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multisampled_two_d_view_reuses_source_texture() -> Result<()> {
        assert!(MetalImageView::reuses_source_texture(
            MTLTextureType::D2Multisample,
            ImageViewType::TwoD,
            1,
        )?);
        Ok(())
    }
}
