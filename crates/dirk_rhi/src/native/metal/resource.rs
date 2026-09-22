use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crate::{
    BindGroupDesc, BindGroupLayoutDesc, BindingResource, BindingType, BufferDesc,
    GraphicsPipelineDesc, ImageAspects, ImageDesc, ImageDimension, ImageUsages, ImageViewDesc,
    ImageViewType, InvalidResourceKind as Ir, MemoryDomain, NativeBuffer, NativeFence,
    NativeTimelineSemaphore, PipelineLayoutDesc, Result, SamplerDesc, ShaderDesc, ShaderSource,
    ShaderStage,
};
use metal::{
    CompileOptions, DepthStencilDescriptor, DepthStencilState, Function, MTLColorWriteMask,
    MTLResourceOptions, MTLSamplerMipFilter, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    RenderPipelineDescriptor, RenderPipelineState, SamplerDescriptor, SamplerState, SharedEvent,
    StencilDescriptor, Texture, TextureDescriptor, VertexDescriptor,
};

use super::{backend::Context, backend_error, convert};

pub(crate) const VERTEX_BUFFER_BASE: u64 = 16;

/// Metal buffer and its RHI allocation metadata.
pub struct MetalBuffer {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: metal::Buffer,
    pub(crate) size: u64,
    pub(crate) memory: MemoryDomain,
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
            return Err(crate::Error::from(Ir::Empty));
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
        })
    }
}

unsafe impl NativeBuffer for MetalBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    unsafe fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if self.memory == MemoryDomain::Device {
            return Err(crate::InvalidResourceKind::NotHostAccessible.into());
        }
        let length =
            u64::try_from(data.len()).map_err(|error| crate::Error::Backend(error.into()))?;
        if offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(crate::InvalidResourceKind::OutOfRange.into());
        }
        let offset = usize::try_from(offset).map_err(|_| crate::InvalidResourceKind::OutOfRange)?;
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

    unsafe fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
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
pub struct MetalImage {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: Texture,
    pub(crate) format: crate::TextureFormat,
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
            return Err(crate::Error::from(Ir::Empty));
        }
        if desc.array_layers > 1 && desc.samples != crate::SampleCount::One {
            return Err(crate::Error::Backend(anyhow::anyhow!(
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
                } else if desc.samples != crate::SampleCount::One {
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
        format: crate::TextureFormat,
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
                | ImageViewType::CubeArray => Err(crate::Error::from(Ir::Mismatch)),
            },
            MTLTextureType::D2MultisampleArray | MTLTextureType::D3 => Err(crate::Error::Backend(
                anyhow::anyhow!("the RHI cannot represent this Metal texture view type"),
            )),
            _ => Ok(false),
        }
    }

    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &ImageViewDesc<'_, super::MetalBackend>,
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
pub struct MetalShader {
    pub(crate) context: Arc<Context>,
    pub(crate) function: Function,
    pub(crate) stage: ShaderStage,
}

impl MetalShader {
    pub(crate) fn create(context: &Arc<Context>, desc: &ShaderDesc<'_>) -> Result<Self> {
        let ShaderSource::Msl(source) = desc.source else {
            return Err(crate::UnsupportedOperation::ShaderSource(desc.source.language()).into());
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
pub struct MetalBindGroupLayout {
    pub(crate) context: Arc<Context>,
    pub(crate) entries: Vec<crate::BindGroupLayoutEntry>,
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
            entries,
        })
    }
}

pub(crate) enum NativeBinding {
    Buffer {
        buffer: Borrowed<metal::BufferRef>,
        offset: u64,
    },
    SampledImage {
        view: Borrowed<metal::TextureRef>,
        sampler: Borrowed<metal::SamplerStateRef>,
    },
    StorageImage(Borrowed<metal::TextureRef>),
}

/// Metal resources associated with one bind group.
pub struct MetalBindGroup {
    pub(crate) context: Arc<Context>,
    pub(crate) layout: Vec<crate::BindGroupLayoutEntry>,
    pub(crate) entries: Vec<(u32, NativeBinding)>,
}

impl MetalBindGroup {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &BindGroupDesc<'_, super::MetalBackend>,
    ) -> Result<Self> {
        require_context(context, &desc.layout.context)?;
        let mut entries = Vec::with_capacity(desc.entries.len());
        for entry in desc.entries {
            let layout_entry = desc
                .layout
                .entries
                .iter()
                .find(|layout| layout.binding == entry.binding)
                .ok_or_else(|| crate::Error::from(Ir::Mismatch))?;
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
                        return Err(crate::Error::from(Ir::OutOfRange));
                    }
                    NativeBinding::Buffer {
                        buffer: Borrowed::new(&buffer.raw),
                        offset: *offset,
                    }
                }
                (BindingResource::SampledImage { view, sampler }, BindingType::SampledImage) => {
                    require_context(context, &view.context)?;
                    require_context(context, &sampler.context)?;
                    NativeBinding::SampledImage {
                        view: Borrowed::new(&view.raw),
                        sampler: Borrowed::new(&sampler.raw),
                    }
                }
                (BindingResource::StorageImage(view), BindingType::StorageImage) => {
                    require_context(context, &view.context)?;
                    NativeBinding::StorageImage(Borrowed::new(&view.raw))
                }
                _ => {
                    return Err(crate::Error::from(Ir::Mismatch));
                }
            };
            entries.push((entry.binding, resource));
        }
        if entries.len() != desc.layout.entries.len() {
            return Err(crate::Error::from(Ir::Mismatch));
        }
        entries.sort_unstable_by_key(|entry| entry.0);
        Ok(Self {
            context: context.clone(),
            layout: desc.layout.entries.clone(),
            entries,
        })
    }
}

/// Metal pipeline binding layout.
pub struct MetalPipelineLayout {
    pub(crate) context: Arc<Context>,
    pub(crate) layouts: Vec<Vec<crate::BindGroupLayoutEntry>>,
    pub(crate) vertex: crate::BindingMap,
    pub(crate) fragment: crate::BindingMap,
}

impl MetalPipelineLayout {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &PipelineLayoutDesc<'_, super::MetalBackend>,
    ) -> Result<Self> {
        let mut layouts = Vec::with_capacity(desc.bind_group_layouts.len());
        for layout in desc.bind_group_layouts {
            require_context(context, &layout.context)?;
            layouts.push(layout.entries.clone());
        }
        let entries = layouts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let vertex = crate::BindingMap::new(&entries, ShaderStage::Vertex)?;
        let fragment = crate::BindingMap::new(&entries, ShaderStage::Fragment)?;
        vertex.validate(crate::Limits::default())?;
        fragment.validate(crate::Limits::default())?;
        Ok(Self {
            context: context.clone(),
            layouts,
            vertex,
            fragment,
        })
    }
}

/// Metal render pipeline state.
pub struct MetalGraphicsPipeline {
    pub(crate) context: Arc<Context>,
    pub(crate) raw: RenderPipelineState,
    pub(crate) depth: Option<DepthStencilState>,
    pub(crate) depth_bias: crate::DepthBiasState,
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
        desc: &GraphicsPipelineDesc<'_, super::MetalBackend>,
    ) -> Result<Self> {
        // Metal always enables restart for indexed triangle strips.
        if desc.raster.topology == crate::PrimitiveTopology::TriangleStrip
            && desc.primitive_restart.is_none()
        {
            return Err(crate::UnsupportedOperation::Capability(
                "disabling primitive restart for Metal triangle strips",
            )
            .into());
        }
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
            return Err(crate::Error::from(Ir::Mismatch));
        }
        let vertex_descriptor = VertexDescriptor::new();
        for (buffer_index, layout) in desc.vertex_buffers.iter().enumerate() {
            let metal_index =
                VERTEX_BUFFER_BASE + u64::try_from(buffer_index).map_err(|_| Ir::OutOfRange)?;
            let metal_layout = vertex_descriptor
                .layouts()
                .object_at(metal_index)
                .ok_or_else(|| crate::Error::from(Ir::OutOfRange))?;
            metal_layout.set_stride(u64::from(layout.stride));
            metal_layout.set_step_function(convert::vertex_step(layout.step_mode));
            metal_layout.set_step_rate(1);
            for attribute in layout.attributes {
                let metal_attribute = vertex_descriptor
                    .attributes()
                    .object_at(u64::from(attribute.location))
                    .ok_or_else(|| crate::Error::from(Ir::OutOfRange))?;
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
                .ok_or_else(|| crate::Error::from(Ir::OutOfRange))?;
            state.set_pixel_format(convert::format(target.format));
            let mut write_mask = MTLColorWriteMask::empty();
            for (rhi, metal) in [
                (crate::ColorWrites::RED, MTLColorWriteMask::Red),
                (crate::ColorWrites::GREEN, MTLColorWriteMask::Green),
                (crate::ColorWrites::BLUE, MTLColorWriteMask::Blue),
                (crate::ColorWrites::ALPHA, MTLColorWriteMask::Alpha),
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
                crate::TextureFormat::Depth24UnormStencil8
                    | crate::TextureFormat::Depth32FloatStencil8
            ) {
                descriptor.set_stencil_attachment_pixel_format(convert::format(depth.format));
            }
            let state = DepthStencilDescriptor::new();
            state.set_depth_compare_function(convert::compare(depth.compare));
            state.set_depth_write_enabled(depth.write_enabled);
            if let Some(stencil) = depth.stencil {
                let face = |face: crate::StencilFaceState| {
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
                crate::DepthBiasState::default()
            },
            topology: convert::topology(desc.raster.topology),
            winding: convert::winding(desc.raster.front_face),
            cull: convert::cull(desc.raster.cull_mode),
        })
    }
}

/// CPU-waitable completion of the actual native command buffers, including errors.
pub struct MetalFence {
    pub(crate) context: Arc<Context>,
    commands: parking_lot::Mutex<Vec<metal::CommandBuffer>>,
    signaled: AtomicBool,
}
impl MetalFence {
    pub(crate) fn create(context: &Arc<Context>, signaled: bool) -> Self {
        Self {
            context: context.clone(),
            commands: parking_lot::Mutex::new(Vec::new()),
            signaled: AtomicBool::new(signaled),
        }
    }
    pub(crate) fn track(&self, commands: &[metal::CommandBuffer]) {
        self.commands.lock().clone_from(&commands.to_vec());
    }
}
unsafe impl NativeFence for MetalFence {
    unsafe fn wait(&self, timeout_ns: u64) -> Result<()> {
        let started = Instant::now();
        loop {
            let commands = self.commands.lock();
            if self.signaled.load(Ordering::Acquire) {
                return Ok(());
            }
            if commands
                .iter()
                .any(|command| command.status() == metal::MTLCommandBufferStatus::Error)
            {
                return Err(crate::Error::DeviceLost);
            }
            if !commands.is_empty()
                && commands
                    .iter()
                    .all(|command| command.status() == metal::MTLCommandBufferStatus::Completed)
            {
                self.signaled.store(true, Ordering::Release);
                return Ok(());
            }
            if timeout_ns != u64::MAX && started.elapsed() >= Duration::from_nanos(timeout_ns) {
                return Err(crate::Error::Timeout);
            }
            drop(commands);
            std::thread::yield_now();
        }
    }
    unsafe fn reset(&self) -> Result<()> {
        unsafe { self.wait(0) }?;
        self.commands.lock().clear();
        self.signaled.store(false, Ordering::Release);
        Ok(())
    }
}

/// Metal shared event used as an RHI timeline semaphore.
#[derive(Clone)]
pub struct MetalTimelineSemaphore {
    pub(crate) context: Arc<Context>,
    pub(crate) event: SharedEvent,
}

unsafe impl NativeTimelineSemaphore for MetalTimelineSemaphore {
    unsafe fn wait(&self, value: u64, timeout_ns: u64) -> Result<()> {
        wait_event(&self.event, value, timeout_ns)
    }

    unsafe fn value(&self) -> Result<u64> {
        Ok(self.event.signaled_value())
    }
}

fn wait_event(event: &metal::SharedEventRef, value: u64, timeout_ns: u64) -> Result<()> {
    let started = Instant::now();
    let timeout = Duration::from_nanos(timeout_ns);
    while event.signaled_value() < value {
        if timeout_ns != u64::MAX && started.elapsed() >= timeout {
            return Err(crate::Error::Timeout);
        }
        std::thread::yield_now();
    }
    Ok(())
}

pub(crate) fn require_context(expected: &Arc<Context>, actual: &Arc<Context>) -> Result<()> {
    if Arc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(crate::Error::from(Ir::ForeignInstance))
    }
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

/// Unretained native binding. The public recording operation carries the lifetime obligation.
pub(crate) struct Borrowed<T>(std::ptr::NonNull<T>);
// SAFETY: only shared native references cross threads; callers of get uphold lifetime.
unsafe impl<T: Sync> Send for Borrowed<T> {}
// SAFETY: shared access is valid for Sync native reference types.
unsafe impl<T: Sync> Sync for Borrowed<T> {}
impl<T> Borrowed<T> {
    pub(crate) fn new(value: &T) -> Self {
        Self(std::ptr::NonNull::from(value))
    }
    pub(crate) unsafe fn get(&self) -> &T {
        unsafe { self.0.as_ref() }
    }
}
// Metal's owned native handles retain on clone. At retirement, move their final
// release into the cycle bucket; no per-binding resource owner is cloned.
macro_rules! retire {
    ($ty:ty, $field:ident, $kind:ident) => {
        impl Drop for $ty {
            fn drop(&mut self) {
                self.context
                    .retire(super::backend::Garbage::$kind(self.$field.clone()));
            }
        }
    };
}
retire!(MetalBuffer, raw, Buffer);
retire!(MetalImage, raw, Texture);
retire!(MetalImageView, raw, Texture);
retire!(MetalSampler, raw, Sampler);
retire!(MetalShader, function, Shader);
impl Drop for MetalGraphicsPipeline {
    fn drop(&mut self) {
        self.context
            .retire(super::backend::Garbage::Pipeline(self.raw.clone()));
        if let Some(depth) = &self.depth {
            self.context
                .retire(super::backend::Garbage::Depth(depth.clone()));
        }
    }
}
