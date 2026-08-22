use std::{ffi::CString, fmt, sync::Arc};

use ash::vk;
use dirk_rhi::{
    BindGroupDesc, BindGroupLayoutDesc, BindingResource, BindingType, Buffer, BufferDesc, Fence,
    GraphicsPipelineDesc, ImageDesc, ImageViewDesc, InvalidResource as Ir, Result, SamplerDesc,
    ShaderDesc, ShaderSource, ShaderStage, TimelineSemaphore,
};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use parking_lot::Mutex;

use crate::{
    VulkanBackend, convert,
    device::{Context, Garbage, Retained},
    presentation::SwapchainGeneration,
    vk_error,
};

#[derive(Clone)]
/// Allocated Vulkan buffer with shared ownership.
pub struct VulkanBuffer(Arc<BufferInner>);

impl fmt::Debug for VulkanBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanBuffer")
            .field("raw", &self.0.raw)
            .field("size", &self.0.size)
            .finish()
    }
}

struct BufferInner {
    context: Arc<Context>,
    raw: vk::Buffer,
    size: u64,
    allocation: Mutex<Option<Allocation>>,
}

impl VulkanBuffer {
    pub(crate) fn create(context: &Arc<Context>, desc: &BufferDesc<'_>) -> Result<Self> {
        if desc.size == 0 {
            return Err(Ir::Empty.into());
        }
        let create_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(convert::buffer_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let raw = unsafe { context.device.create_buffer(&create_info, None) }.map_err(vk_error)?;
        let requirements = unsafe { context.device.get_buffer_memory_requirements(raw) };
        let allocation = match context.allocate(&AllocationCreateDesc {
            name: desc.label,
            requirements,
            location: convert::memory(desc.memory),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { context.device.destroy_buffer(raw, None) };
                return Err(error);
            }
        };
        if let Err(error) = unsafe {
            context
                .device
                .bind_buffer_memory(raw, allocation.memory(), allocation.offset())
        } {
            unsafe { context.device.destroy_buffer(raw, None) };
            context.allocator_free(allocation);
            return Err(vk_error(error));
        }
        Ok(Self(Arc::new(BufferInner {
            context: context.clone(),
            raw,
            size: desc.size,
            allocation: Mutex::new(Some(allocation)),
        })))
    }

    #[must_use]
    /// Returns the native buffer handle.
    pub fn raw(&self) -> vk::Buffer {
        self.0.raw
    }

    #[must_use]
    /// Returns the allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.0.size
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Buffer for VulkanBuffer {
    fn size(&self) -> u64 {
        self.0.size
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let data_len = u64::try_from(data.len()).map_err(|_| Ir::OutOfRange)?;
        if offset
            .checked_add(data_len)
            .is_none_or(|end| end > self.0.size)
        {
            return Err(Ir::OutOfRange.into());
        }
        let allocation = self.0.allocation.lock();
        let allocation = allocation.as_ref().ok_or(Ir::BadState)?;
        let mapped = allocation.mapped_ptr().ok_or(Ir::NotHostAccessible)?;
        let host_offset = usize::try_from(offset).map_err(|_| Ir::OutOfRange)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped.as_ptr().cast::<u8>().add(host_offset),
                data.len(),
            );
        }
        if !allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            let atom_size = self.0.context.non_coherent_atom_size.max(1);
            let absolute_offset = allocation.offset() + offset;
            let flush_offset = absolute_offset / atom_size * atom_size;
            let range = vk::MappedMemoryRange::default()
                .memory(unsafe { allocation.memory() })
                .offset(flush_offset)
                .size(vk::WHOLE_SIZE);
            unsafe {
                self.0
                    .context
                    .device
                    .flush_mapped_memory_ranges(std::slice::from_ref(&range))
                    .map_err(vk_error)?;
            }
        }
        Ok(())
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.get_mut().take() {
            self.context.retire(Garbage::Buffer {
                raw: self.raw,
                allocation,
            });
        }
    }
}

#[derive(Clone)]
/// Owned or swapchain-provided Vulkan image.
pub struct VulkanImage(Arc<ImageInner>);

impl fmt::Debug for VulkanImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanImage")
            .field("raw", &self.0.raw)
            .finish()
    }
}

enum ImageOwnership {
    Owned {
        context: Arc<Context>,
        allocation: Mutex<Option<Allocation>>,
    },
    Surface {
        generation: Arc<SwapchainGeneration>,
    },
}

struct ImageInner {
    raw: vk::Image,
    format: dirk_rhi::TextureFormat,
    extent: dirk_rhi::Extent3d,
    ownership: ImageOwnership,
}

impl VulkanImage {
    pub(crate) fn create(context: &Arc<Context>, desc: &ImageDesc<'_>) -> Result<Self> {
        if desc.extent.width == 0 || desc.extent.height == 0 || desc.extent.depth == 0 {
            return Err(Ir::Empty.into());
        }
        let image_type = if desc.extent.depth > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };
        let create_info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .format(convert::format(desc.format))
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(convert::samples(desc.samples))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(convert::image_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let raw = unsafe { context.device.create_image(&create_info, None) }.map_err(vk_error)?;
        let requirements = unsafe { context.device.get_image_memory_requirements(raw) };
        let allocation = match context.allocate(&AllocationCreateDesc {
            name: desc.label,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { context.device.destroy_image(raw, None) };
                return Err(error);
            }
        };
        if let Err(error) = unsafe {
            context
                .device
                .bind_image_memory(raw, allocation.memory(), allocation.offset())
        } {
            unsafe { context.device.destroy_image(raw, None) };
            context.allocator_free(allocation);
            return Err(vk_error(error));
        }
        Ok(Self(Arc::new(ImageInner {
            raw,
            format: desc.format,
            extent: desc.extent,
            ownership: ImageOwnership::Owned {
                context: context.clone(),
                allocation: Mutex::new(Some(allocation)),
            },
        })))
    }

    pub(crate) fn surface(
        generation: Arc<SwapchainGeneration>,
        raw: vk::Image,
        format: dirk_rhi::TextureFormat,
        extent: dirk_rhi::Extent3d,
    ) -> Self {
        Self(Arc::new(ImageInner {
            raw,
            format,
            extent,
            ownership: ImageOwnership::Surface { generation },
        }))
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        match &self.0.ownership {
            ImageOwnership::Owned { context, .. } => context,
            ImageOwnership::Surface { generation } => &generation.context,
        }
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    #[must_use]
    /// Returns the native image handle.
    pub fn raw(&self) -> vk::Image {
        self.0.raw
    }

    #[must_use]
    /// Returns the neutral image format.
    pub fn format(&self) -> dirk_rhi::TextureFormat {
        self.0.format
    }

    #[must_use]
    /// Returns the image extent.
    pub fn extent(&self) -> dirk_rhi::Extent3d {
        self.0.extent
    }
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if let ImageOwnership::Owned {
            context,
            allocation,
        } = &mut self.ownership
            && let Some(allocation) = allocation.get_mut().take()
        {
            context.retire(Garbage::Image {
                raw: self.raw,
                allocation,
            });
        }
    }
}

#[derive(Clone)]
/// Shared Vulkan image view.
pub struct VulkanImageView(Arc<ImageViewInner>);

impl fmt::Debug for VulkanImageView {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanImageView")
            .field("raw", &self.0.raw)
            .finish()
    }
}

enum ImageViewOwnership {
    Owned(Arc<Context>),
    Surface(Arc<SwapchainGeneration>),
}

struct ImageViewInner {
    raw: vk::ImageView,
    aspects: dirk_rhi::ImageAspects,
    ownership: ImageViewOwnership,
    _image: Option<VulkanImage>,
}

impl VulkanImageView {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &ImageViewDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(desc.image.raw())
            .view_type(convert::view_type(desc.view_type))
            .format(convert::format(desc.image.format()))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: convert::aspects(desc.aspects),
                base_mip_level: desc.base_mip_level,
                level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                layer_count: desc.array_layer_count,
            });
        let raw =
            unsafe { context.device.create_image_view(&create_info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(ImageViewInner {
            raw,
            aspects: desc.aspects,
            ownership: ImageViewOwnership::Owned(context.clone()),
            _image: Some(desc.image.clone()),
        })))
    }

    pub(crate) fn surface(generation: Arc<SwapchainGeneration>, raw: vk::ImageView) -> Self {
        Self(Arc::new(ImageViewInner {
            raw,
            aspects: dirk_rhi::ImageAspects::COLOR,
            ownership: ImageViewOwnership::Surface(generation),
            _image: None,
        }))
    }

    #[must_use]
    /// Returns the native image-view handle.
    pub fn raw(&self) -> vk::ImageView {
        self.0.raw
    }

    pub(crate) fn aspects(&self) -> dirk_rhi::ImageAspects {
        self.0.aspects
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        match &self.0.ownership {
            ImageViewOwnership::Owned(context) => context,
            ImageViewOwnership::Surface(generation) => &generation.context,
        }
    }
}

impl Drop for ImageViewInner {
    fn drop(&mut self) {
        match &self.ownership {
            ImageViewOwnership::Owned(context) => context.retire(Garbage::ImageView(self.raw)),
            ImageViewOwnership::Surface(_generation) => {}
        }
    }
}

#[derive(Clone)]
/// Shared Vulkan texture sampler.
pub struct VulkanSampler(Arc<SamplerInner>);

impl fmt::Debug for VulkanSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanSampler")
            .field("raw", &self.0.raw)
            .finish()
    }
}

struct SamplerInner {
    context: Arc<Context>,
    raw: vk::Sampler,
}

impl VulkanSampler {
    pub(crate) fn create(context: &Arc<Context>, desc: &SamplerDesc<'_>) -> Result<Self> {
        if desc.max_anisotropy > context.capabilities.max_sampler_anisotropy {
            return Err(Ir::OutOfRange.into());
        }
        let anisotropy = context.sampler_anisotropy && desc.max_anisotropy > 1;
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(convert::filter(desc.mag_filter))
            .min_filter(convert::filter(desc.min_filter))
            .mipmap_mode(convert::mipmap_filter(desc.mip_filter))
            .address_mode_u(convert::address(desc.address_u))
            .address_mode_v(convert::address(desc.address_v))
            .address_mode_w(convert::address(desc.address_w))
            .anisotropy_enable(anisotropy)
            .max_anisotropy(if anisotropy {
                f32::from(desc.max_anisotropy)
            } else {
                1.0
            })
            .min_lod(desc.lod_min)
            .max_lod(desc.lod_max);
        let raw = unsafe { context.device.create_sampler(&create_info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(SamplerInner {
            context: context.clone(),
            raw,
        })))
    }

    #[must_use]
    /// Returns the native sampler handle.
    pub fn raw(&self) -> vk::Sampler {
        self.0.raw
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for SamplerInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::Sampler(self.raw));
    }
}

#[derive(Clone)]
/// Shared Vulkan shader module.
pub struct VulkanShader(Arc<ShaderInner>);

struct ShaderInner {
    context: Arc<Context>,
    raw: vk::ShaderModule,
    stage: ShaderStage,
    entry: CString,
}

impl VulkanShader {
    pub(crate) fn create(context: &Arc<Context>, desc: &ShaderDesc<'_>) -> Result<Self> {
        let ShaderSource::SpirV(code) = desc.source else {
            return Err(
                dirk_rhi::UnsupportedOperation::ShaderSource(desc.source.language()).into(),
            );
        };
        let entry = CString::new(desc.entry).map_err(|_| Ir::Malformed)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        let raw =
            unsafe { context.device.create_shader_module(&create_info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(ShaderInner {
            context: context.clone(),
            raw,
            stage: desc.stage,
            entry,
        })))
    }

    #[must_use]
    /// Returns the native shader-module handle.
    pub fn raw(&self) -> vk::ShaderModule {
        self.0.raw
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for ShaderInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::Shader(self.raw));
    }
}

#[derive(Clone)]
/// Vulkan descriptor-set layout implementing an RHI bind-group layout.
pub struct VulkanBindGroupLayout(Arc<BindGroupLayoutInner>);

struct BindGroupLayoutInner {
    context: Arc<Context>,
    raw: vk::DescriptorSetLayout,
    entries: Vec<dirk_rhi::BindGroupLayoutEntry>,
}

impl VulkanBindGroupLayout {
    pub(crate) fn create(context: &Arc<Context>, desc: &BindGroupLayoutDesc<'_>) -> Result<Self> {
        let mut bindings = std::collections::HashSet::new();
        if desc
            .entries
            .iter()
            .any(|entry| !bindings.insert(entry.binding))
        {
            return Err(Ir::Mismatch.into());
        }
        let entries = desc
            .entries
            .iter()
            .map(|entry| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(entry.binding)
                    .descriptor_type(convert::binding(entry.ty))
                    .descriptor_count(1)
                    .stage_flags(convert::shader_stages(entry.visibility))
            })
            .collect::<Vec<_>>();
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&entries);
        let raw = unsafe {
            context
                .device
                .create_descriptor_set_layout(&create_info, None)
        }
        .map_err(vk_error)?;
        Ok(Self(Arc::new(BindGroupLayoutInner {
            context: context.clone(),
            raw,
            entries: desc.entries.to_vec(),
        })))
    }

    #[must_use]
    /// Returns the native descriptor-set layout.
    pub fn raw(&self) -> vk::DescriptorSetLayout {
        self.0.raw
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for BindGroupLayoutInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::BindGroupLayout(self.raw));
    }
}

#[derive(Clone)]
/// Vulkan descriptor set and its owned descriptor pool.
pub struct VulkanBindGroup(Arc<BindGroupInner>);

struct BindGroupInner {
    context: Arc<Context>,
    raw: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    _layout: VulkanBindGroupLayout,
    _buffers: Vec<VulkanBuffer>,
    _views: Vec<VulkanImageView>,
    _samplers: Vec<VulkanSampler>,
}

enum DescriptorData {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
}

struct DescriptorResources {
    data: Vec<DescriptorData>,
    buffers: Vec<VulkanBuffer>,
    views: Vec<VulkanImageView>,
    samplers: Vec<VulkanSampler>,
}

impl VulkanBindGroup {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &BindGroupDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        if desc.entries.len() != desc.layout.0.entries.len() {
            return Err(Ir::Mismatch.into());
        }
        if !Arc::ptr_eq(context, desc.layout.context()) {
            return Err(Ir::ForeignInstance.into());
        }
        let DescriptorResources {
            data,
            buffers,
            views,
            samplers,
        } = Self::descriptor_resources(context, desc)?;
        let mut counts = std::collections::HashMap::<vk::DescriptorType, u32>::new();
        for entry in &desc.layout.0.entries {
            *counts.entry(convert::binding(entry.ty)).or_default() += 1;
        }
        let pool_sizes = counts
            .into_iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            })
            .collect::<Vec<_>>();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool =
            unsafe { context.device.create_descriptor_pool(&pool_info, None) }.map_err(vk_error)?;
        let layouts = [desc.layout.raw()];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let raw = match unsafe { context.device.allocate_descriptor_sets(&allocate_info) } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe { context.device.destroy_descriptor_pool(pool, None) };
                return Err(vk_error(error));
            }
        };

        let writes = desc
            .entries
            .iter()
            .zip(&data)
            .map(|(entry, data)| {
                let expected = desc
                    .layout
                    .0
                    .entries
                    .iter()
                    .find(|layout| layout.binding == entry.binding)
                    .expect("bindings were validated while descriptor data was built");
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(raw)
                    .dst_binding(entry.binding)
                    .descriptor_type(convert::binding(expected.ty));
                match data {
                    DescriptorData::Buffer(info) => write.buffer_info(std::slice::from_ref(info)),
                    DescriptorData::Image(info) => write.image_info(std::slice::from_ref(info)),
                }
            })
            .collect::<Vec<_>>();
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };

        Ok(Self(Arc::new(BindGroupInner {
            context: context.clone(),
            raw,
            pool,
            _layout: desc.layout.clone(),
            _buffers: buffers,
            _views: views,
            _samplers: samplers,
        })))
    }

    fn descriptor_resources(
        context: &Arc<Context>,
        desc: &BindGroupDesc<'_, VulkanBackend>,
    ) -> Result<DescriptorResources> {
        let mut resources = DescriptorResources {
            data: Vec::with_capacity(desc.entries.len()),
            buffers: Vec::new(),
            views: Vec::new(),
            samplers: Vec::new(),
        };
        let mut seen = std::collections::HashSet::new();
        for entry in desc.entries {
            if !seen.insert(entry.binding) {
                return Err(Ir::Mismatch.into());
            }
            let expected = desc
                .layout
                .0
                .entries
                .iter()
                .find(|layout| layout.binding == entry.binding)
                .ok_or(Ir::Mismatch)?;
            match (&entry.resource, expected.ty) {
                (
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    },
                    BindingType::UniformBuffer | BindingType::StorageBuffer,
                ) if Arc::ptr_eq(context, buffer.context())
                    && *size > 0
                    && offset
                        .checked_add(*size)
                        .is_some_and(|end| end <= buffer.size()) =>
                {
                    resources
                        .data
                        .push(DescriptorData::Buffer(vk::DescriptorBufferInfo {
                            buffer: buffer.raw(),
                            offset: *offset,
                            range: *size,
                        }));
                    resources.buffers.push((*buffer).clone());
                }
                (BindingResource::SampledImage { view, sampler }, BindingType::SampledImage)
                    if Arc::ptr_eq(context, view.context())
                        && Arc::ptr_eq(context, sampler.context()) =>
                {
                    resources
                        .data
                        .push(DescriptorData::Image(vk::DescriptorImageInfo {
                            sampler: sampler.raw(),
                            image_view: view.raw(),
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        }));
                    resources.views.push((*view).clone());
                    resources.samplers.push((*sampler).clone());
                }
                (BindingResource::StorageImage(view), BindingType::StorageImage)
                    if Arc::ptr_eq(context, view.context()) =>
                {
                    resources
                        .data
                        .push(DescriptorData::Image(vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: view.raw(),
                            image_layout: vk::ImageLayout::GENERAL,
                        }));
                    resources.views.push((*view).clone());
                }
                _ => {
                    return Err(Ir::Mismatch.into());
                }
            }
        }
        Ok(resources)
    }

    #[must_use]
    /// Returns the native descriptor-set handle.
    pub fn raw(&self) -> vk::DescriptorSet {
        self.0.raw
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for BindGroupInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::DescriptorPool(self.pool));
    }
}

#[derive(Clone)]
/// Shared Vulkan pipeline layout.
pub struct VulkanPipelineLayout(Arc<PipelineLayoutInner>);

struct PipelineLayoutInner {
    context: Arc<Context>,
    raw: vk::PipelineLayout,
    _layouts: Vec<VulkanBindGroupLayout>,
}

impl VulkanPipelineLayout {
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &dirk_rhi::PipelineLayoutDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        if desc
            .bind_group_layouts
            .iter()
            .any(|layout| !Arc::ptr_eq(context, layout.context()))
        {
            return Err(Ir::ForeignInstance.into());
        }
        let raw_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|layout| layout.raw())
            .collect::<Vec<_>>();
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&raw_layouts);
        let raw = unsafe { context.device.create_pipeline_layout(&create_info, None) }
            .map_err(vk_error)?;
        Ok(Self(Arc::new(PipelineLayoutInner {
            context: context.clone(),
            raw,
            _layouts: desc
                .bind_group_layouts
                .iter()
                .map(|layout| (*layout).clone())
                .collect(),
        })))
    }

    #[must_use]
    /// Returns the native pipeline-layout handle.
    pub fn raw(&self) -> vk::PipelineLayout {
        self.0.raw
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }
}

impl Drop for PipelineLayoutInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::PipelineLayout(self.raw));
    }
}

#[derive(Clone)]
/// Shared Vulkan graphics pipeline.
pub struct VulkanGraphicsPipeline(Arc<GraphicsPipelineInner>);

struct GraphicsPipelineInner {
    context: Arc<Context>,
    raw: vk::Pipeline,
    _layout: VulkanPipelineLayout,
}

impl VulkanGraphicsPipeline {
    #[allow(
        clippy::too_many_lines,
        reason = "pipeline state translation stays together to keep borrowed Vulkan create-info lifetimes explicit"
    )]
    pub(crate) fn create(
        context: &Arc<Context>,
        desc: &GraphicsPipelineDesc<'_, VulkanBackend>,
    ) -> Result<Self> {
        if !Arc::ptr_eq(context, desc.layout.context())
            || !Arc::ptr_eq(context, desc.vertex.context())
            || desc
                .fragment
                .is_some_and(|fragment| !Arc::ptr_eq(context, fragment.context()))
        {
            return Err(Ir::ForeignInstance.into());
        }
        if desc.vertex.0.stage != ShaderStage::Vertex
            || desc
                .fragment
                .is_some_and(|fragment| fragment.0.stage != ShaderStage::Fragment)
        {
            return Err(Ir::Mismatch.into());
        }
        let mut shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(desc.vertex.raw())
                .name(&desc.vertex.0.entry),
        ];
        if let Some(fragment) = desc.fragment {
            let fragment_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment.raw())
                .name(&fragment.0.entry);
            shader_stages.push(fragment_stage);
        }
        let bindings = desc
            .vertex_buffers
            .iter()
            .enumerate()
            .map(|(index, layout)| {
                Ok(vk::VertexInputBindingDescription {
                    binding: u32::try_from(index).map_err(|_| Ir::OutOfRange)?,
                    stride: layout.stride,
                    input_rate: convert::input_rate(layout.step_mode),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut attributes = Vec::new();
        for (binding, layout) in desc.vertex_buffers.iter().enumerate() {
            let binding = u32::try_from(binding).map_err(|_| Ir::OutOfRange)?;
            attributes.extend(layout.attributes.iter().map(|attribute| {
                vk::VertexInputAttributeDescription {
                    location: attribute.location,
                    binding,
                    format: convert::vertex_format(attribute.format),
                    offset: attribute.offset,
                }
            }));
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(convert::topology(desc.raster.topology))
            .primitive_restart_enable(desc.primitive_restart);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let bias_enabled =
            desc.depth_bias.constant_factor != 0.0 || desc.depth_bias.slope_factor != 0.0;
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(convert::cull(desc.raster.cull_mode))
            .front_face(convert::front_face(desc.raster.front_face))
            .depth_bias_enable(bias_enabled)
            .depth_bias_constant_factor(desc.depth_bias.constant_factor)
            .depth_bias_slope_factor(desc.depth_bias.slope_factor)
            .depth_bias_clamp(desc.depth_bias.clamp)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(convert::samples(desc.samples))
            .alpha_to_coverage_enable(desc.alpha_to_coverage);
        let color_attachments = desc
            .color_formats
            .iter()
            .map(|_| match desc.blend {
                Some(blend) => {
                    let component = |component: dirk_rhi::BlendComponent| {
                        vk::PipelineColorBlendAttachmentState::default()
                            .blend_enable(true)
                            .src_color_blend_factor(convert::blend_factor(component.source))
                            .dst_color_blend_factor(convert::blend_factor(component.destination))
                            .color_blend_op(convert::blend_op(component.operation))
                            .src_alpha_blend_factor(convert::blend_factor(component.source))
                            .dst_alpha_blend_factor(convert::blend_factor(component.destination))
                            .alpha_blend_op(convert::blend_op(component.operation))
                            .color_write_mask(vk::ColorComponentFlags::RGBA)
                    };
                    component(blend.color)
                }
                None => vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA),
            })
            .collect::<Vec<_>>();
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_attachments);
        let depth =
            desc.depth
                .map_or_else(vk::PipelineDepthStencilStateCreateInfo::default, |depth| {
                    let mut info = vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(true)
                        .depth_write_enable(depth.write_enabled)
                        .depth_compare_op(convert::compare(depth.compare));
                    if let Some(stencil) = depth.stencil {
                        let face = |face: dirk_rhi::StencilFaceState| {
                            vk::StencilOpState::default()
                                .fail_op(convert::stencil_op(face.fail_op))
                                .pass_op(convert::stencil_op(face.pass_op))
                                .depth_fail_op(convert::stencil_op(face.depth_fail_op))
                                .compare_op(convert::compare(face.compare))
                                .compare_mask(stencil.read_mask)
                                .write_mask(stencil.write_mask)
                        };
                        info = info
                            .stencil_test_enable(true)
                            .front(face(stencil.front))
                            .back(face(stencil.back));
                    } else {
                        info = info.stencil_test_enable(false);
                    }
                    info
                });
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = desc
            .color_formats
            .iter()
            .copied()
            .map(convert::format)
            .collect::<Vec<_>>();
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(
                desc.depth
                    .map_or(vk::Format::UNDEFINED, |depth| convert::format(depth.format)),
            );
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .depth_stencil_state(&depth)
            .dynamic_state(&dynamic)
            .layout(desc.layout.raw())
            .push_next(&mut rendering);
        let raw = unsafe {
            context.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, error)| vk_error(error))?[0];
        Ok(Self(Arc::new(GraphicsPipelineInner {
            context: context.clone(),
            raw,
            _layout: desc.layout.clone(),
        })))
    }

    #[must_use]
    /// Returns the native graphics-pipeline handle.
    pub fn raw(&self) -> vk::Pipeline {
        self.0.raw
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Drop for GraphicsPipelineInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::Pipeline(self.raw));
    }
}

macro_rules! sync_resource {
    ($public:ident, $inner:ident, $raw:ty, $garbage:ident, $resource_doc:literal, $raw_doc:literal) => {
        #[derive(Clone)]
        #[doc = $resource_doc]
        pub struct $public(Arc<$inner>);

        struct $inner {
            context: Arc<Context>,
            raw: $raw,
        }

        impl $public {
            #[must_use]
            #[doc = $raw_doc]
            pub fn raw(&self) -> $raw {
                self.0.raw
            }
        }

        impl Drop for $inner {
            fn drop(&mut self) {
                self.context.retire(Garbage::$garbage(self.raw));
            }
        }
    };
}

sync_resource!(
    VulkanTimelineSemaphore,
    TimelineSemaphoreInner,
    vk::Semaphore,
    Semaphore,
    "Shared Vulkan timeline semaphore.",
    "Returns the native semaphore handle."
);

/// Shared Vulkan submission fence.
#[derive(Clone)]
pub struct VulkanFence(Arc<FenceInner>);

struct FenceInner {
    context: Arc<Context>,
    raw: vk::Fence,
    retained: Mutex<Vec<Retained>>,
}

impl VulkanFence {
    /// Returns the native fence handle.
    #[must_use]
    pub fn raw(&self) -> vk::Fence {
        self.0.raw
    }

    pub(crate) fn retain_resources(&self, resources: impl IntoIterator<Item = Retained>) {
        self.0.retained.lock().extend(resources);
    }

    pub(crate) fn release_resources(&self) {
        self.0.retained.lock().clear();
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl Fence for VulkanFence {
    fn wait(&self, timeout_ns: u64) -> Result<()> {
        unsafe {
            self.0
                .context
                .device
                .wait_for_fences(std::slice::from_ref(&self.0.raw), true, timeout_ns)
                .map_err(vk_error)
        }
    }

    fn reset(&self) -> Result<()> {
        self.release_resources();
        unsafe {
            self.0
                .context
                .device
                .reset_fences(std::slice::from_ref(&self.0.raw))
                .map_err(vk_error)
        }
    }
}

impl Drop for FenceInner {
    fn drop(&mut self) {
        self.context.retire(Garbage::Fence(self.raw));
    }
}

impl VulkanFence {
    pub(crate) fn create(context: &Arc<Context>, signaled: bool) -> Result<Self> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let raw = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)
        }
        .map_err(vk_error)?;
        Ok(Self(Arc::new(FenceInner {
            context: context.clone(),
            raw,
            retained: Mutex::new(Vec::new()),
        })))
    }
}

impl VulkanTimelineSemaphore {
    pub(crate) fn create(context: &Arc<Context>, initial_value: u64) -> Result<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let raw =
            unsafe { context.device.create_semaphore(&create_info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(TimelineSemaphoreInner {
            context: context.clone(),
            raw,
        })))
    }

    pub(crate) fn retain(&self) -> Retained {
        self.0.clone()
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.0.context
    }
}

impl TimelineSemaphore for VulkanTimelineSemaphore {
    fn wait(&self, value: u64, timeout_ns: u64) -> Result<()> {
        let semaphores = [self.0.raw];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            self.0
                .context
                .device
                .wait_semaphores(&wait_info, timeout_ns)
                .map_err(vk_error)
        }
    }

    fn value(&self) -> Result<u64> {
        unsafe {
            self.0
                .context
                .device
                .get_semaphore_counter_value(self.0.raw)
                .map_err(vk_error)
        }
    }
}
