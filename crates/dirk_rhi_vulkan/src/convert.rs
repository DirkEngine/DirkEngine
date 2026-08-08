use ash::vk;
use dirk_rhi::{
    AddressMode, BindingType, BufferUsages, CompareOp, CullMode, FilterMode, Format, FrontFace,
    ImageAspects, ImageState, ImageUsages, ImageViewType, IndexFormat, MemoryDomain,
    PipelineStages, PrimitiveTopology, QueueType, SampleCount, ShaderStages, StoreOp,
    VertexStepMode,
};
use gpu_allocator::MemoryLocation;

pub(crate) fn format(value: Format) -> vk::Format {
    match value {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        Format::Rg32Float => vk::Format::R32G32_SFLOAT,
        Format::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
        Format::Depth16Unorm => vk::Format::D16_UNORM,
        Format::Depth24UnormStencil8 => vk::Format::D24_UNORM_S8_UINT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
        Format::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

pub(crate) fn rhi_format(value: vk::Format) -> Option<Format> {
    Some(match value {
        vk::Format::R8G8B8A8_UNORM => Format::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => Format::Rgba8Srgb,
        vk::Format::B8G8R8A8_UNORM => Format::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => Format::Bgra8Srgb,
        vk::Format::R32G32_SFLOAT => Format::Rg32Float,
        vk::Format::R32G32B32_SFLOAT => Format::Rgb32Float,
        vk::Format::D16_UNORM => Format::Depth16Unorm,
        vk::Format::D24_UNORM_S8_UINT => Format::Depth24UnormStencil8,
        vk::Format::D32_SFLOAT => Format::Depth32Float,
        vk::Format::D32_SFLOAT_S8_UINT => Format::Depth32FloatStencil8,
        _ => return None,
    })
}

pub(crate) fn samples(value: SampleCount) -> vk::SampleCountFlags {
    match value {
        SampleCount::One => vk::SampleCountFlags::TYPE_1,
        SampleCount::Two => vk::SampleCountFlags::TYPE_2,
        SampleCount::Four => vk::SampleCountFlags::TYPE_4,
        SampleCount::Eight => vk::SampleCountFlags::TYPE_8,
    }
}

pub(crate) fn buffer_usage(value: BufferUsages) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    for (rhi, vulkan) in [
        (BufferUsages::COPY_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
        (BufferUsages::COPY_DST, vk::BufferUsageFlags::TRANSFER_DST),
        (BufferUsages::VERTEX, vk::BufferUsageFlags::VERTEX_BUFFER),
        (BufferUsages::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
        (BufferUsages::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
        (BufferUsages::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
    ] {
        if value.contains(rhi) {
            flags |= vulkan;
        }
    }
    flags
}

pub(crate) fn image_usage(value: ImageUsages) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    for (rhi, vulkan) in [
        (ImageUsages::COPY_SRC, vk::ImageUsageFlags::TRANSFER_SRC),
        (ImageUsages::COPY_DST, vk::ImageUsageFlags::TRANSFER_DST),
        (ImageUsages::SAMPLED, vk::ImageUsageFlags::SAMPLED),
        (ImageUsages::STORAGE, vk::ImageUsageFlags::STORAGE),
        (
            ImageUsages::COLOR_ATTACHMENT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ),
        (
            ImageUsages::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
        (
            ImageUsages::TRANSIENT_ATTACHMENT,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        ),
    ] {
        if value.contains(rhi) {
            flags |= vulkan;
        }
    }
    flags
}

pub(crate) fn aspects(value: ImageAspects) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    for (rhi, vulkan) in [
        (ImageAspects::COLOR, vk::ImageAspectFlags::COLOR),
        (ImageAspects::DEPTH, vk::ImageAspectFlags::DEPTH),
        (ImageAspects::STENCIL, vk::ImageAspectFlags::STENCIL),
    ] {
        if value.contains(rhi) {
            flags |= vulkan;
        }
    }
    flags
}

pub(crate) fn shader_stages(value: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    for (rhi, vulkan) in [
        (ShaderStages::VERTEX, vk::ShaderStageFlags::VERTEX),
        (ShaderStages::FRAGMENT, vk::ShaderStageFlags::FRAGMENT),
        (ShaderStages::COMPUTE, vk::ShaderStageFlags::COMPUTE),
    ] {
        if value.contains(rhi) {
            flags |= vulkan;
        }
    }
    flags
}

pub(crate) fn pipeline_stages(value: PipelineStages) -> vk::PipelineStageFlags2 {
    if value == PipelineStages::ALL {
        return vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    let mut flags = vk::PipelineStageFlags2::empty();
    for (rhi, vulkan) in [
        (PipelineStages::COPY, vk::PipelineStageFlags2::COPY),
        (
            PipelineStages::VERTEX,
            vk::PipelineStageFlags2::VERTEX_SHADER,
        ),
        (
            PipelineStages::FRAGMENT,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
        ),
        (
            PipelineStages::COLOR_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        ),
        (
            PipelineStages::COMPUTE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        ),
    ] {
        if value.contains(rhi) {
            flags |= vulkan;
        }
    }
    flags
}

pub(crate) fn image_state(
    value: ImageState,
) -> (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout) {
    match value {
        ImageState::Undefined => (
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
        ),
        ImageState::CopySource => (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        ImageState::CopyDestination => (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        ImageState::ShaderRead => (
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        ImageState::ShaderWrite => (
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        ImageState::ColorAttachment => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        ImageState::DepthStencilAttachment => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        ImageState::Present => (
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::PRESENT_SRC_KHR,
        ),
    }
}

pub(crate) fn memory(value: MemoryDomain) -> MemoryLocation {
    match value {
        MemoryDomain::Device => MemoryLocation::GpuOnly,
        MemoryDomain::Upload => MemoryLocation::CpuToGpu,
        MemoryDomain::Readback => MemoryLocation::GpuToCpu,
    }
}

pub(crate) fn binding(value: BindingType) -> vk::DescriptorType {
    match value {
        BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::SampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        BindingType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
    }
}

pub(crate) fn view_type(value: ImageViewType) -> vk::ImageViewType {
    match value {
        ImageViewType::TwoD => vk::ImageViewType::TYPE_2D,
        ImageViewType::TwoDArray => vk::ImageViewType::TYPE_2D_ARRAY,
        ImageViewType::Cube => vk::ImageViewType::CUBE,
    }
}

pub(crate) fn filter(value: FilterMode) -> vk::Filter {
    match value {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

pub(crate) fn mipmap_filter(value: FilterMode) -> vk::SamplerMipmapMode {
    match value {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

pub(crate) fn address(value: AddressMode) -> vk::SamplerAddressMode {
    match value {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

pub(crate) fn topology(value: PrimitiveTopology) -> vk::PrimitiveTopology {
    match value {
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
    }
}

pub(crate) fn cull(value: CullMode) -> vk::CullModeFlags {
    match value {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

pub(crate) fn front_face(value: FrontFace) -> vk::FrontFace {
    match value {
        FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
        FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
    }
}

pub(crate) fn compare(value: CompareOp) -> vk::CompareOp {
    match value {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

pub(crate) fn index(value: IndexFormat) -> vk::IndexType {
    match value {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

pub(crate) fn input_rate(value: VertexStepMode) -> vk::VertexInputRate {
    match value {
        VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
        VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
    }
}

pub(crate) fn store(value: StoreOp) -> vk::AttachmentStoreOp {
    match value {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

pub(crate) fn queue(value: QueueType) -> QueueKind {
    match value {
        QueueType::Graphics => QueueKind::Graphics,
        QueueType::Compute => QueueKind::Compute,
        QueueType::Copy => QueueKind::Copy,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum QueueKind {
    Graphics,
    Compute,
    Copy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_round_trip() {
        for format in [
            Format::Rgba8Unorm,
            Format::Rgba8Srgb,
            Format::Bgra8Unorm,
            Format::Bgra8Srgb,
            Format::Rg32Float,
            Format::Rgb32Float,
            Format::Depth16Unorm,
            Format::Depth24UnormStencil8,
            Format::Depth32Float,
            Format::Depth32FloatStencil8,
        ] {
            assert_eq!(rhi_format(super::format(format)), Some(format));
        }
    }

    #[test]
    fn combined_usage_preserves_each_vulkan_role() {
        let usage =
            buffer_usage(BufferUsages::COPY_DST | BufferUsages::VERTEX | BufferUsages::STORAGE);

        assert!(usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert!(usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER));
        assert!(usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
    }

    #[test]
    fn image_states_map_to_sync2_layouts_and_access() {
        let (stage, access, layout) = image_state(ImageState::ColorAttachment);

        assert_eq!(stage, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        assert!(access.contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        assert_eq!(layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }
}
