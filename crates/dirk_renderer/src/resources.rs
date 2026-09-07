#![allow(unsafe_code)]

pub mod buffer;
pub mod command_pool;
pub mod descriptors;
pub mod device;
pub mod image;
pub mod swapchain;
pub mod sync;
pub mod upload;

#[cfg(target_vendor = "apple")]
pub(crate) type ActiveBackend = dirk_rhi_metal::MetalBackend;
#[cfg(not(target_vendor = "apple"))]
pub(crate) type ActiveBackend = dirk_rhi_vulkan::VulkanBackend;
pub(crate) type ActiveRhi = dirk_rhi::Rhi<ActiveBackend>;

pub(crate) type ActiveBuffer = <ActiveRhi as dirk_rhi::Api>::Buffer;
pub(crate) type ActiveCommandBuffer = dirk_rhi::CommandEncoder<ActiveBackend>;
pub(crate) type ActiveRecordedCommands = dirk_rhi::RecordedCommands<ActiveBackend>;
pub(crate) type ActiveRenderPass<'a> = dirk_rhi::RenderPass<'a, ActiveBackend>;
pub(crate) type ActiveFence = <ActiveRhi as dirk_rhi::Api>::Fence;
pub(crate) type ActiveGraphicsPipeline = <ActiveRhi as dirk_rhi::Api>::GraphicsPipeline;
pub(crate) type ActiveImage = <ActiveRhi as dirk_rhi::Api>::Image;
pub(crate) type ActiveImageView = <ActiveRhi as dirk_rhi::Api>::ImageView;
pub(crate) type ActiveBindGroup = <ActiveRhi as dirk_rhi::Api>::BindGroup;
pub(crate) type ActivePipelineLayout = <ActiveRhi as dirk_rhi::Api>::PipelineLayout;
pub(crate) type ActiveSampler = <ActiveRhi as dirk_rhi::Api>::Sampler;
pub(crate) type ActiveShader = <ActiveRhi as dirk_rhi::Api>::Shader;
pub(crate) type ActiveSurface = <ActiveRhi as dirk_rhi::Api>::Surface;
pub(crate) type ActiveSurfaceFrame = <ActiveRhi as dirk_rhi::Api>::SurfaceFrame;
pub(crate) type ActiveSwapchain = <ActiveRhi as dirk_rhi::Api>::Swapchain;
pub(crate) type ActiveTimelineSemaphore = <ActiveRhi as dirk_rhi::Api>::TimelineSemaphore;
