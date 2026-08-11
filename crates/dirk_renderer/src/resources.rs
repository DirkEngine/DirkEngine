#![allow(unsafe_code)]

pub mod buffer;
pub mod command_pool;
pub mod descriptors;
pub mod device;
pub mod image;
pub mod swapchain;
pub mod sync;

pub(crate) type ActiveRhi = dirk_rhi_vulkan::VulkanBackend;

pub(crate) type ActiveBuffer = <ActiveRhi as dirk_rhi::Backend>::Buffer;
pub(crate) type ActiveCommandBuffer = <ActiveRhi as dirk_rhi::Backend>::CommandBuffer;
pub(crate) type ActiveCommandPool = <ActiveRhi as dirk_rhi::Backend>::CommandPool;
pub(crate) type ActiveFence = <ActiveRhi as dirk_rhi::Backend>::Fence;
pub(crate) type ActiveGraphicsPipeline = <ActiveRhi as dirk_rhi::Backend>::GraphicsPipeline;
pub(crate) type ActiveImage = <ActiveRhi as dirk_rhi::Backend>::Image;
pub(crate) type ActiveImageView = <ActiveRhi as dirk_rhi::Backend>::ImageView;
pub(crate) type ActiveBindGroup = <ActiveRhi as dirk_rhi::Backend>::BindGroup;
pub(crate) type ActivePipelineLayout = <ActiveRhi as dirk_rhi::Backend>::PipelineLayout;
pub(crate) type ActiveSampler = <ActiveRhi as dirk_rhi::Backend>::Sampler;
pub(crate) type ActiveShader = <ActiveRhi as dirk_rhi::Backend>::Shader;
pub(crate) type ActiveSurface = <ActiveRhi as dirk_rhi::Backend>::Surface;
pub(crate) type ActiveSurfaceFrame = <ActiveRhi as dirk_rhi::Backend>::SurfaceFrame;
pub(crate) type ActiveSwapchain = <ActiveRhi as dirk_rhi::Backend>::Swapchain;
pub(crate) type ActiveTimelineSemaphore = <ActiveRhi as dirk_rhi::Backend>::TimelineSemaphore;
