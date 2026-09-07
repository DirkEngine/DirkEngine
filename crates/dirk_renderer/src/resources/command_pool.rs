//! Typed command allocation; the shared RHI owns native pools through completion.
use crate::{
    Result,
    resources::{ActiveBackend, ActiveRhi},
};
use dirk_rhi::{CommandEncoder, QueueKind};
pub use dirk_rhi::{CopyQueue as Transfer, Graphics};
use std::{marker::PhantomData, sync::Arc};
pub type CommandBuffer = CommandEncoder<ActiveBackend>;

pub struct CommandPool<Q: QueueKind> {
    rhi: Arc<ActiveRhi>,
    kind: PhantomData<Q>,
    // Temporary compatibility allocator for egui-ash's synchronous uploads.
    #[cfg(renderer_editor)]
    legacy: dirk_rhi_vulkan::VulkanCommandPool,
}
impl<Q: QueueKind> CommandPool<Q> {
    #[cfg_attr(
        not(renderer_editor),
        allow(
            clippy::unnecessary_wraps,
            reason = "the editor compatibility allocator is fallible; keep one feature-independent signature"
        )
    )]
    pub fn build(rhi: &Arc<ActiveRhi>) -> Result<Self> {
        Ok(Self {
            rhi: rhi.clone(), kind: PhantomData,
            #[cfg(renderer_editor)]
            // SAFETY: renderer initialization is exclusive; egui waits for its uploads.
            legacy: unsafe { dirk_rhi::Backend::create_command_pool(rhi.native(), dirk_rhi::QueueType::Graphics)? },
        })
    }
    pub fn begin(&self, label: &str) -> Result<CommandEncoder<ActiveBackend, Q>> {
        Ok(self.rhi.create_encoder(label)?)
    }
    pub fn begin_single_time(&self) -> Result<CommandEncoder<ActiveBackend, Q>> {
        self.begin("renderer upload")
    }
    pub fn submit_and_wait(&self, command: CommandEncoder<ActiveBackend, Q>) -> Result<()> {
        self.rhi
            .queue::<Q>()
            .submit(vec![command.finish()?], &dirk_rhi::SubmitInfo::default())?
            .wait(u64::MAX)?;
        Ok(())
    }
    #[cfg(renderer_editor)]
    pub fn raw(&self) -> ash::vk::CommandPool {
        self.legacy.raw()
    }
}
