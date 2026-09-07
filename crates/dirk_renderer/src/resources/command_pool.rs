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
}
impl<Q: QueueKind> CommandPool<Q> {
    pub fn build(rhi: &Arc<ActiveRhi>) -> Self {
        Self {
            rhi: rhi.clone(),
            kind: PhantomData,
        }
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
}
