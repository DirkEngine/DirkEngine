use std::{marker::PhantomData, sync::Arc};

#[cfg(feature = "editor")]
use std::ops::Deref;

#[cfg(feature = "editor")]
use ash::vk;
use dirk_rhi::{Backend as _, CommandBuffer as _, Fence as _, QueueType, Submission};

use crate::{
    Result,
    resources::{ActiveCommandBuffer, ActiveCommandPool, ActiveRhi},
};

#[derive(Debug)]
pub struct Graphics;
#[derive(Debug)]
pub struct Transfer;

/// Queue marker implemented by command-pool kinds used by the renderer.
pub trait Pool {
    /// Semantic RHI queue used by this pool.
    const QUEUE: QueueType;
}

impl Pool for Graphics {
    const QUEUE: QueueType = QueueType::Graphics;
}

impl Pool for Transfer {
    const QUEUE: QueueType = QueueType::Copy;
}

/// Typed renderer wrapper around an RHI command pool.
pub struct CommandPool<Type: Pool> {
    rhi: Arc<ActiveRhi>,
    inner: ActiveCommandPool,
    pool_type: PhantomData<Type>,
}

impl<Type: Pool> CommandPool<Type> {
    /// Creates a resettable command pool for this marker's queue.
    pub fn build(rhi: &Arc<ActiveRhi>) -> Result<Self> {
        Ok(Self {
            inner: rhi.create_command_pool(Type::QUEUE)?,
            rhi: rhi.clone(),
            pool_type: PhantomData,
        })
    }

    #[cfg(feature = "editor")]
    pub fn raw(&self) -> vk::CommandPool {
        self.inner.raw()
    }

    pub fn allocate_buffer(&self) -> Result<CommandBuffer> {
        let inner = self.rhi.create_command_buffer(&self.inner)?;
        Ok(CommandBuffer {
            #[cfg(feature = "editor")]
            raw: inner.raw(),
            inner,
            rhi: self.rhi.clone(),
            queue: Type::QUEUE,
        })
    }

    pub fn begin_single_time(&self) -> Result<CommandBuffer> {
        let mut command = self.allocate_buffer()?;
        command.inner.begin("immediate renderer command", true)?;
        Ok(command)
    }
}

/// Renderer command buffer backed by the active RHI.
pub struct CommandBuffer {
    #[cfg(feature = "editor")]
    raw: vk::CommandBuffer,
    inner: ActiveCommandBuffer,
    rhi: Arc<ActiveRhi>,
    queue: QueueType,
}

impl CommandBuffer {
    pub fn begin(&mut self, label: &str) -> Result<()> {
        self.inner.begin(label, false)?;
        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        self.inner.end()?;
        Ok(())
    }

    pub(crate) fn rhi_mut(&mut self) -> &mut ActiveCommandBuffer {
        &mut self.inner
    }

    pub(crate) fn rhi(&self) -> &ActiveCommandBuffer {
        &self.inner
    }

    /// Ends, submits, and waits for a short-lived command buffer.
    pub fn end_and_submit(&mut self) -> Result<()> {
        self.inner.end()?;
        let fence = self.rhi.create_fence(false)?;
        self.rhi.submit(
            self.queue,
            &Submission {
                command_buffers: &[&self.inner],
                surface_frames: &[],
                wait_timelines: &[],
                signal_timelines: &[],
                fence: &fence,
            },
        )?;
        fence.wait(u64::MAX)?;
        Ok(())
    }
}

#[cfg(feature = "editor")]
impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
