use std::{
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    RendererProperties, Result,
    resources::{
        ActiveRhi,
        command_pool::{CommandPool, Graphics, Transfer},
    },
};

/// Shared renderer device state and its command pools.
#[derive(Clone)]
pub struct RenderDevice(Arc<RenderDeviceInner>);

impl Deref for RenderDevice {
    type Target = RenderDeviceInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct RenderDeviceInner {
    /// Pool used for short resource uploads.
    pub transfer_pool: CommandPool<Transfer>,
    /// Pool used for frame and mip-generation commands.
    pub graphics_pool: CommandPool<Graphics>,
    pub properties: RendererProperties,
    current_frame: Arc<AtomicUsize>,
    /// Active render hardware interface. Declared last so it outlives resources above.
    pub rhi: Arc<ActiveRhi>,
}

pub struct FrameCounters {
    pub current_frame: Arc<AtomicUsize>,
}

impl RenderDevice {
    pub fn new(
        rhi: Arc<ActiveRhi>,
        properties: RendererProperties,
        frame_counters: FrameCounters,
    ) -> Result<Self> {
        let transfer_pool = CommandPool::build(&rhi)?;
        let graphics_pool = CommandPool::build(&rhi)?;

        Ok(Self(Arc::new(RenderDeviceInner {
            transfer_pool,
            graphics_pool,
            properties,
            current_frame: frame_counters.current_frame,
            rhi,
        })))
    }

    pub fn current_frame(&self) -> usize {
        self.current_frame.load(Ordering::Relaxed)
    }
}
