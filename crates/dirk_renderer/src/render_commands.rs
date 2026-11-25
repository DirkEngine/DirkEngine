//! This crate has the render commands system. It allows threads
//! to submit functions that will be run on the renderer. This
//! notable allows systems to interact with the renderer.
//!
//! These types are private as they are for internal renderer use
//! only. They should not be used by other engine systems.
use std::sync::mpsc::{self, Receiver, Sender};

use crate::{Renderer, Result};

type RenderCommand = Box<dyn FnOnce(&mut Renderer) -> Result<()> + Send + 'static>;

pub struct RenderCommandSender {
    tx: Sender<RenderCommand>,
}

impl RenderCommandSender {
    pub fn enqueue_command<F>(&self, command: F)
    where
        F: FnOnce(&mut Renderer) -> Result<()> + Send + 'static,
    {
        let _ = self.tx.send(Box::new(command));
    }
}

impl Clone for RenderCommandSender {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

pub struct RenderCommandReceiver {
    rx: Receiver<RenderCommand>,
}

impl RenderCommandReceiver {
    pub fn collect(&self) -> Vec<RenderCommand> {
        self.rx.try_iter().collect()
    }
}

pub fn channel() -> (RenderCommandSender, RenderCommandReceiver) {
    let (tx, rx) = mpsc::channel();
    (RenderCommandSender { tx }, RenderCommandReceiver { rx })
}
