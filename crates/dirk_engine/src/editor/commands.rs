//! Editor commands. Menus can submit commands that will be run by the editor.

use std::sync::mpsc::Sender;

use crate::editor::EditorWindowId;

/// Sends editor runtime commands from UI capabilities.
#[derive(Clone)]
pub struct EditorCommandSender {
    commands: Sender<EditorCommand>,
}

impl EditorCommandSender {
    pub(crate) fn new(commands: Sender<EditorCommand>) -> Self {
        Self { commands }
    }

    /// Requests that an editor window be opened.
    pub fn open_window(&self, id: EditorWindowId) {
        let _ = self.commands.send(EditorCommand::OpenWindow(id));
    }
}

pub(crate) enum EditorCommand {
    OpenWindow(EditorWindowId),
}
