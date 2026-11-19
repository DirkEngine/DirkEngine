//! Editor panels showing each player's viewport, as bindless textures in
//! egui, and forwarding their input to the player.

use std::{collections::HashMap, sync::Arc};

use dirk_engine::editor::{
    EditorServices, EditorWindowDescriptor, EditorWindowId, VIEWPORT_CATEGORY,
};
use dirk_input::{ButtonState, InputEvent, egui::input_events_from_egui_response};
use dirk_player::{PlayerId, PlayerInputSender};
use dirk_rhi::Rhi;
use parking_lot::Mutex;

use crate::{Result, viewport::Viewport};

/// Extent of a viewport, in pixels.
type Extent = (u32, u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewportEditorEntry {
    /// The viewport output's bindless index, as a user texture.
    texture_id: egui::TextureId,
    extent: Extent,
    ready: bool,
    requested_extent: Option<Extent>,
}

pub struct ViewportEditor {
    state: Arc<Mutex<ViewportEditorState>>,
    input_sender: PlayerInputSender,
    windows: HashMap<PlayerId, EditorWindowId>,
}

impl ViewportEditor {
    pub fn new(input_sender: PlayerInputSender) -> Self {
        Self {
            state: Arc::new(Mutex::new(ViewportEditorState::default())),
            input_sender,
            windows: HashMap::new(),
        }
    }

    /// Adds a panel showing `viewport` for `player`.
    pub fn add_viewport(
        &mut self,
        player: PlayerId,
        viewport: &Viewport,
        editor: &EditorServices,
    ) -> Result<()> {
        if self.windows.contains_key(&player) {
            self.remove_viewport(player, editor);
        }

        self.state.lock().insert(
            player,
            ViewportEditorEntry {
                texture_id: Self::texture_id(viewport)?,
                extent: viewport.extent(),
                ready: false,
                requested_extent: None,
            },
        );

        let state = Arc::clone(&self.state);
        let input_sender = self.input_sender.clone();
        let descriptor = EditorWindowDescriptor {
            title: format!("Viewport {player}"),
            category: VIEWPORT_CATEGORY.to_owned(),
            default_open: true,
            show_in_list: true,
        };
        let window_id = editor.add_window_fn(descriptor, move |ui, _context| {
            draw_viewport_window(ui, &state, &input_sender, player);
            Ok(())
        });
        self.windows.insert(player, window_id);
        Ok(())
    }

    pub fn remove_viewport(&mut self, player: PlayerId, editor: &EditorServices) {
        if let Some(window) = self.windows.remove(&player) {
            editor.remove_window(window);
        }
        self.state.lock().remove(player);
    }

    /// Resizes viewports to the size their panels asked for. Call while a
    /// frame is open: the UI recorded in it may still show old outputs.
    pub fn apply_resize_requests(
        &mut self,
        rhi: &Rhi,
        viewports: &mut HashMap<PlayerId, Viewport>,
    ) -> Result<()> {
        let requests = self.state.lock().take_resize_requests();
        for (player, requested_extent) in requests {
            let Some(viewport) = viewports.get_mut(&player) else {
                continue;
            };
            if viewport.extent() == requested_extent {
                continue;
            }
            viewport.resize(rhi, requested_extent)?;
            self.state.lock().replace_texture(
                player,
                Self::texture_id(viewport)?,
                viewport.extent(),
            );
        }
        Ok(())
    }

    pub fn sync_ready_state(&self, viewports: &HashMap<PlayerId, Viewport>) {
        let mut state = self.state.lock();
        for (player, entry) in state.entries_mut() {
            if let Some(viewport) = viewports.get(player) {
                // Renderable viewports render every frame, before the UI
                // shows them.
                entry.ready = viewport.is_renderable();
                entry.extent = viewport.extent();
            } else {
                entry.ready = false;
            }
        }
    }

    /// The user texture showing `viewport`'s output: its bindless index.
    fn texture_id(viewport: &Viewport) -> Result<egui::TextureId> {
        Ok(egui::TextureId::User(u64::from(viewport.texture()?.slot())))
    }
}

#[derive(Default)]
pub struct ViewportEditorState {
    entries: HashMap<PlayerId, ViewportEditorEntry>,
    previous_pointers: HashMap<PlayerId, egui::Pos2>,
    pointer_captures: HashMap<PlayerId, bool>,
}

impl ViewportEditorState {
    fn insert(&mut self, player: PlayerId, entry: ViewportEditorEntry) {
        self.entries.insert(player, entry);
    }

    fn remove(&mut self, player: PlayerId) {
        self.entries.remove(&player);
    }

    fn entry(&self, player: PlayerId) -> Option<ViewportEditorEntry> {
        self.entries.get(&player).copied()
    }

    fn entries_mut(&mut self) -> impl Iterator<Item = (&PlayerId, &mut ViewportEditorEntry)> {
        self.entries.iter_mut()
    }

    fn request_extent(&mut self, player: PlayerId, extent: Extent) {
        if let Some(entry) = self.entries.get_mut(&player) {
            entry.requested_extent = Some(clamp_extent(extent));
        }
    }

    #[must_use]
    fn take_resize_requests(&mut self) -> Vec<(PlayerId, Extent)> {
        self.entries
            .iter_mut()
            .filter_map(|(player, entry)| {
                entry
                    .requested_extent
                    .take()
                    .map(|extent| (*player, clamp_extent(extent)))
            })
            .collect()
    }

    fn replace_texture(&mut self, player: PlayerId, texture_id: egui::TextureId, extent: Extent) {
        if let Some(entry) = self.entries.get_mut(&player) {
            entry.texture_id = texture_id;
            entry.extent = extent;
            entry.ready = false;
        }
    }

    fn previous_pointer(&self, player: PlayerId) -> Option<egui::Pos2> {
        self.previous_pointers.get(&player).copied()
    }

    fn set_previous_pointer(&mut self, player: PlayerId, position: Option<egui::Pos2>) {
        if let Some(position) = position {
            self.previous_pointers.insert(player, position);
        } else {
            self.previous_pointers.remove(&player);
        }
    }

    fn pointer_captured(&self, player: PlayerId) -> bool {
        self.pointer_captures.get(&player).copied().unwrap_or(false)
    }

    fn pointer_capture_state(&self, player: PlayerId) -> (bool, Option<egui::Pos2>) {
        let captured = self.pointer_captured(player);
        let previous = captured.then(|| self.previous_pointer(player)).flatten();
        (captured, previous)
    }

    fn set_pointer_captured(&mut self, player: PlayerId, captured: bool) {
        if captured {
            self.pointer_captures.insert(player, true);
        } else {
            self.pointer_captures.remove(&player);
        }
    }
}

fn draw_viewport_window(
    ui: &mut egui::Ui,
    state: &Arc<Mutex<ViewportEditorState>>,
    input_sender: &PlayerInputSender,
    player: PlayerId,
) {
    let available = ui.available_size_before_wrap();
    let requested_extent = extent_from_points(available, ui.ctx().pixels_per_point());

    let entry = {
        let mut state = state.lock();
        state.request_extent(player, requested_extent);
        state.entry(player)
    };

    let Some(entry) = entry else {
        ui.centered_and_justified(|ui| {
            ui.label("Waiting for camera");
        });
        return;
    };

    if !entry.ready {
        ui.centered_and_justified(|ui| {
            ui.label("Waiting for camera");
        });
        return;
    }

    let response = ui
        .add(egui::Image::new((entry.texture_id, available)).sense(egui::Sense::click_and_drag()));
    let (was_captured, previous_pointer) = state.lock().pointer_capture_state(player);
    let events = input_events_from_egui_response(ui, &response, previous_pointer);
    for event in &events {
        input_sender.send(player, event.clone());
    }
    let mut captured = was_captured;
    for event in &events {
        match event {
            InputEvent::PointerButton {
                state: ButtonState::Pressed,
                ..
            } => captured = true,
            InputEvent::PointerButton {
                state: ButtonState::Released,
                ..
            }
            | InputEvent::PointerLeft
            | InputEvent::FocusChanged(false) => captured = false,
            InputEvent::Key { .. }
            | InputEvent::PointerMoved { .. }
            | InputEvent::PointerEntered
            | InputEvent::Scroll { .. }
            | InputEvent::FocusChanged(true)
            | InputEvent::Text(_)
            | InputEvent::Ime(_)
            | InputEvent::ModifiersChanged(_) => {}
        }
    }
    let latest_pointer = ui.input(|input| input.pointer.latest_pos());
    let mut state = state.lock();
    state.set_pointer_captured(player, captured);
    if captured {
        state.set_previous_pointer(player, latest_pointer);
    } else {
        state.set_previous_pointer(player, None);
    }
}

fn extent_from_points(size: egui::Vec2, pixels_per_point: f32) -> Extent {
    let pixels_per_point = pixels_per_point.max(f32::EPSILON);
    (
        point_size_to_pixels(size.x, pixels_per_point),
        point_size_to_pixels(size.y, pixels_per_point),
    )
}

fn point_size_to_pixels(points: f32, pixels_per_point: f32) -> u32 {
    if points.is_finite() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (points.max(1.0) * pixels_per_point).round().max(1.0) as u32
        }
    } else {
        1
    }
}

fn clamp_extent((width, height): Extent) -> Extent {
    (width.max(1), height.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn player(index: u32) -> PlayerId {
        PlayerId::default() + index
    }

    #[test]
    fn viewport_window_state_records_requested_extents_clamped_to_at_least_1x1() {
        let mut state = ViewportEditorState::default();
        state.insert(
            player(0),
            ViewportEditorEntry {
                texture_id: egui::TextureId::User(7),
                extent: (640, 480),
                ready: false,
                requested_extent: None,
            },
        );

        state.request_extent(player(0), (0, 0));

        assert_eq!(state.take_resize_requests(), vec![(player(0), (1, 1))]);
    }

    #[test]
    fn removing_a_viewport_clears_shared_state() {
        let mut state = ViewportEditorState::default();
        state.insert(
            player(0),
            ViewportEditorEntry {
                texture_id: egui::TextureId::User(1),
                extent: (1, 1),
                ready: false,
                requested_extent: None,
            },
        );

        state.remove(player(0));

        assert_eq!(state.entry(player(0)), None);
    }

    #[test]
    fn descriptor_window_bookkeeping_maps_one_player_to_one_editor_window_entry() {
        let editor = EditorServices::new();
        let first = editor.add_window_fn(
            EditorWindowDescriptor {
                title: "Viewport 0".to_owned(),
                category: VIEWPORT_CATEGORY.to_owned(),
                default_open: true,
                show_in_list: true,
            },
            |_ui, _context| Ok(()),
        );
        let second = editor.add_window_fn(
            EditorWindowDescriptor {
                title: "Viewport 1".to_owned(),
                category: VIEWPORT_CATEGORY.to_owned(),
                default_open: true,
                show_in_list: true,
            },
            |_ui, _context| Ok(()),
        );
        let mut windows = HashMap::new();

        assert_eq!(windows.insert(player(0), first), None);
        assert_eq!(windows.insert(player(1), second), None);

        assert_eq!(windows.get(&player(0)), Some(&first));
        assert_eq!(windows.get(&player(1)), Some(&second));
        assert_eq!(windows.len(), 2);
    }
}
