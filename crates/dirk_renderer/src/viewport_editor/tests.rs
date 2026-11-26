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
