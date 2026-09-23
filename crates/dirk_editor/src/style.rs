//! Default editor theme and palette.

use dirk_engine::editor::EditorStyle;

/// Color palette for the default Dirk editor theme.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditorPalette {
    /// Main editor background.
    pub background: egui::Color32,
    /// Primary panel fill.
    pub panel: egui::Color32,
    /// Alternate panel fill.
    pub panel_alt: egui::Color32,
    /// Floating surface fill.
    pub surface: egui::Color32,
    /// Emphasized floating surface fill.
    pub surface_high: egui::Color32,
    /// Default interactive control fill.
    pub control: egui::Color32,
    /// Hovered interactive control fill.
    pub control_hovered: egui::Color32,
    /// Active control fill for specialized controls.
    pub control_active: egui::Color32,
    /// Low-contrast border color.
    pub stroke_subtle: egui::Color32,
    /// High-contrast border color.
    pub stroke_strong: egui::Color32,
    /// Default text color.
    pub text: egui::Color32,
    /// Muted text color.
    pub text_muted: egui::Color32,
    /// Bright text color.
    pub text_bright: egui::Color32,
    /// Primary accent color.
    pub accent: egui::Color32,
    /// Hovered accent color.
    pub accent_hovered: egui::Color32,
    /// Selection fill color.
    pub selection: egui::Color32,
    /// Warning text color.
    pub warn: egui::Color32,
    /// Error text color.
    pub error: egui::Color32,
}

impl Default for EditorPalette {
    fn default() -> Self {
        Self {
            background: color(0x0d, 0x0f, 0x11),
            panel: color(0x16, 0x18, 0x1b),
            panel_alt: color(0x1d, 0x20, 0x24),
            surface: color(0x20, 0x23, 0x27),
            surface_high: color(0x28, 0x2c, 0x32),
            control: color(0x25, 0x28, 0x2d),
            control_hovered: color(0x30, 0x35, 0x3c),
            control_active: color(0x2f, 0x56, 0x68),
            stroke_subtle: color(0x2c, 0x30, 0x35),
            stroke_strong: color(0x4a, 0x51, 0x5b),
            text: color(0xd0, 0xd5, 0xdc),
            text_muted: color(0x8c, 0x94, 0x9d),
            text_bright: color(0xf0, 0xf3, 0xf6),
            accent: color(0x58, 0x8a, 0xa8),
            accent_hovered: color(0x7a, 0xa7, 0xc0),
            selection: color(0x2b, 0x59, 0x73),
            warn: color(0xd8, 0x9a, 0x3a),
            error: color(0xe0, 0x62, 0x5c),
        }
    }
}

/// Returns the default compact dark editor style.
#[must_use]
pub fn default_editor_style() -> EditorStyle {
    EditorPalette::default().into()
}

impl From<EditorPalette> for EditorStyle {
    fn from(palette: EditorPalette) -> Self {
        Self::new(move |ctx| {
            ctx.set_theme(egui::Theme::Dark);

            let style = editor_egui_style(palette);

            ctx.set_style_of(egui::Theme::Dark, style.clone());
            ctx.set_style_of(egui::Theme::Light, style);
        })
    }
}

impl From<&EditorPalette> for EditorStyle {
    fn from(palette: &EditorPalette) -> Self {
        Self::from(*palette)
    }
}

fn editor_egui_style(palette: EditorPalette) -> egui::Style {
    let mut style = egui::Style::default();
    apply_compact_spacing(&mut style.spacing);
    apply_compact_text_styles(&mut style.text_styles);
    style.visuals = editor_visuals(palette);
    style
}

fn apply_compact_spacing(spacing: &mut egui::style::Spacing) {
    spacing.item_spacing = egui::vec2(4.0, 2.0);
    spacing.window_margin = egui::Margin::symmetric(6, 5);
    spacing.menu_margin = egui::Margin::symmetric(5, 3);
    spacing.button_padding = egui::vec2(5.0, 2.0);
    spacing.indent = 12.0;
    spacing.interact_size = egui::vec2(22.0, 18.0);
    spacing.slider_width = 96.0;
    spacing.slider_rail_height = 4.0;
    spacing.combo_width = 118.0;
    spacing.text_edit_width = 220.0;
    spacing.icon_width = 14.0;
    spacing.icon_width_inner = 8.0;
    spacing.icon_spacing = 4.0;
    spacing.menu_spacing = 2.0;
    spacing.scroll.bar_width = 8.0;
    spacing.scroll.handle_min_length = 24.0;
    spacing.scroll.bar_inner_margin = 2.0;
    spacing.scroll.bar_outer_margin = 1.0;
}

fn apply_compact_text_styles(
    text_styles: &mut std::collections::BTreeMap<egui::TextStyle, egui::FontId>,
) {
    text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(9.0, egui::FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(12.0, egui::FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(12.0, egui::FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(12.0, egui::FontFamily::Monospace),
    );
    text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
}

fn editor_visuals(palette: EditorPalette) -> egui::Visuals {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(palette.text);
    visuals.window_fill = palette.surface;
    visuals.panel_fill = palette.background;
    visuals.window_stroke = egui::Stroke::new(1.0_f32, palette.stroke_subtle);
    visuals.window_corner_radius = egui::CornerRadius::same(2);
    visuals.menu_corner_radius = egui::CornerRadius::same(2);
    visuals.window_shadow = egui::Shadow {
        offset: [0, 3],
        blur: 8,
        spread: 0,
        color: egui::Color32::from_black_alpha(95),
    };
    visuals.popup_shadow = egui::Shadow {
        offset: [0, 2],
        blur: 6,
        spread: 0,
        color: egui::Color32::from_black_alpha(90),
    };
    visuals.faint_bg_color = palette.panel_alt;
    visuals.extreme_bg_color = palette.background;
    visuals.text_edit_bg_color = Some(palette.panel);
    visuals.code_bg_color = palette.panel_alt;
    visuals.selection.bg_fill = palette.selection;
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, palette.text_bright);
    visuals.hyperlink_color = palette.accent_hovered;
    visuals.warn_fg_color = palette.warn;
    visuals.error_fg_color = palette.error;
    visuals.weak_text_color = Some(palette.text_muted);
    visuals.striped = true;
    visuals.slider_trailing_fill = true;
    visuals.button_frame = true;
    visuals.collapsing_header_frame = false;
    visuals.indent_has_left_vline = true;
    visuals.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.6 };
    apply_widget_visuals(&mut visuals.widgets, palette);
    visuals
}

fn apply_widget_visuals(widgets: &mut egui::style::Widgets, palette: EditorPalette) {
    widgets.noninteractive = widget_visuals(
        palette.panel,
        palette.surface,
        palette.stroke_subtle,
        palette.text_muted,
    );
    widgets.inactive = widget_visuals(
        palette.control,
        palette.control,
        palette.stroke_subtle,
        palette.text,
    );
    widgets.hovered = widget_visuals(
        palette.control_hovered,
        palette.control_hovered,
        palette.stroke_strong,
        palette.text_bright,
    );
    widgets.active = widget_visuals(
        palette.control_active,
        palette.control_active,
        palette.accent,
        palette.text_bright,
    );
    widgets.open = widget_visuals(
        palette.surface_high,
        palette.surface_high,
        palette.accent,
        palette.text_bright,
    );
}

fn widget_visuals(
    bg_fill: egui::Color32,
    weak_bg_fill: egui::Color32,
    bg_stroke: egui::Color32,
    fg_stroke: egui::Color32,
) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill,
        weak_bg_fill,
        bg_stroke: egui::Stroke::new(1.0_f32, bg_stroke),
        corner_radius: egui::CornerRadius::same(2),
        fg_stroke: egui::Stroke::new(1.0_f32, fg_stroke),
        expansion: 0.0,
    }
}

fn color(r: u8, g: u8, b: u8) -> egui::Color32 {
    egui::Color32::from_rgb(r, g, b)
}
