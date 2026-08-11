use dirk_rhi::BindGroupLayoutEntry;

/// Type-level description of one renderer bind-group layout.
pub trait SetLayout {
    const BINDINGS: &'static [BindGroupLayoutEntry];
}
