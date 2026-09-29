#![doc = include_str!("../README.md")]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Ident, ItemTrait, parse_macro_input};

mod event;
mod universe;

/// Derive the `Event` trait for a `struct` or `enum`.
///
/// Generates a `debug(&self) -> String` implementation that formats the value
/// using a caller-supplied template, or falls back to `"{self:?}"` when none
/// is provided.
///
/// # The `#[event("…")]` attribute
///
/// Attach `#[event("…")]` to the type (for structs) or to individual variants
/// (for enums) to control the debug message.
///
/// ## Named-field structs / variants
///
/// All field names are bound in scope so you can interpolate them:
///
/// ```rust
/// # pub trait Event: Send + Clone + 'static { fn debug(&self) -> String; }
/// # use dirk_proc::Event;
/// #[derive(Event, Clone)]
/// #[event("player {name} joined with {hp} hp")]
/// struct PlayerJoined { name: String, hp: u32 }
/// ```
///
/// ## Tuple (unnamed) structs / variants
///
/// Use positional placeholders `{0}`, `{1}`, … which are rewritten to the
/// internal bindings `_0`, `_1`, …:
///
/// ```rust
/// # pub trait Event: Send + Clone + 'static { fn debug(&self) -> String; }
/// # use dirk_proc::Event;
/// #[derive(Event, Clone)]
/// enum Msg {
///     #[event("moved to ({0}, {1})")]
///     Moved(f32, f32),
/// }
/// ```
///
/// ## Unit structs / variants
///
/// The `{self:?}` fallback (or a static string) works fine:
///
/// ```rust
/// # pub trait Event: Send + Clone + 'static { fn debug(&self) -> String; }
/// # use dirk_proc::Event;
/// #[derive(Event, Clone)]
/// #[event("server stopped")]
/// struct ServerStopped;
/// ```
///
/// ## Unsupported placeholders
///
/// The format string is expanded without positional arguments, so every
/// placeholder must name a field. Implicit positional placeholders (`{}`,
/// `{:?}`), `.*` precision and `N$` width/precision arguments are rejected
/// with an error pointing at the format string:
///
/// ```compile_fail
/// # pub trait Event: Send + Clone + 'static { fn debug(&self) -> String; }
/// # use dirk_proc::Event;
/// #[derive(Event, Clone)]
/// #[event("key pressed: {}")] // use `{0}` instead
/// struct KeyPressed(u32);
/// ```
///
/// ```compile_fail
/// # pub trait Event: Send + Clone + 'static { fn debug(&self) -> String; }
/// # use dirk_proc::Event;
/// #[derive(Event, Clone)]
/// #[event("value {0:.*}")] // use a literal precision such as `{0:.2}`
/// struct Value(usize, f32);
/// ```
#[proc_macro_derive(Event, attributes(event))]
pub fn derive_event(input: proc_macro::TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let result: syn::Result<proc_macro2::TokenStream> = match input.data {
        Data::Enum(ref data) => event::derive_event_enum(&input, data),
        Data::Struct(ref data) => event::derive_event_struct(&input, data),
        Data::Union(_) => Err(syn::Error::new(
            input.ident.span(),
            "`Event` can only be derived for structs and enums",
        )),
    };

    result.unwrap_or_else(|e| e.to_compile_error()).into()
}

/// Derive the `Component` marker trait for any type.
///
/// This is a zero-boilerplate derive that simply emits an empty `impl Component
/// for …` block, respecting any generics on the type:
///
/// ```rust
/// # trait Component {}
/// # use dirk_proc::Component;
/// #[derive(Component, Clone)]
/// struct Transform { position: (i32, i32) }
/// ```
#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    empty_derive(input, &format_ident!("Component"))
}

pub(crate) fn empty_derive(input: TokenStream, trait_ident: &Ident) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #trait_ident for #name #ty_generics #where_clause {}
    }
    .into()
}

/// Generates a type-erased `Any...` type & a storage struct for a
/// trait that has `System` as a supertrait of the ECS system.
///
/// # Example
///
/// ```rust
/// # pub trait System: 'static { fn name() -> &'static str; }
/// # use dirk_proc::system_trait;
/// #[system_trait]
/// pub trait RenderSystem: System {
///     fn render(&self, delta_time: f32);
/// }
///
/// // The macro generates:
/// //
/// //   pub(crate) trait AnyRenderSystem {
/// //       fn render(&self, delta_time: f32);
/// //   }
/// //
/// //   impl<T: RenderSystem> AnyRenderSystem for T { … }
/// //
/// //   pub(crate) struct RenderSystemStorage { … }
/// //   impl RenderSystemStorage {
/// //       pub fn new() -> Self { … }
/// //       pub fn insert(&mut self, system: impl RenderSystem) { … }
/// //       pub fn iter(&self) -> std::slice::Iter<…> { … }
/// //   }
///
/// // Implementing the system:
/// # use dirk_proc::System;
/// #[derive(System)]
/// struct MyRenderer;
///
/// impl RenderSystem for MyRenderer {
///     fn render(&self, delta_time: f32) {
///         // draw stuff
///     }
/// }
///
/// // Registering and iterating systems:
/// let mut storage = RenderSystemStorage::new();
/// storage.push(MyRenderer);
///
/// for system in storage.iter() {
///     system.render(0.016);
/// }
/// ```
#[proc_macro_attribute]
pub fn system_trait(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemTrait);
    universe::generate_system_code(&input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive the `System` marker trait, providing the required `name()` method.
///
/// This derive generates the implementation for `name()` automatically,
/// choosing the system name through the following priority order:
///
/// 1. The value of the `name` key inside a `#[system(name = "…")]` attribute,
///    if present.
/// 2. The identifier of the type itself (stringified), as a fallback.
///
/// # Helper attribute: `#[system(name = "…")]`
///
/// Use this optional attribute to give a system a human-readable name that is
/// different from its Rust type name. This is particularly useful for debug
/// output, logging, and profiling, where the raw type name may be noisy or
/// insufficiently descriptive.
///
/// # Examples
///
/// ## Using the default (type name) as the system name
///
/// ```rust
/// # pub trait System: 'static { fn name() -> &'static str; }
/// # use dirk_proc::System;
/// #[derive(System)]
/// struct PhysicsSystem;
///
/// assert_eq!(PhysicsSystem::name(), "PhysicsSystem");
/// ```
///
/// ## Providing a custom name via the helper attribute
///
/// ```rust
/// # pub trait System: 'static { fn name() -> &'static str; }
/// # use dirk_proc::System;
/// #[derive(System)]
/// #[system(name = "Collision Detection")]
/// struct NarrowPhaseCollision;
///
/// assert_eq!(NarrowPhaseCollision::name(), "Collision Detection");
/// ```
///
/// ## Used together with a `#[system_trait]`-decorated trait
///
/// This derive is designed to be combined with types that implement traits
/// expanded by [`system_trait`]:
///
/// ```rust
/// # pub trait System: 'static { fn name() -> &'static str; }
/// # use dirk_proc::{System, system_trait};
/// #[system_trait]
/// pub trait AudioSystem: System {
///     fn play(&self, clip_id: u32);
/// }
///
/// #[derive(System)]
/// #[system(name = "Spatial Audio")]
/// struct SpatialAudioSystem;
///
/// impl AudioSystem for SpatialAudioSystem {
///     fn play(&self, clip_id: u32) { /* … */ }
/// }
///
/// assert_eq!(SpatialAudioSystem::name(), "Spatial Audio");
/// ```
///
/// [`system_trait`]: macro@system_trait
#[proc_macro_derive(System, attributes(system))]
pub fn derive_system(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    universe::derive_system(&input).into()
}
