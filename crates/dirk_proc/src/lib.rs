#![doc = include_str!("../README.md")]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Ident, parse_macro_input};

mod event;

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
