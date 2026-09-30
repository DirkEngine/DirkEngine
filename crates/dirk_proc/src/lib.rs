#![doc = include_str!("../README.md")]

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input};

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

/// Derive `dirk_universe`'s `Component` trait for any type.
///
/// The component is mutable: systems may edit it through `&mut C` queries and
/// commands may insert or remove it. `#[component(read_only)]` makes it
/// read-only instead, so only the engine writes it, as with derived
/// components. Generics on the type are respected:
///
/// ```rust
/// # extern crate self as dirk_universe;
/// # pub mod components {
/// #     pub trait Component { type Mutability; }
/// #     pub enum Mutable {}
/// #     pub enum ReadOnly {}
/// # }
/// # use components::Component;
/// # use dirk_proc::Component;
/// #[derive(Component, Clone)]
/// struct Transform { position: (i32, i32) }
///
/// #[derive(Component, Clone)]
/// #[component(read_only)]
/// struct Speed(f32);
/// # fn main() {}
/// ```
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let mut read_only = false;
    for attr in input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("component"))
    {
        let parsed = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("read_only") {
                read_only = true;
                Ok(())
            } else {
                Err(meta.error("expected `read_only`"))
            }
        });
        if let Err(error) = parsed {
            return error.to_compile_error().into();
        }
    }
    let mutability = if read_only {
        quote!(::dirk_universe::components::ReadOnly)
    } else {
        quote!(::dirk_universe::components::Mutable)
    };
    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics ::dirk_universe::components::Component for #name #ty_generics #where_clause {
            type Mutability = #mutability;
        }
    }
    .into()
}
