#![doc = include_str!("../README.md")]

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, parse_macro_input, parse_quote};

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
/// commands may insert or remove it. `#[component(derived)]` makes it the
/// output of a derivation instead, so only that derivation writes it.
/// Generics on the type are respected:
///
/// ```rust
/// # extern crate self as dirk_universe;
/// # pub mod components {
/// #     pub trait Component: std::fmt::Debug + Send + 'static { type Mutability; }
/// #     pub enum Mutable {}
/// #     pub enum Derived {}
/// # }
/// # use components::Component;
/// # use dirk_proc::Component;
/// #[derive(Debug, Component)]
/// struct Transform { position: (i32, i32) }
///
/// #[derive(Debug, PartialEq, Component)]
/// #[component(derived)]
/// struct Speed(f32);
/// # fn main() {}
/// ```
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let mut derived = false;
    for attr in input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("component"))
    {
        let parsed = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("derived") {
                derived = true;
                Ok(())
            } else {
                Err(meta.error("expected `derived`"))
            }
        });
        if let Err(error) = parsed {
            return error.to_compile_error().into();
        }
    }
    let mutability = if derived {
        quote!(::dirk_universe::components::Derived)
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

/// Derive `dirk_shaders`' `HeapIndices` trait for a struct, listing the heap
/// indices of every field, each of whose types must implement it too. Push
/// data and the elements of frame allocations derive it, so the RHI keeps
/// the resources they index until the work reading them completes. Generic
/// structs list the fields of their type parameters, which must implement
/// it:
///
/// ```rust
/// # extern crate self as dirk_shaders;
/// # pub mod bindless {
/// #     pub enum HeapIndex { Sampler(u32) }
/// #     pub unsafe trait HeapIndices {
/// #         fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V);
/// #     }
/// #     unsafe impl HeapIndices for u32 {
/// #         fn heap_indices<V: FnMut(HeapIndex)>(&self, _: &mut V) {}
/// #     }
/// #     #[derive(Clone, Copy)]
/// #     pub struct SamplerIndex(pub u32);
/// #     unsafe impl HeapIndices for SamplerIndex {
/// #         fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
/// #             visit(HeapIndex::Sampler(self.0));
/// #         }
/// #     }
/// # }
/// # use bindless::{HeapIndex, HeapIndices, SamplerIndex};
/// # use dirk_proc::HeapIndices;
/// #[derive(HeapIndices)]
/// struct Push<T> { sampler: SamplerIndex, value: T }
///
/// # fn main() {
/// let push = Push { sampler: SamplerIndex(3), value: 7_u32 };
/// let mut samplers = Vec::new();
/// push.heap_indices(&mut |HeapIndex::Sampler(slot)| samplers.push(slot));
/// assert_eq!(samplers, [3]);
/// # }
/// ```
///
/// Enums and unions hold indices the derive cannot list, so it rejects them:
///
/// ```compile_fail
/// # use dirk_proc::HeapIndices;
/// #[derive(HeapIndices)]
/// enum Push { Texture(u32) }
/// ```
#[proc_macro_derive(HeapIndices)]
pub fn derive_heap_indices(input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);
    let Data::Struct(data) = &input.data else {
        return syn::Error::new(
            input.ident.span(),
            "`HeapIndices` can only be derived for structs",
        )
        .to_compile_error()
        .into();
    };
    let trait_path = quote!(::dirk_shaders::bindless::HeapIndices);
    let fields: Vec<_> = data.fields.members().collect();
    if !input.generics.params.is_empty() {
        let types: Vec<_> = data.fields.iter().map(|field| field.ty.clone()).collect();
        let where_clause = input.generics.make_where_clause();
        for ty in types {
            where_clause.predicates.push(parse_quote!(#ty: #trait_path));
        }
    }
    // Structs without fields visit nothing.
    let visit = if fields.is_empty() {
        quote!(_visit)
    } else {
        quote!(visit)
    };
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    quote! {
        // SAFETY: lists the indices of every field, each of which lists its
        // own.
        unsafe impl #impl_generics #trait_path for #name #ty_generics #where_clause {
            fn heap_indices<V: FnMut(::dirk_shaders::bindless::HeapIndex)>(
                &self,
                #visit: &mut V,
            ) {
                #(#trait_path::heap_indices(&self.#fields, #visit);)*
            }
        }
    }
    .into()
}
