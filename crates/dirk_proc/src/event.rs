use std::fmt::Write;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, DataEnum, DataStruct, DeriveInput, Fields, FieldsUnnamed, Ident, LitStr};

pub fn derive_event_enum(input: &DeriveInput, data: &DataEnum) -> syn::Result<TokenStream> {
    let name = &input.ident;
    // Split generics into the three parts needed for an impl block:
    // <T: Any>  |  <T>  |  where T: Any
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let content = if data.variants.is_empty() {
        // Nothing to match on; fall back to the Debug representation.
        quote! { format!("{self:?}") }
    } else {
        let arms = data
            .variants
            .iter()
            .map(|variant| {
                let var_name = &variant.ident;
                let fmt = get_message_format_from_attrs(&variant.attrs)?;
                let (pattern, format_expr) = create_field_bindings(&variant.fields, &fmt)?;
                Ok(quote! {
                    #[allow(unused_variables)]
                    Self::#var_name #pattern => #format_expr,
                })
            })
            .collect::<syn::Result<Vec<_>>>()?;

        quote! { match self { #(#arms)* } }
    };

    Ok(quote! {
        impl #impl_generics Event for #name #ty_generics #where_clause {
            fn debug(&self) -> String {
                #content
            }
        }
    })
}

pub fn derive_event_struct(input: &DeriveInput, data: &DataStruct) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fmt = get_message_format_from_attrs(&input.attrs)?;
    let (pattern, format_expr) = create_field_bindings(&data.fields, &fmt)?;

    // Unit structs need no destructuring; everything else gets a `let Self …`.
    let content = if let Fields::Unit = data.fields {
        format_expr
    } else {
        quote! {
            #[allow(unused_variables)]
            let Self #pattern = self;
            #format_expr
        }
    };

    Ok(quote! {
        impl #impl_generics Event for #name #ty_generics #where_clause {
            fn debug(&self) -> String {
                #content
            }
        }
    })
}

/// This will create the field destructuring pattern & the expression for the
/// formatting of the field
/// Returns (pattern, format expression)
fn create_field_bindings(
    fields: &Fields,
    message_format: &str,
) -> syn::Result<(TokenStream, TokenStream)> {
    match fields {
        // struct Foo { x: T, y: T }  →  let Self { x, y, .. } = self;
        Fields::Named(named) => {
            let idents: Vec<_> = named.named.iter().map(|f| &f.ident).collect();
            // `..` lets the pattern survive the addition of new fields.
            let pattern = quote! { { #(#idents,)* .. } };
            let format_expr = quote! { format!(#message_format) };

            Ok((pattern, format_expr))
        }
        // struct Foo(T, T)  →  let Self(_0, _1) = self;
        Fields::Unnamed(unnamed) => {
            let (fmt, idents) = rewrite_unnamed_placeholders(message_format, unnamed)?;
            // No `..`: we enumerate every field, so the pattern is already
            // exhaustive. Adding `..` would cause an "unnecessary `..`" lint.
            let pattern = quote! { ( #(#idents,)* ) };
            let format_expr = quote! { format!(#fmt) };

            Ok((pattern, format_expr))
        }
        // struct Foo;  →  nothing to destructure.
        Fields::Unit => Ok((quote! {}, quote! { format!(#message_format) })),
    }
}

/// Extracts the format string from the first `#[event("…")]` attribute found.
///
/// Returns `"{self:?}"` when no `#[event]` attribute is present.
/// Emits a `syn::Error` with a proper source span if the attribute argument
/// is not a string literal.
fn get_message_format_from_attrs(attrs: &[Attribute]) -> syn::Result<String> {
    for attr in attrs {
        if attr.path().is_ident("event") {
            return Ok(attr
                .parse_args::<LitStr>()
                .map_err(|_| {
                    syn::Error::new_spanned(attr, "expected a string literal: #[event(\"…\")]")
                })?
                .value());
        }
    }
    Ok(String::from("{self:?}"))
}

/// Rewrites active tuple-field placeholders while preserving escaped braces and
/// formatting specifications. Dynamic positional widths are intentionally rejected.
fn rewrite_unnamed_placeholders(
    format: &str,
    fields: &FieldsUnnamed,
) -> syn::Result<(String, Vec<Ident>)> {
    let idents: Vec<Ident> = fields
        .unnamed
        .iter()
        .enumerate()
        .map(|(i, _)| quote::format_ident!("_{i}"))
        .collect();
    let mut output = String::with_capacity(format.len());
    let bytes = format.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if bytes[cursor] == b'{' {
            if bytes.get(cursor + 1) == Some(&b'{') {
                output.push_str("{{");
                cursor += 2;
                continue;
            }
            let start = cursor;
            cursor += 1;
            let index_start = cursor;
            while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
                cursor += 1;
            }
            if cursor > index_start && matches!(bytes.get(cursor), Some(b':' | b'}')) {
                let index = format[index_start..cursor].parse::<usize>().map_err(|_| {
                    syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "tuple field index is too large",
                    )
                })?;
                if index >= fields.unnamed.len() {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        format!("tuple event has no field {index}"),
                    ));
                }
                let end = format[cursor..]
                    .find('}')
                    .map(|offset| cursor + offset)
                    .ok_or_else(|| {
                        syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "unclosed event format placeholder",
                        )
                    })?;
                let spec = &format[cursor..end];
                if spec.contains('{')
                    || spec
                        .as_bytes()
                        .windows(2)
                        .any(|pair| pair[0].is_ascii_digit() && pair[1] == b'$')
                {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "dynamic positional format widths are not supported in tuple events",
                    ));
                }
                write!(&mut output, "{{_{index}{spec}}}").expect("writing to String cannot fail");
                cursor = end + 1;
            } else {
                // Named captures such as `{self:?}` are handled by `format!`.
                output.push_str(&format[start..cursor]);
            }
        } else if bytes[cursor] == b'}' && bytes.get(cursor + 1) == Some(&b'}') {
            output.push_str("}}");
            cursor += 2;
        } else {
            let ch = format[cursor..].chars().next().expect("valid UTF-8");
            output.push(ch);
            cursor += ch.len_utf8();
        }
    }
    Ok((output, idents))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{DeriveInput, parse_str};

    // ── rewrite_unnamed_placeholders ──────────────────────────────────────────

    fn unnamed_fields_from(src: &str) -> FieldsUnnamed {
        let input: DeriveInput = parse_str(src).expect("test input should parse");
        match input.data {
            syn::Data::Struct(s) => match s.fields {
                Fields::Unnamed(u) => u,
                _ => panic!("expected unnamed fields"),
            },
            _ => panic!("expected a struct"),
        }
    }

    #[test]
    fn rewrite_replaces_all_positional_placeholders() {
        let fields = unnamed_fields_from("struct Foo(u32, String);");
        let (fmt, idents) = rewrite_unnamed_placeholders("{0} and {1}", &fields).unwrap();
        assert_eq!(fmt, "{_0} and {_1}");
        assert_eq!(idents.len(), 2);
        assert_eq!(idents[0], quote::format_ident!("_0"));
        assert_eq!(idents[1], quote::format_ident!("_1"));
    }

    #[test]
    fn rewrite_leaves_non_positional_format_intact() {
        let fields = unnamed_fields_from("struct Foo(u32);");
        let (fmt, idents) = rewrite_unnamed_placeholders("no placeholders", &fields).unwrap();
        assert_eq!(fmt, "no placeholders");
        assert_eq!(idents.len(), 1);
    }

    #[test]
    fn rewrite_handles_partial_placeholder_use() {
        let fields = unnamed_fields_from("struct Foo(u32, String, bool);");
        let (fmt, _) = rewrite_unnamed_placeholders("only {1} matters", &fields).unwrap();
        assert_eq!(fmt, "only {_1} matters");
    }

    #[test]
    fn rewrite_handles_repeated_placeholder() {
        let fields = unnamed_fields_from("struct Foo(u32, String);");
        let (fmt, _) = rewrite_unnamed_placeholders("{0} then {0} again", &fields).unwrap();
        assert_eq!(fmt, "{_0} then {_0} again");
    }

    #[test]
    fn rewrite_preserves_escaped_braces_and_format_specs() {
        let fields = unnamed_fields_from("struct Foo(f32);");
        let (fmt, _) =
            rewrite_unnamed_placeholders("literal {{0}}, value {0:.2?}", &fields).unwrap();
        assert_eq!(fmt, "literal {{0}}, value {_0:.2?}");
    }

    #[test]
    fn rewrite_rejects_invalid_field_indices() {
        let fields = unnamed_fields_from("struct Foo(u32);");
        assert!(rewrite_unnamed_placeholders("{1}", &fields).is_err());
    }

    // ── get_message_format_from_attrs ─────────────────────────────────────────

    #[test]
    fn format_falls_back_to_debug_repr_when_no_attr() {
        let input: DeriveInput = parse_str("struct Foo;").expect("test input should parse");
        let result =
            get_message_format_from_attrs(&input.attrs).expect("default format should parse");
        assert_eq!(result, "{self:?}");
    }

    #[test]
    fn format_extracts_literal_from_event_attr() {
        let input: DeriveInput =
            parse_str(r#"#[event("hello world")] struct Foo;"#).expect("test input should parse");
        let result =
            get_message_format_from_attrs(&input.attrs).expect("event format should parse");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn format_returns_error_for_non_string_arg() {
        let input: DeriveInput =
            parse_str(r"#[event(42)] struct Foo;").expect("test input should parse");
        assert!(get_message_format_from_attrs(&input.attrs).is_err());
    }

    #[test]
    fn format_ignores_unrelated_attributes() {
        let input: DeriveInput =
            parse_str(r"#[derive(Debug)] struct Foo;").expect("test input should parse");
        let result =
            get_message_format_from_attrs(&input.attrs).expect("default format should parse");
        assert_eq!(result, "{self:?}");
    }
}
