use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Attribute, DataEnum, DataStruct, DeriveInput, Fields, Ident, LitStr};

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
    message_format: &LitStr,
) -> syn::Result<(TokenStream, TokenStream)> {
    match fields {
        // struct Foo { x: T, y: T }  →  let Self { x, y, .. } = self;
        Fields::Named(named) => {
            let fmt = rewrite_placeholders(message_format, None)?;
            let idents: Vec<_> = named.named.iter().map(|f| &f.ident).collect();
            // `..` lets the pattern survive the addition of new fields.
            let pattern = quote! { { #(#idents,)* .. } };
            let format_expr = quote! { format!(#fmt) };

            Ok((pattern, format_expr))
        }
        // struct Foo(T, T)  →  let Self(_0, _1) = self;
        Fields::Unnamed(unnamed) => {
            let fmt = rewrite_placeholders(message_format, Some(unnamed.unnamed.len()))?;
            let idents: Vec<Ident> = (0..unnamed.unnamed.len())
                .map(|i| quote::format_ident!("_{i}"))
                .collect();
            // No `..`: we enumerate every field, so the pattern is already
            // exhaustive. Adding `..` would cause an "unnecessary `..`" lint.
            let pattern = quote! { ( #(#idents,)* ) };
            let format_expr = quote! { format!(#fmt) };

            Ok((pattern, format_expr))
        }
        // struct Foo;  →  nothing to destructure.
        Fields::Unit => {
            let fmt = rewrite_placeholders(message_format, None)?;
            Ok((quote! {}, quote! { format!(#fmt) }))
        }
    }
}

/// Extracts the format string from the first `#[event("…")]` attribute found.
///
/// Returns `"{self:?}"` when no `#[event]` attribute is present.
/// Emits a `syn::Error` with a proper source span if the attribute argument
/// is not a string literal.
fn get_message_format_from_attrs(attrs: &[Attribute]) -> syn::Result<LitStr> {
    for attr in attrs {
        if attr.path().is_ident("event") {
            return attr.parse_args::<LitStr>().map_err(|_| {
                syn::Error::new_spanned(attr, "expected a string literal: #[event(\"…\")]")
            });
        }
    }
    Ok(LitStr::new("{self:?}", Span::call_site()))
}

/// Validates an event format string and rewrites tuple-field placeholders.
///
/// Event formats are passed to `format!` without positional arguments, so
/// every placeholder must name what it prints: a named capture such as `{x}`
/// or `{self:?}`, or, for tuple events (`tuple_fields` is `Some(len)`), a
/// field index such as `{0}`, which is rewritten to the `_0` binding. Implicit
/// positional placeholders (`{}`, `{:?}`), `.*` precision and `N$`
/// width/precision arguments would all require positional arguments and are
/// rejected here with an error spanning the format string literal.
///
/// Returns the rewritten format string literal, keeping the original span so
/// any remaining `format!` errors point at the attribute.
fn rewrite_placeholders(format: &LitStr, tuple_fields: Option<usize>) -> syn::Result<LitStr> {
    let value = format.value();
    let error = |message: String| syn::Error::new(format.span(), message);
    let mut output = String::with_capacity(value.len());
    let mut rest = value.as_str();

    while let Some(position) = rest.find(['{', '}']) {
        output.push_str(&rest[..position]);
        let tail = &rest[position..];

        if let Some(after) = tail.strip_prefix("{{") {
            output.push_str("{{");
            rest = after;
            continue;
        }
        if let Some(after) = tail.strip_prefix("}}") {
            output.push_str("}}");
            rest = after;
            continue;
        }
        if tail.starts_with('}') {
            return Err(error(
                "unmatched `}` in event format; use `}}` for a literal brace".to_owned(),
            ));
        }

        // A brace immediately before alignment is a fill character, not a
        // delimiter. Skip it before looking for the placeholder's end.
        let end = tail.find([':', '}']).and_then(|index| {
            if tail.as_bytes()[index] == b'}' {
                Some(index)
            } else {
                let spec = skip_fill_and_alignment(&tail[index + 1..]);
                spec.find('}').map(|end| tail.len() - spec.len() + end)
            }
        });
        let Some(end) = end else {
            return Err(error(
                "unclosed `{` in event format; use `{{` for a literal brace".to_owned(),
            ));
        };
        let placeholder = &tail[1..end];
        rest = &tail[end + 1..];

        let (argument, spec) = match placeholder.split_once(':') {
            Some((argument, spec)) => (argument, Some(spec)),
            None => (placeholder, None),
        };
        if let Some(spec) = spec {
            validate_spec(spec)
                .map_err(|message| error(format!("`{{{placeholder}}}`: {message}")))?;
        }

        output.push('{');
        if argument.is_empty() {
            return Err(error(format!(
                "`{{{placeholder}}}`: implicit positional placeholders are not supported in \
                 event formats; reference a field by name (`{{field}}`) or, in tuple events, \
                 by index (`{{0}}`)"
            )));
        } else if argument.bytes().all(|byte| byte.is_ascii_digit()) {
            let Some(len) = tuple_fields else {
                return Err(error(format!(
                    "`{{{placeholder}}}`: positional placeholders are only supported in tuple \
                     events; reference named fields by name"
                )));
            };
            let index = argument
                .parse::<usize>()
                .ok()
                .filter(|index| *index < len)
                .ok_or_else(|| {
                    error(format!(
                        "`{{{placeholder}}}`: tuple event has no field {argument} (it has {len})"
                    ))
                })?;
            output.push('_');
            output.push_str(&index.to_string());
        } else {
            // Named captures such as `{self:?}` are resolved by `format!`.
            output.push_str(argument);
        }
        if let Some(spec) = spec {
            output.push(':');
            output.push_str(spec);
        }
        output.push('}');
    }
    output.push_str(rest);

    Ok(LitStr::new(&output, format.span()))
}

/// Skips the optional fill character and alignment, preserving UTF-8 boundaries.
fn skip_fill_and_alignment(spec: &str) -> &str {
    let mut chars = spec.chars();
    let first = chars.next();
    if matches!(chars.next(), Some('<' | '>' | '^')) {
        chars.as_str()
    } else if matches!(first, Some('<' | '>' | '^')) {
        &spec[1..]
    } else {
        spec
    }
}

/// Rejects format specs that need positional arguments.
fn validate_spec(spec: &str) -> Result<(), &'static str> {
    let spec = skip_fill_and_alignment(spec);
    if spec.contains('{') {
        return Err("nested placeholders are not supported in event formats");
    }
    if spec.contains(".*") {
        return Err(
            "`.*` precision needs a positional argument; use a literal precision \
             such as `.2` or a named capture such as `.prec$`",
        );
    }
    // Split on format syntax, not identifier characters: capture names can
    // contain Unicode. `name$` is a capture; `N$` is a positional argument.
    let positional_argument = spec.match_indices('$').any(|(dollar, _)| {
        let name = spec[..dollar]
            .rsplit(['+', '-', '#', '.', '$'])
            .next()
            .unwrap_or_default();
        !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_digit())
    });
    if positional_argument {
        return Err(
            "positional `N$` width/precision arguments are not supported in event \
             formats; use a literal value or a named capture such as `width$`",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{DeriveInput, parse_str};

    // ── rewrite_placeholders ──────────────────────────────────────────────────

    fn lit(format: &str) -> LitStr {
        LitStr::new(format, Span::call_site())
    }

    fn rewrite_tuple(format: &str, fields: usize) -> syn::Result<String> {
        rewrite_placeholders(&lit(format), Some(fields)).map(|lit| lit.value())
    }

    fn rewrite_named(format: &str) -> syn::Result<String> {
        rewrite_placeholders(&lit(format), None).map(|lit| lit.value())
    }

    fn tuple_error(format: &str, fields: usize) -> String {
        match rewrite_tuple(format, fields) {
            Ok(rewritten) => panic!("`{format}` should be rejected, got `{rewritten}`"),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn rewrite_replaces_all_positional_placeholders() -> syn::Result<()> {
        assert_eq!(rewrite_tuple("{0} and {1}", 2)?, "{_0} and {_1}");
        Ok(())
    }

    #[test]
    fn rewrite_leaves_non_positional_format_intact() -> syn::Result<()> {
        assert_eq!(rewrite_tuple("no placeholders", 1)?, "no placeholders");
        assert_eq!(rewrite_tuple("{self:?}", 1)?, "{self:?}");
        Ok(())
    }

    #[test]
    fn rewrite_handles_partial_placeholder_use() -> syn::Result<()> {
        assert_eq!(rewrite_tuple("only {1} matters", 3)?, "only {_1} matters");
        Ok(())
    }

    #[test]
    fn rewrite_handles_repeated_placeholder() -> syn::Result<()> {
        assert_eq!(
            rewrite_tuple("{0} then {0} again", 2)?,
            "{_0} then {_0} again"
        );
        Ok(())
    }

    #[test]
    fn rewrite_preserves_escaped_braces_and_format_specs() -> syn::Result<()> {
        assert_eq!(
            rewrite_tuple("literal {{0}}, value {0:.2?}", 1)?,
            "literal {{0}}, value {_0:.2?}"
        );
        assert_eq!(rewrite_tuple("ü {0:>8} ✓", 1)?, "ü {_0:>8} ✓");
        Ok(())
    }

    #[test]
    fn rewrite_accepts_named_width_and_precision_captures() -> syn::Result<()> {
        assert_eq!(rewrite_named("{x:>width$.prec2$}")?, "{x:>width$.prec2$}");
        Ok(())
    }

    #[test]
    fn rewrite_rejects_invalid_field_indices() {
        assert!(tuple_error("{1}", 1).contains("tuple event has no field 1"));
        assert!(tuple_error("{99999999999999999999999}", 1).contains("has no field"));
    }

    #[test]
    fn rewrite_rejects_implicit_positional_placeholders() {
        for format in ["{}", "value: {:?}", "{:>4}"] {
            assert!(
                tuple_error(format, 1).contains("implicit positional placeholders"),
                "{format}"
            );
            assert!(rewrite_named(format).is_err(), "{format}");
        }
    }

    #[test]
    fn rewrite_rejects_positional_precision_and_width_arguments() {
        assert!(tuple_error("{0:.*}", 2).contains("`.*` precision"));
        for format in ["{0:1$}", "{0:.1$}", "{0:}>1$}", "{0:🦀^.1$}"] {
            assert!(tuple_error(format, 2).contains("`N$`"), "{format}");
        }
        assert!(tuple_error("{0:{}}", 2).contains("nested placeholders"));
    }

    #[test]
    fn rewrite_rejects_positional_placeholders_outside_tuple_events() {
        let err = rewrite_named("{0}").expect_err("named events have no positional fields");
        assert!(err.to_string().contains("only supported in tuple events"));
    }

    #[test]
    fn rewrite_rejects_unbalanced_braces() {
        assert!(tuple_error("open {0", 1).contains("unclosed `{`"));
        assert!(tuple_error("open {0:}>5", 1).contains("unclosed `{`"));
        assert!(tuple_error("close }", 1).contains("unmatched `}`"));
    }

    // ── get_message_format_from_attrs ─────────────────────────────────────────

    #[test]
    fn format_falls_back_to_debug_repr_when_no_attr() {
        let input: DeriveInput = parse_str("struct Foo;").expect("test input should parse");
        let result =
            get_message_format_from_attrs(&input.attrs).expect("default format should parse");
        assert_eq!(result.value(), "{self:?}");
    }

    #[test]
    fn format_extracts_literal_from_event_attr() {
        let input: DeriveInput =
            parse_str(r#"#[event("hello world")] struct Foo;"#).expect("test input should parse");
        let result =
            get_message_format_from_attrs(&input.attrs).expect("event format should parse");
        assert_eq!(result.value(), "hello world");
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
        assert_eq!(result.value(), "{self:?}");
    }

    #[test]
    fn derive_reports_format_errors_as_compile_errors() {
        let input: DeriveInput = parse_str(r#"#[event("value: {}")] struct Foo(u32);"#)
            .expect("test input should parse");
        let syn::Data::Struct(data) = &input.data else {
            panic!("expected a struct");
        };
        let err = derive_event_struct(&input, data).expect_err("`{}` must be rejected");
        assert!(err.to_string().contains("implicit positional placeholders"));
    }
}
