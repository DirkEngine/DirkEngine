use super::*;
use syn::{DeriveInput, parse_str};

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

#[test]
fn format_falls_back_to_debug_repr_when_no_attr() {
    let input: DeriveInput = parse_str("struct Foo;").expect("test input should parse");
    let result = get_message_format_from_attrs(&input.attrs).expect("default format should parse");
    assert_eq!(result.value(), "{self:?}");
}

#[test]
fn format_extracts_literal_from_event_attr() {
    let input: DeriveInput =
        parse_str(r#"#[event("hello world")] struct Foo;"#).expect("test input should parse");
    let result = get_message_format_from_attrs(&input.attrs).expect("event format should parse");
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
    let result = get_message_format_from_attrs(&input.attrs).expect("default format should parse");
    assert_eq!(result.value(), "{self:?}");
}

#[test]
fn derive_reports_format_errors_as_compile_errors() {
    let input: DeriveInput =
        parse_str(r#"#[event("value: {}")] struct Foo(u32);"#).expect("test input should parse");
    let syn::Data::Struct(data) = &input.data else {
        panic!("expected a struct");
    };
    let err = derive_event_struct(&input, data).expect_err("`{}` must be rejected");
    assert!(err.to_string().contains("implicit positional placeholders"));
}
