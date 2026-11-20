use std::path::PathBuf;

use super::Module;

/// The push data type `function` reads in a module of crate `shaders`
/// at `path`, holding `code`.
fn push_type(path: &[&str], code: &str, function: &str) -> String {
    let module = Module {
        path: path.iter().map(ToString::to_string).collect(),
        items: syn::parse_file(code).expect("valid code").items,
        dir: PathBuf::new(),
    };
    module.push_type("shaders", function).expect("resolvable")
}

#[test]
fn push_types_are_named_as_their_module_reaches_them() {
    let code = r"
        use dirk_shaders::types::{MeshPush, PresentPush as Present};
        use crate::common;

        pub struct Local { value: u32 }

        #[spirv(vertex)]
        pub fn imported(#[spirv(push_constant)] push: &MeshPush) {}
        #[spirv(fragment)]
        pub fn renamed(uv: Vec2, #[spirv(push_constant)] push: &Present) {}
        #[spirv(fragment)]
        pub fn local(#[spirv(push_constant)] push: &Local) {}
        #[spirv(fragment)]
        pub fn through_module(#[spirv(push_constant)] push: &common::Shared) {}
        #[spirv(fragment)]
        pub fn parent(#[spirv(push_constant)] push: &super::Outer) {}
        #[spirv(compute(threads(64)))]
        pub fn array(#[spirv(push_constant)] push: &[u32; 4]) {}
        #[spirv(vertex)]
        pub fn none(#[spirv(vertex_index)] index: i32) {}
    ";
    let resolve = |function| push_type(&["meshes"], code, function);
    assert_eq!(resolve("imported"), "::dirk_shaders::types::MeshPush");
    assert_eq!(resolve("renamed"), "::dirk_shaders::types::PresentPush");
    assert_eq!(resolve("local"), "::shaders::meshes::Local");
    assert_eq!(resolve("through_module"), "::shaders::common::Shared");
    assert_eq!(resolve("parent"), "::shaders::Outer");
    assert_eq!(resolve("array"), "[u32; 4]");
    assert_eq!(resolve("none"), "()");
}
