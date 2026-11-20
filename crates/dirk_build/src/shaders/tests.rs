use dirk_shaders::bindless::heap;
use rspirv::{
    dr::{Builder, Operand},
    spirv::{Decoration, Dim, FunctionControl, ImageFormat, StorageClass},
};

use super::{Types, type_name};

#[test]
fn entry_points_become_pascal_case_types() {
    assert_eq!(type_name("fill_vs"), "FillVs");
    assert_eq!(type_name("mesh::shadow_fs"), "MeshShadowFs");
}

/// Checks a module declaring an array of single-sampled 2D images, of
/// `depth` and `sampled` as `OpTypeImage` encodes them, at `binding`.
fn check(binding: u32, depth: u32, sampled: u32) -> anyhow::Result<()> {
    let mut builder = Builder::new();
    let float = builder.type_float(32, None);
    let format = ImageFormat::Unknown;
    let image = builder.type_image(float, Dim::Dim2D, depth, 0, 0, sampled, format, None);
    let array = builder.type_runtime_array(image);
    let pointer = builder.type_pointer(None, StorageClass::UniformConstant, array);
    let variable = builder.variable(pointer, None, StorageClass::UniformConstant, None);
    builder.decorate(
        variable,
        Decoration::DescriptorSet,
        [Operand::LiteralBit32(0)],
    );
    builder.decorate(
        variable,
        Decoration::Binding,
        [Operand::LiteralBit32(binding)],
    );
    Types(&builder.module()).check_bindings()
}

#[test]
fn heap_arrays_are_declared_with_the_elements_they_hold() {
    // Depth 2 is unknown, as `Image!` declares by default; sampled 2 is
    // a storage image.
    check(heap::TEXTURES_2D, 2, 1).expect("2D textures");
    check(heap::STORAGE_IMAGES_2D, 2, 2).expect("2D storage images");
    assert!(check(heap::TEXTURES_2D_ARRAY, 2, 1).is_err(), "not arrayed");
    assert!(check(heap::TEXTURES_2D, 1, 1).is_err(), "depth");
    assert!(check(heap::TEXTURES_2D, 2, 2).is_err(), "storage");
    assert!(check(heap::SAMPLERS, 2, 1).is_err(), "not samplers");
    assert!(check(64, 2, 1).is_err(), "no array there");
}

/// Whether a module storing a `u32` to a variable of `class`, or only
/// loading it, writes storage.
fn writes(class: StorageClass, store: bool) -> anyhow::Result<bool> {
    let mut builder = Builder::new();
    let (void, uint) = (builder.type_void(), builder.type_int(32, 0));
    let pointer = builder.type_pointer(None, class, uint);
    let variable = builder.variable(pointer, None, class, None);
    let function = builder.type_function(void, []);
    builder.begin_function(void, None, FunctionControl::NONE, function)?;
    builder.begin_block(None)?;
    if store {
        let zero = builder.constant_bit32(uint, 0);
        builder.store(variable, zero, None, [])?;
    } else {
        builder.load(uint, None, variable, None, [])?;
    }
    builder.ret()?;
    builder.end_function()?;
    Types(&builder.module()).writes_storage()
}

#[test]
fn only_stores_to_storage_write_it() -> anyhow::Result<()> {
    assert!(writes(StorageClass::StorageBuffer, true)?, "storage stores");
    assert!(
        !writes(StorageClass::StorageBuffer, false)?,
        "storage loads"
    );
    assert!(!writes(StorageClass::Private, true)?, "private stores");
    Ok(())
}
