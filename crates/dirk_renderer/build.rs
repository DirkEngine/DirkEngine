//! Builds platform configuration and generated shader metadata for `dirk_renderer`.

use anyhow::{Context, anyhow, bail};
use cargo_gpu_install::{
    install::Install,
    spirv_builder::{ModuleResult, SpirvMetadata},
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use rspirv_reflect::rspirv::{
    binary::Parser,
    dr::{Instruction, Loader, Module as SpirvModule, Operand},
    spirv::{Decoration, ExecutionModel, Op, StorageClass},
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

fn main() -> anyhow::Result<()> {
    dirk_build::configure_platform();

    println!("cargo:rustc-check-cfg=cfg(validation)");

    let profile = std::env::var("PROFILE").unwrap_or_default();
    if profile != "release" {
        println!("cargo:rustc-cfg=validation");
    }

    build_shaders()?;

    Ok(())
}

fn build_shaders() -> anyhow::Result<()> {
    let shader_crate = PathBuf::from("shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    println!("cargo:rerun-if-changed=shaders/Cargo.toml");
    println!("cargo:rerun-if-changed=shaders/src/lib.rs");

    let backend = Install::from_shader_crate(shader_crate.clone()).run()?;
    let mut builder = backend.to_spirv_builder(shader_crate, "spirv-unknown-vulkan1.3");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(false);
    builder.multimodule = true;
    builder.spirv_metadata = SpirvMetadata::None;

    let spv_result = builder.build()?;
    let ModuleResult::MultiModule(modules) = spv_result.module else {
        return Err(anyhow!("expected one SPIR-V module per shader entry point"));
    };
    let mut shaders = Vec::new();
    for (entrypoint, source_path) in modules {
        let output_path = out_dir.join(format!("{entrypoint}.spv"));
        fs::copy(&source_path, &output_path)?;
        shaders.push(reflect_shader(&entrypoint, &source_path)?);
    }
    shaders.sort_by(|left, right| left.entrypoint.cmp(&right.entrypoint));
    fs::write(
        out_dir.join("generated_shaders.rs"),
        generate_shader_module(&shaders).to_string(),
    )?;

    Ok(())
}

#[derive(Debug)]
struct ReflectedShader {
    entrypoint: String,
    type_name: String,
    stage: ShaderStage,
    set_layouts: Vec<Vec<DescriptorBinding>>,
    vertex_inputs: Vec<VertexInput>,
}

#[derive(Clone, Debug)]
struct DescriptorBinding {
    binding: u32,
    descriptor_type: &'static str,
    descriptor_count: u32,
    stage_flags: &'static str,
}

#[derive(Debug)]
struct VertexInput {
    location: u32,
    format: &'static str,
    size: u32,
    offset: u32,
}

struct ReflectedDescriptor {
    descriptor_type: &'static str,
    descriptor_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug)]
enum ScalarKind {
    Sint,
    Uint,
    Float,
}

#[derive(Clone, Copy, Debug)]
struct Scalar {
    kind: ScalarKind,
    width: u8,
}

#[derive(Clone, Copy, Debug)]
enum VectorSize {
    Bi,
    Tri,
    Quad,
}

impl From<VectorSize> for u32 {
    fn from(size: VectorSize) -> Self {
        match size {
            VectorSize::Bi => 2,
            VectorSize::Tri => 3,
            VectorSize::Quad => 4,
        }
    }
}

fn reflect_shader(entrypoint: &str, spv_path: &Path) -> anyhow::Result<ReflectedShader> {
    let bytes = fs::read(spv_path)
        .with_context(|| format!("failed to read SPIR-V for shader entry point `{entrypoint}`"))?;
    reflect_shader_spirv(entrypoint, &bytes)
}

fn reflect_shader_spirv(entrypoint: &str, bytes: &[u8]) -> anyhow::Result<ReflectedShader> {
    let module = parse_spirv(bytes)
        .with_context(|| format!("failed to parse SPIR-V fallback for shader `{entrypoint}`"))?;
    let raw = RawReflection::new(&module);
    let (stage, function_id, interface_ids) = raw.entry_point(entrypoint)?;
    let used_ids = raw.used_ids(function_id);
    let set_layouts = raw.descriptor_sets(entrypoint, stage, &used_ids)?;
    let vertex_inputs = if stage == ShaderStage::Vertex {
        raw.vertex_inputs(entrypoint, &interface_ids)?
    } else {
        Vec::new()
    };

    Ok(ReflectedShader {
        entrypoint: entrypoint.to_owned(),
        type_name: shader_type_name(entrypoint, stage)?,
        stage,
        set_layouts,
        vertex_inputs,
    })
}

fn parse_spirv(bytes: &[u8]) -> anyhow::Result<SpirvModule> {
    let mut loader = Loader::new();
    Parser::new(bytes, &mut loader).parse()?;
    Ok(loader.module())
}

fn build_vertex_inputs(
    entrypoint: &str,
    mut reflected: Vec<(u32, &'static str, u32)>,
) -> anyhow::Result<Vec<VertexInput>> {
    reflected.sort_by_key(|(location, _, _)| *location);

    let mut offset = 0_u32;
    let mut inputs = Vec::new();
    for (location, format, size) in reflected {
        inputs.push(VertexInput {
            location,
            format,
            size,
            offset,
        });
        offset = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("shader `{entrypoint}` vertex input stride overflowed u32"))?;
    }
    Ok(inputs)
}

fn scalar_format(
    entrypoint: &str,
    location: u32,
    size: Option<VectorSize>,
    scalar: Scalar,
) -> anyhow::Result<(&'static str, u32)> {
    let components = size.map_or(1, u32::from);
    let format = match (scalar.kind, scalar.width, components) {
        (ScalarKind::Float, 4, 1) => "R32_SFLOAT",
        (ScalarKind::Float, 4, 2) => "R32G32_SFLOAT",
        (ScalarKind::Float, 4, 3) => "R32G32B32_SFLOAT",
        (ScalarKind::Float, 4, 4) => "R32G32B32A32_SFLOAT",
        (ScalarKind::Sint, 4, 1) => "R32_SINT",
        (ScalarKind::Sint, 4, 2) => "R32G32_SINT",
        (ScalarKind::Sint, 4, 3) => "R32G32B32_SINT",
        (ScalarKind::Sint, 4, 4) => "R32G32B32A32_SINT",
        (ScalarKind::Uint, 4, 1) => "R32_UINT",
        (ScalarKind::Uint, 4, 2) => "R32G32_UINT",
        (ScalarKind::Uint, 4, 3) => "R32G32B32_UINT",
        (ScalarKind::Uint, 4, 4) => "R32G32B32A32_UINT",
        _ => bail!(
            "shader `{entrypoint}` location {location} uses unsupported vertex scalar {:?} with width {} and {components} component(s)",
            scalar.kind,
            scalar.width
        ),
    };
    Ok((format, u32::from(scalar.width) * components))
}

#[derive(Default)]
struct RawDecorations {
    location: Option<u32>,
    descriptor_set: Option<u32>,
    binding: Option<u32>,
    built_in: bool,
}

struct RawReflection<'a> {
    module: &'a SpirvModule,
    decorations: HashMap<u32, RawDecorations>,
    assignments: HashMap<u32, &'a Instruction>,
}

const ENTRY_POINT_STAGE_OPERAND: usize = 0;
const ENTRY_POINT_FUNCTION_OPERAND: usize = 1;
const ENTRY_POINT_NAME_OPERAND: usize = 2;
const ENTRY_POINT_INTERFACE_OPERAND_START: usize = 3;
const TYPE_IMAGE_SAMPLED_OPERAND: usize = 5;

impl<'a> RawReflection<'a> {
    fn new(module: &'a SpirvModule) -> Self {
        let mut decorations = HashMap::<u32, RawDecorations>::new();
        for instruction in &module.annotations {
            if instruction.class.opcode != Op::Decorate {
                continue;
            }
            let Some(target) = id_ref(instruction, 0) else {
                continue;
            };
            let entry = decorations.entry(target).or_default();
            match decoration(instruction, 1) {
                Some(Decoration::Location) => entry.location = literal_u32(instruction, 2),
                Some(Decoration::DescriptorSet) => {
                    entry.descriptor_set = literal_u32(instruction, 2);
                }
                Some(Decoration::Binding) => entry.binding = literal_u32(instruction, 2),
                Some(Decoration::BuiltIn) => entry.built_in = true,
                _ => {}
            }
        }

        let assignments = module
            .types_global_values
            .iter()
            .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
            .collect();

        Self {
            module,
            decorations,
            assignments,
        }
    }

    fn entry_point(&self, entrypoint: &str) -> anyhow::Result<(ShaderStage, u32, Vec<u32>)> {
        for instruction in &self.module.entry_points {
            let Some(name) = literal_string(instruction, ENTRY_POINT_NAME_OPERAND) else {
                continue;
            };
            if name != entrypoint {
                continue;
            }
            let model = execution_model(instruction, ENTRY_POINT_STAGE_OPERAND)
                .ok_or_else(|| anyhow!("shader `{entrypoint}` entry point is missing a stage"))?;
            let stage = match model {
                ExecutionModel::Vertex => ShaderStage::Vertex,
                ExecutionModel::Fragment => ShaderStage::Fragment,
                ExecutionModel::GLCompute => ShaderStage::Compute,
                _ => bail!("shader `{entrypoint}` uses unsupported execution model {model:?}"),
            };
            let function_id =
                id_ref(instruction, ENTRY_POINT_FUNCTION_OPERAND).ok_or_else(|| {
                    anyhow!("shader `{entrypoint}` entry point is missing a function id")
                })?;
            let interface_ids = instruction
                .operands
                .iter()
                .skip(ENTRY_POINT_INTERFACE_OPERAND_START)
                .filter_map(|operand| match operand {
                    Operand::IdRef(id) => Some(*id),
                    _ => None,
                })
                .collect();
            return Ok((stage, function_id, interface_ids));
        }
        bail!("SPIR-V module did not contain entry point `{entrypoint}`")
    }

    fn used_ids(&self, function_id: u32) -> HashSet<u32> {
        let mut used_ids = HashSet::new();
        let mut visited_functions = HashSet::new();
        self.collect_function_used_ids(function_id, &mut used_ids, &mut visited_functions);
        used_ids
    }

    fn collect_function_used_ids(
        &self,
        function_id: u32,
        used_ids: &mut HashSet<u32>,
        visited_functions: &mut HashSet<u32>,
    ) {
        if !visited_functions.insert(function_id) {
            return;
        }
        let Some(function) = self.module.functions.iter().find(|function| {
            function.def.as_ref().and_then(|def| def.result_id) == Some(function_id)
        }) else {
            return;
        };

        for block in &function.blocks {
            for instruction in &block.instructions {
                for operand in &instruction.operands {
                    if let Operand::IdRef(id) = operand {
                        used_ids.insert(*id);
                        if self.is_function_id(*id) {
                            self.collect_function_used_ids(*id, used_ids, visited_functions);
                        }
                    }
                }
            }
        }
    }

    fn is_function_id(&self, id: u32) -> bool {
        self.module
            .functions
            .iter()
            .any(|function| function.def.as_ref().and_then(|def| def.result_id) == Some(id))
    }

    fn descriptor_sets(
        &self,
        entrypoint: &str,
        stage: ShaderStage,
        used_ids: &HashSet<u32>,
    ) -> anyhow::Result<Vec<Vec<DescriptorBinding>>> {
        let mut sets = BTreeMap::<u32, Vec<DescriptorBinding>>::new();
        for instruction in &self.module.types_global_values {
            if instruction.class.opcode != Op::Variable {
                continue;
            }
            let Some(result_id) = instruction.result_id else {
                continue;
            };
            if !used_ids.contains(&result_id) {
                continue;
            }
            let Some(decorations) = self.decorations.get(&result_id) else {
                continue;
            };
            let (Some(set), Some(binding)) = (decorations.descriptor_set, decorations.binding)
            else {
                continue;
            };
            let descriptor = self.descriptor(entrypoint, instruction)?;
            sets.entry(set).or_default().push(DescriptorBinding {
                binding,
                descriptor_type: descriptor.descriptor_type,
                descriptor_count: descriptor.descriptor_count,
                stage_flags: stage_flags(stage),
            });
        }

        let Some(max_set) = sets.keys().next_back().copied() else {
            return Ok(Vec::new());
        };
        let mut layouts = Vec::new();
        for set in 0..=max_set {
            let mut bindings = sets.remove(&set).unwrap_or_default();
            bindings.sort_by_key(|binding| binding.binding);
            layouts.push(bindings);
        }
        Ok(layouts)
    }

    fn descriptor(
        &self,
        entrypoint: &str,
        variable: &Instruction,
    ) -> anyhow::Result<ReflectedDescriptor> {
        let storage_class = storage_class(variable, 0).ok_or_else(|| {
            anyhow!("shader `{entrypoint}` has descriptor variable without storage class")
        })?;
        let pointer_type = variable.result_type.ok_or_else(|| {
            anyhow!("shader `{entrypoint}` has descriptor variable without pointer type")
        })?;
        let pointee = self.pointer_pointee(pointer_type)?;
        let (_, descriptor_count) = self.strip_descriptor_array(entrypoint, pointee)?;

        match storage_class {
            StorageClass::Uniform => Ok(ReflectedDescriptor {
                descriptor_type: "UNIFORM_BUFFER",
                descriptor_count,
            }),
            StorageClass::UniformConstant => {
                let descriptor_type = self.uniform_constant_descriptor_type(entrypoint, pointee)?;
                Ok(ReflectedDescriptor {
                    descriptor_type,
                    descriptor_count,
                })
            }
            StorageClass::StorageBuffer => bail!(
                "shader `{entrypoint}` uses a storage buffer descriptor; storage descriptors are not supported yet"
            ),
            _ => bail!(
                "shader `{entrypoint}` uses unsupported descriptor storage class {storage_class:?}"
            ),
        }
    }

    fn strip_descriptor_array(&self, entrypoint: &str, type_id: u32) -> anyhow::Result<(u32, u32)> {
        let instruction = self
            .assignments
            .get(&type_id)
            .ok_or_else(|| anyhow!("missing SPIR-V descriptor type id {type_id}"))?;
        match instruction.class.opcode {
            Op::TypeArray => {
                let element = id_ref(instruction, 0)
                    .ok_or_else(|| anyhow!("SPIR-V array type id {type_id} lacks element type"))?;
                let length_id = id_ref(instruction, 1)
                    .ok_or_else(|| anyhow!("SPIR-V array type id {type_id} lacks length id"))?;
                let descriptor_count = self.constant_u32(length_id)?;
                let (element_type, nested_count) =
                    self.strip_descriptor_array(entrypoint, element)?;
                Ok((
                    element_type,
                    descriptor_count.checked_mul(nested_count).ok_or_else(|| {
                        anyhow!("shader `{entrypoint}` descriptor array count overflowed u32")
                    })?,
                ))
            }
            Op::TypeRuntimeArray => bail!(
                "shader `{entrypoint}` uses runtime-sized descriptor arrays, which are not supported"
            ),
            _ => Ok((type_id, 1)),
        }
    }

    fn constant_u32(&self, constant_id: u32) -> anyhow::Result<u32> {
        let instruction = self
            .assignments
            .get(&constant_id)
            .ok_or_else(|| anyhow!("missing SPIR-V constant id {constant_id}"))?;
        if instruction.class.opcode != Op::Constant {
            bail!("SPIR-V id {constant_id} is not an OpConstant");
        }
        literal_u32(instruction, 0)
            .ok_or_else(|| anyhow!("SPIR-V constant id {constant_id} is not a u32 literal"))
    }

    fn uniform_constant_descriptor_type(
        &self,
        entrypoint: &str,
        type_id: u32,
    ) -> anyhow::Result<&'static str> {
        let (element_type, _) = self.strip_descriptor_array(entrypoint, type_id)?;
        let instruction = self
            .assignments
            .get(&element_type)
            .ok_or_else(|| anyhow!("missing SPIR-V descriptor type id {element_type}"))?;
        match instruction.class.opcode {
            Op::TypeSampledImage => Ok("COMBINED_IMAGE_SAMPLER"),
            Op::TypeImage => {
                if sampled_image_mode(instruction) == Some(2) {
                    bail!(
                        "shader `{entrypoint}` uses a storage image descriptor; storage image descriptors are not supported yet"
                    )
                }
                Ok("SAMPLED_IMAGE")
            }
            Op::TypeSampler => Ok("SAMPLER"),
            _ => bail!(
                "shader `{entrypoint}` uses unsupported uniform-constant descriptor opcode {:?}",
                instruction.class.opcode
            ),
        }
    }

    fn vertex_inputs(
        &self,
        entrypoint: &str,
        interface_ids: &[u32],
    ) -> anyhow::Result<Vec<VertexInput>> {
        let mut reflected = Vec::new();
        for id in interface_ids {
            let Some(decorations) = self.decorations.get(id) else {
                continue;
            };
            if decorations.built_in {
                continue;
            }
            let Some(location) = decorations.location else {
                continue;
            };
            let variable = self.assignments.get(id).ok_or_else(|| {
                anyhow!("shader `{entrypoint}` location {location} is not assigned")
            })?;
            if storage_class(variable, 0) != Some(StorageClass::Input) {
                continue;
            }
            let pointer_type = variable.result_type.ok_or_else(|| {
                anyhow!("shader `{entrypoint}` input location {location} has no pointer type")
            })?;
            let pointee = self.pointer_pointee(pointer_type)?;
            let (format, size) = self.vertex_format(entrypoint, location, pointee)?;
            reflected.push((location, format, size));
        }
        build_vertex_inputs(entrypoint, reflected)
    }

    fn pointer_pointee(&self, pointer_type: u32) -> anyhow::Result<u32> {
        let instruction = self
            .assignments
            .get(&pointer_type)
            .ok_or_else(|| anyhow!("missing SPIR-V pointer type id {pointer_type}"))?;
        if instruction.class.opcode != Op::TypePointer {
            bail!("SPIR-V type id {pointer_type} is not an OpTypePointer");
        }
        id_ref(instruction, 1)
            .ok_or_else(|| anyhow!("SPIR-V pointer type id {pointer_type} lacks a pointee type"))
    }

    fn vertex_format(
        &self,
        entrypoint: &str,
        location: u32,
        type_id: u32,
    ) -> anyhow::Result<(&'static str, u32)> {
        let instruction = self
            .assignments
            .get(&type_id)
            .ok_or_else(|| anyhow!("missing SPIR-V vertex input type id {type_id}"))?;
        match instruction.class.opcode {
            Op::TypeFloat => {
                let width = literal_u32(instruction, 0)
                    .ok_or_else(|| anyhow!("SPIR-V float type id {type_id} lacks width"))?;
                scalar_format(
                    entrypoint,
                    location,
                    None,
                    raw_scalar(ScalarKind::Float, width)?,
                )
            }
            Op::TypeInt => {
                let width = literal_u32(instruction, 0)
                    .ok_or_else(|| anyhow!("SPIR-V int type id {type_id} lacks width"))?;
                let signed = literal_u32(instruction, 1)
                    .ok_or_else(|| anyhow!("SPIR-V int type id {type_id} lacks signedness"))?;
                let kind = if signed == 0 {
                    ScalarKind::Uint
                } else {
                    ScalarKind::Sint
                };
                scalar_format(entrypoint, location, None, raw_scalar(kind, width)?)
            }
            Op::TypeVector => {
                let component = id_ref(instruction, 0).ok_or_else(|| {
                    anyhow!("SPIR-V vector type id {type_id} lacks component type")
                })?;
                let components = literal_u32(instruction, 1).ok_or_else(|| {
                    anyhow!("SPIR-V vector type id {type_id} lacks component count")
                })?;
                let (kind, width) = self.scalar_type(component)?;
                scalar_format(
                    entrypoint,
                    location,
                    Some(raw_vector_size(entrypoint, location, components)?),
                    raw_scalar(kind, width)?,
                )
            }
            Op::TypeMatrix => bail!(
                "shader `{entrypoint}` location {location} uses a matrix vertex input, which is not supported"
            ),
            _ => bail!(
                "shader `{entrypoint}` location {location} uses unsupported SPIR-V vertex input opcode {:?}",
                instruction.class.opcode
            ),
        }
    }

    fn scalar_type(&self, type_id: u32) -> anyhow::Result<(ScalarKind, u32)> {
        let instruction = self
            .assignments
            .get(&type_id)
            .ok_or_else(|| anyhow!("missing SPIR-V scalar type id {type_id}"))?;
        match instruction.class.opcode {
            Op::TypeFloat => Ok((
                ScalarKind::Float,
                literal_u32(instruction, 0)
                    .ok_or_else(|| anyhow!("SPIR-V float type id {type_id} lacks width"))?,
            )),
            Op::TypeInt => {
                let kind = if literal_u32(instruction, 1).unwrap_or(0) == 0 {
                    ScalarKind::Uint
                } else {
                    ScalarKind::Sint
                };
                Ok((
                    kind,
                    literal_u32(instruction, 0)
                        .ok_or_else(|| anyhow!("SPIR-V int type id {type_id} lacks width"))?,
                ))
            }
            _ => bail!("SPIR-V type id {type_id} is not a scalar"),
        }
    }
}

fn raw_scalar(kind: ScalarKind, width_bits: u32) -> anyhow::Result<Scalar> {
    let width = match width_bits {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        _ => {
            bail!("SPIR-V scalar width {width_bits} is unsupported; expected 8, 16, 32, or 64 bits")
        }
    };
    Ok(Scalar { kind, width })
}

fn raw_vector_size(entrypoint: &str, location: u32, components: u32) -> anyhow::Result<VectorSize> {
    match components {
        2 => Ok(VectorSize::Bi),
        3 => Ok(VectorSize::Tri),
        4 => Ok(VectorSize::Quad),
        _ => bail!(
            "shader `{entrypoint}` location {location} uses unsupported vector width {components}"
        ),
    }
}

fn id_ref(instruction: &Instruction, index: usize) -> Option<u32> {
    match instruction.operands.get(index) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

fn literal_u32(instruction: &Instruction, index: usize) -> Option<u32> {
    match instruction.operands.get(index) {
        Some(Operand::LiteralBit32(value)) => Some(*value),
        _ => None,
    }
}

fn literal_string(instruction: &Instruction, index: usize) -> Option<&str> {
    match instruction.operands.get(index) {
        Some(Operand::LiteralString(value)) => Some(value.as_str()),
        _ => None,
    }
}

fn decoration(instruction: &Instruction, index: usize) -> Option<Decoration> {
    match instruction.operands.get(index) {
        Some(Operand::Decoration(value)) => Some(*value),
        _ => None,
    }
}

fn execution_model(instruction: &Instruction, index: usize) -> Option<ExecutionModel> {
    match instruction.operands.get(index) {
        Some(Operand::ExecutionModel(value)) => Some(*value),
        _ => None,
    }
}

fn storage_class(instruction: &Instruction, index: usize) -> Option<StorageClass> {
    match instruction.operands.get(index) {
        Some(Operand::StorageClass(value)) => Some(*value),
        _ => None,
    }
}

fn sampled_image_mode(instruction: &Instruction) -> Option<u32> {
    literal_u32(instruction, TYPE_IMAGE_SAMPLED_OPERAND)
}

fn stage_flags(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "VERTEX",
        ShaderStage::Fragment => "FRAGMENT",
        ShaderStage::Compute => "COMPUTE",
    }
}

fn shader_type_name(entrypoint: &str, stage: ShaderStage) -> anyhow::Result<String> {
    let mut name = String::new();
    let suffix = entrypoint.rsplit_once('_').map(|(_, suffix)| suffix);
    let stem = match suffix {
        Some("vs" | "fs" | "cs") => entrypoint
            .rsplit_once('_')
            .map_or(entrypoint, |(stem, _)| stem),
        _ => entrypoint,
    };
    for part in stem.split('_').filter(|part| !part.is_empty()) {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            name.extend(first.to_uppercase());
            name.push_str(chars.as_str());
        }
    }
    if name.is_empty() {
        bail!("shader entry point `{entrypoint}` does not contain a valid Rust type-name stem");
    }
    name.push_str(match suffix {
        Some("vs") => {
            ensure_entrypoint_suffix_matches_stage(entrypoint, stage, ShaderStage::Vertex)?;
            "VS"
        }
        Some("fs") => {
            ensure_entrypoint_suffix_matches_stage(entrypoint, stage, ShaderStage::Fragment)?;
            "FS"
        }
        Some("cs") => {
            ensure_entrypoint_suffix_matches_stage(entrypoint, stage, ShaderStage::Compute)?;
            "CS"
        }
        _ => shader_stage_suffix(stage),
    });
    Ok(name)
}

fn ensure_entrypoint_suffix_matches_stage(
    entrypoint: &str,
    actual: ShaderStage,
    expected: ShaderStage,
) -> anyhow::Result<()> {
    if actual == expected {
        Ok(())
    } else {
        bail!(
            "shader entry point `{entrypoint}` suffix implies {expected:?}, but reflection reported {actual:?}"
        )
    }
}

fn shader_stage_suffix(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "VS",
        ShaderStage::Fragment => "FS",
        ShaderStage::Compute => "CS",
    }
}

fn generate_shader_module(shaders: &[ReflectedShader]) -> TokenStream {
    let shaders = shaders.iter().map(generate_shader).collect::<Vec<_>>();

    quote! {
        use dirk_rhi::{
            BindGroupLayoutEntry, BindingType, ShaderStage, ShaderStages, VertexAttribute,
            VertexBufferLayout, VertexFormat, VertexStepMode,
        };
        use crate::shaders::metadata::{FragmentShader, Shader, VertexShader};

        #(#shaders)*
    }
}

fn generate_shader(shader: &ReflectedShader) -> TokenStream {
    let const_prefix = shader_const_prefix(&shader.entrypoint);
    let type_name = format_ident!("{}", shader.type_name);
    let entrypoint = &shader.entrypoint;
    let stage = match shader.stage {
        ShaderStage::Vertex => quote!(ShaderStage::Vertex),
        ShaderStage::Fragment => quote!(ShaderStage::Fragment),
        ShaderStage::Compute => quote!(ShaderStage::Compute),
    };

    let set_binding_idents = shader
        .set_layouts
        .iter()
        .enumerate()
        .map(|(set, _)| format_ident!("{const_prefix}_SET_{set}_BINDINGS"))
        .collect::<Vec<_>>();

    let set_bindings =
        shader
            .set_layouts
            .iter()
            .zip(&set_binding_idents)
            .map(|(bindings, ident)| {
                let bindings = bindings.iter().map(generate_descriptor_binding);
                quote! {
                    const #ident: &[BindGroupLayoutEntry] = &[
                        #(#bindings,)*
                    ];
                }
            });

    let vertex_input = if shader.stage == ShaderStage::Vertex {
        let input_attributes_ident = format_ident!("{const_prefix}_INPUT_ATTRIBUTES");
        let input_layouts_ident = format_ident!("{const_prefix}_INPUT_LAYOUTS");
        let stride = shader
            .vertex_inputs
            .last()
            .map_or(0, |input| input.offset + input.size);
        let attributes = shader.vertex_inputs.iter().map(generate_vertex_attribute);

        quote! {
            const #input_attributes_ident: &[VertexAttribute] = &[
                #(#attributes,)*
            ];

            const #input_layouts_ident: &[VertexBufferLayout<'static>] = &[
                VertexBufferLayout {
                    stride: #stride,
                    step_mode: VertexStepMode::Vertex,
                    attributes: #input_attributes_ident,
                },
            ];

            impl VertexShader for #type_name {
                const INPUT_LAYOUTS: &'static [VertexBufferLayout<'static>] = #input_layouts_ident;
            }
        }
    } else {
        quote! {}
    };

    let stage_impl = match shader.stage {
        ShaderStage::Vertex | ShaderStage::Compute => quote! {},
        ShaderStage::Fragment => quote! {
            impl FragmentShader for #type_name {}
        },
    };

    quote! {
        #(#set_bindings)*
        #vertex_input

        pub struct #type_name;

        impl Shader for #type_name {
            const CODE: ShaderCode = shader_code!(#entrypoint);
            const ENTRYPOINT: &'static str = #entrypoint;
            const STAGE: ShaderStage = #stage;
            const SET_LAYOUTS: &'static [&'static [BindGroupLayoutEntry]] = &[
                #(#set_binding_idents,)*
            ];
        }

        #stage_impl
    }
}

fn generate_descriptor_binding(binding: &DescriptorBinding) -> TokenStream {
    assert_eq!(
        binding.descriptor_count, 1,
        "descriptor arrays are not supported by the RHI"
    );
    let descriptor_type = rhi_binding_type(binding.descriptor_type);
    let stage_flags = rhi_shader_stages(binding.stage_flags);
    let binding_index = binding.binding;

    quote! {
        BindGroupLayoutEntry {
            binding: #binding_index,
            ty: BindingType::#descriptor_type,
            visibility: ShaderStages::#stage_flags,
        }
    }
}

fn generate_vertex_attribute(input: &VertexInput) -> TokenStream {
    let location = input.location;
    let format = rhi_format(input.format);
    let offset = input.offset;

    quote! {
        VertexAttribute {
            location: #location,
            format: VertexFormat::#format,
            offset: #offset,
        }
    }
}

fn shader_const_prefix(entrypoint: &str) -> String {
    entrypoint
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn rhi_binding_type(name: &str) -> Ident {
    let name = match name {
        "UNIFORM_BUFFER" => "UniformBuffer",
        "STORAGE_BUFFER" => "StorageBuffer",
        "COMBINED_IMAGE_SAMPLER" | "SAMPLED_IMAGE" | "SAMPLER" => "SampledImage",
        "STORAGE_IMAGE" => "StorageImage",
        _ => panic!("unsupported reflected descriptor type {name}"),
    };
    Ident::new(name, Span::call_site())
}

fn rhi_shader_stages(name: &str) -> Ident {
    let name = match name {
        "VERTEX" => "VERTEX",
        "FRAGMENT" => "FRAGMENT",
        "COMPUTE" => "COMPUTE",
        _ => panic!("unsupported reflected shader stage {name}"),
    };
    Ident::new(name, Span::call_site())
}

fn rhi_format(name: &str) -> Ident {
    let name = match name {
        "R32G32_SFLOAT" => "Float32x2",
        "R32G32B32_SFLOAT" => "Float32x3",
        _ => panic!("unsupported reflected vertex format {name}"),
    };
    Ident::new(name, Span::call_site())
}
