//! Builds rust-gpu shader crates for `dirk_rhi`.

mod msl;
mod push;

use std::{
    collections::{BTreeMap, HashMap},
    env,
    fmt::Write as _,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, anyhow, bail, ensure};
use cargo_gpu_install::{
    install::{Install, InstalledBackend},
    spirv_builder::{Capability, ModuleResult, SpirvMetadata},
};
use dirk_shaders::bindless::heap;
use rspirv::{
    dr::{Instruction, Module, Operand},
    spirv::{Decoration, Dim, ExecutionModel, ImageFormat, Op, StorageClass},
};

/// Compiles the rust-gpu shader crate at `shader_crate` (relative to the
/// calling crate) into one SPIR-V module per entry point, and writes
/// `shaders.rs` to `OUT_DIR`. Include it with
/// `include!(concat!(env!("OUT_DIR"), "/shaders.rs"))`.
///
/// The generated file declares a type per entry point, named after it in
/// `PascalCase` (`main_vs` becomes `MainVs`), implementing `dirk_rhi::Shader`
/// with the stage, push data size and whether it writes storage buffers or
/// images reflected from its module, and the push data type its
/// `#[spirv(push_constant)]` parameter refers to in the crate's source; see
/// the `push` module. Creating a shader whose push data block
/// differs in size from that type fails the build. Modules declaring
/// resources other than the bindless heap's arrays as
/// `dirk_shaders::bindless::heap` lays them out fail the shader build. For
/// Apple targets, each module is also translated to the Metal Shading
/// Language and written beside it; see the `msl` module.
///
/// # Errors
///
/// Returns an error if the rust-gpu backend cannot be installed, the crate
/// does not compile, or a module cannot be read, reflected or translated.
pub fn build_shaders(shader_crate: impl AsRef<Path>) -> anyhow::Result<()> {
    let shader_crate = shader_crate.as_ref();
    let out_dir = PathBuf::from(env::var("OUT_DIR").context("OUT_DIR is not set")?);
    let metal =
        env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| matches!(os.as_str(), "macos" | "ios"));
    let backend = install(shader_crate)?;
    let mut builder = backend.to_spirv_builder(shader_crate, "spirv-unknown-vulkan1.3");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(false);
    builder.multimodule = true;
    builder.spirv_metadata = SpirvMetadata::None;
    // Shaders read the bindless heap's runtime arrays with non-uniform
    // indices; `dirk_rhi` enables these features on every device.
    builder.capabilities.extend([
        Capability::RuntimeDescriptorArray,
        Capability::ShaderNonUniform,
        Capability::SampledImageArrayNonUniformIndexing,
        Capability::StorageImageArrayNonUniformIndexing,
        Capability::StorageBufferArrayNonUniformIndexing,
    ]);
    let ModuleResult::MultiModule(modules) = builder
        .build()
        .with_context(|| format!("compiling the shaders in {}", shader_crate.display()))?
        .module
    else {
        bail!("expected one SPIR-V module per entry point");
    };

    let source = push::Source::new(shader_crate)?;
    let mut generated = String::new();
    for (entry_point, module) in modules.into_iter().collect::<BTreeMap<_, _>>() {
        let spirv =
            fs::read(&module).with_context(|| format!("reading the SPIR-V of `{entry_point}`"))?;
        let reflected = Reflected::new(&entry_point, &spirv)?;
        let file = entry_point.replace("::", "-");
        if metal {
            let msl = msl::translate(&entry_point, &spirv, reflected.stage)?;
            fs::write(out_dir.join(format!("{file}.metal")), msl)?;
        }
        let path = out_dir.join(format!("{file}.spv"));
        fs::write(&path, &spirv)?;
        let path = path.to_str().context("OUT_DIR is not valid UTF-8")?;
        let push = source
            .push_type(&entry_point)
            .with_context(|| format!("finding the push data type of `{entry_point}`"))?;
        writeln!(
            generated,
            "/// The `{entry_point}` shader entry point.\n\
             pub struct {name};\n\
             // SAFETY: rust-gpu compiled this entry point for Vulkan 1.3, the\n\
             // shader build reflected its stage, push data size and storage\n\
             // writes, found its push data type in its source, and checked the\n\
             // heap arrays it declares.\n\
             unsafe impl dirk_rhi::Shader for {name} {{\n    \
                 type Push = {push};\n    \
                 const ENTRY_POINT: &'static str = {entry_point:?};\n    \
                 const STAGE: dirk_rhi::ShaderStage = dirk_rhi::ShaderStage::{stage:?};\n    \
                 const PUSH_SIZE: usize = {push_size};\n    \
                 const WRITES_STORAGE: bool = {writes_storage};\n    \
                 const CODE: &'static [u8] = include_bytes!({path:?});\n\
             }}\n",
            name = type_name(&entry_point),
            stage = reflected.stage,
            push_size = reflected.push_size,
            writes_storage = reflected.writes_storage,
        )?;
    }
    fs::write(out_dir.join("shaders.rs"), generated)?;
    Ok(())
}

/// Installs the rust-gpu toolchain and backend `shader_crate` needs, if
/// missing. Build scripts compiling shaders run concurrently, and rustup
/// installs are not safe to run at once, so installs are serialized with a
/// lock file shared by every build on the machine.
///
/// # Errors
///
/// Returns an error if the lock cannot be taken or the install fails.
pub fn install(shader_crate: &Path) -> anyhow::Result<InstalledBackend> {
    let path = env::temp_dir().join("dirk-rust-gpu-install.lock");
    let lock = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    lock.lock()
        .with_context(|| format!("locking {}", path.display()))?;
    // Unlocked when `lock` is dropped, once installed.
    Install::from_shader_crate(shader_crate.to_path_buf()).run()
}

/// The stage an entry point runs in, named like `dirk_rhi::ShaderStage`.
#[derive(Clone, Copy, Debug)]
enum Stage {
    Vertex,
    Fragment,
    Compute,
}

/// What the shader build reflects from an entry point's module.
struct Reflected {
    stage: Stage,
    /// Size of the push data block, zero without one.
    push_size: u32,
    /// Whether it writes storage buffers or images.
    writes_storage: bool,
}

impl Reflected {
    /// Reflects the module `spirv` holding only `entry_point`.
    fn new(entry_point: &str, spirv: &[u8]) -> anyhow::Result<Self> {
        let module = rspirv::dr::load_bytes(spirv)
            .map_err(|error| anyhow!("{error}"))
            .with_context(|| format!("parsing the SPIR-V of `{entry_point}`"))?;
        let [entry] = module.entry_points.as_slice() else {
            bail!("the module of `{entry_point}` does not hold exactly one entry point");
        };
        let stage = match entry.operands.first() {
            Some(Operand::ExecutionModel(ExecutionModel::Vertex)) => Stage::Vertex,
            Some(Operand::ExecutionModel(ExecutionModel::Fragment)) => Stage::Fragment,
            Some(Operand::ExecutionModel(ExecutionModel::GLCompute)) => Stage::Compute,
            model => bail!("`{entry_point}` runs in unsupported stage {model:?}"),
        };
        let push_size = Types(&module)
            .push_size()
            .with_context(|| format!("reflecting the push data of `{entry_point}`"))?;
        Types(&module)
            .check_bindings()
            .with_context(|| format!("checking the heap arrays `{entry_point}` declares"))?;
        let writes_storage = Types(&module)
            .writes_storage()
            .with_context(|| format!("reflecting the storage `{entry_point}` writes"))?;
        Ok(Self {
            stage,
            push_size,
            writes_storage,
        })
    }
}

/// Type lookups in a SPIR-V module.
struct Types<'a>(&'a Module);

impl Types<'_> {
    /// Size of the module's push constant block, zero without one.
    fn push_size(&self) -> anyhow::Result<u32> {
        let mut blocks = self.0.types_global_values.iter().filter(|instruction| {
            instruction.class.opcode == Op::Variable
                && instruction.operands.first()
                    == Some(&Operand::StorageClass(StorageClass::PushConstant))
        });
        let Some(variable) = blocks.next() else {
            return Ok(0);
        };
        ensure!(blocks.next().is_none(), "more than one push constant block");
        let pointer = self.get(variable.result_type.context("untyped variable")?)?;
        self.size(id_operand(pointer, 1)?)
    }

    /// Checks that every resource the module declares is one of the heap's
    /// arrays, in descriptor set 0, declared with the elements it holds.
    fn check_bindings(&self) -> anyhow::Result<()> {
        let resources = self.0.types_global_values.iter().filter(|instruction| {
            instruction.class.opcode == Op::Variable
                && matches!(
                    instruction.operands.first(),
                    Some(Operand::StorageClass(
                        StorageClass::UniformConstant
                            | StorageClass::Uniform
                            | StorageClass::StorageBuffer
                    ))
                )
        });
        for variable in resources {
            let id = variable.result_id.context("unnamed variable")?;
            let set = self.decoration(id, None, Decoration::DescriptorSet);
            let binding = self.decoration(id, None, Decoration::Binding);
            let (Some(0), Some(binding)) = (set, binding) else {
                bail!("resource {id} is not in descriptor set 0, the heap, but in {set:?}");
            };
            let expected = Element::at(binding)
                .with_context(|| format!("binding {binding} is not one of the heap's arrays"))?;
            let declared = self
                .element(variable)
                .with_context(|| format!("reflecting the array at binding {binding}"))?;
            ensure!(
                declared == expected,
                "binding {binding} holds {expected:?}, not the {declared:?} declared"
            );
        }
        Ok(())
    }

    /// Whether the module writes storage buffers or images: stores or
    /// atomics other than loads through pointers into storage buffers or
    /// image texels, or image writes, which only storage images take.
    fn writes_storage(&self) -> anyhow::Result<bool> {
        let functions = &self.0.functions;
        // The type of every pointer the module defines.
        let types: HashMap<u32, u32> = self
            .0
            .types_global_values
            .iter()
            .chain(functions.iter().flat_map(|function| {
                function
                    .parameters
                    .iter()
                    .chain(function.blocks.iter().flat_map(|block| &block.instructions))
            }))
            .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
            .collect();
        let instructions = functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions);
        for instruction in instructions {
            let opcode = instruction.class.opcode;
            if opcode == Op::ImageWrite {
                return Ok(true);
            }
            let writes_through_pointer =
                matches!(opcode, Op::Store | Op::CopyMemory | Op::CopyMemorySized)
                    || (instruction.class.opname.starts_with("OpAtomic")
                        && opcode != Op::AtomicLoad);
            if !writes_through_pointer {
                continue;
            }
            let pointer = id_operand(instruction, 0)?;
            let ty = types
                .get(&pointer)
                .with_context(|| format!("pointer {pointer} has no type"))?;
            let class = self.get(*ty)?.operands.first();
            if matches!(
                class,
                Some(Operand::StorageClass(
                    StorageClass::StorageBuffer | StorageClass::Image
                ))
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The elements of the runtime array `variable` points to.
    fn element(&self, variable: &Instruction) -> anyhow::Result<Element> {
        let pointer = self.get(variable.result_type.context("untyped variable")?)?;
        let array = self.get(id_operand(pointer, 1)?)?;
        ensure!(
            array.class.opcode == Op::TypeRuntimeArray,
            "it is not a runtime array"
        );
        let element = self.get(id_operand(array, 0)?)?;
        let class = variable.operands.first();
        Ok(match (class, element.class.opcode) {
            (Some(Operand::StorageClass(StorageClass::StorageBuffer)), Op::TypeStruct) => {
                Element::StorageBuffer
            }
            (Some(Operand::StorageClass(StorageClass::UniformConstant)), Op::TypeSampler) => {
                Element::Sampler
            }
            (Some(Operand::StorageClass(StorageClass::UniformConstant)), Op::TypeImage) => {
                let Some(&Operand::Dim(dim)) = element.operands.get(1) else {
                    bail!("the image has no dimensionality");
                };
                let [depth, arrayed, multisampled, sampled] =
                    [2, 3, 4, 5].map(|operand| literal_operand(element, operand));
                let (arrayed, depth) = (arrayed? == 1, depth? == 1);
                ensure!(multisampled? == 0, "the images are multisampled");
                match sampled? {
                    1 => Element::Texture {
                        dim,
                        arrayed,
                        depth,
                    },
                    2 if !depth
                        && element.operands.get(6)
                            == Some(&Operand::ImageFormat(ImageFormat::Unknown)) =>
                    {
                        Element::StorageImage { dim, arrayed }
                    }
                    _ => bail!("the images are neither sampled nor formatless storage images"),
                }
            }
            (class, opcode) => bail!("{opcode:?} elements in {class:?} are not in the heap"),
        })
    }

    /// Size in bytes of the type `id`, as laid out by its decorations.
    fn size(&self, id: u32) -> anyhow::Result<u32> {
        let ty = self.get(id)?;
        Ok(match ty.class.opcode {
            Op::TypeInt | Op::TypeFloat => literal_operand(ty, 0)? / 8,
            Op::TypeVector | Op::TypeMatrix => {
                self.size(id_operand(ty, 0)?)? * literal_operand(ty, 1)?
            }
            Op::TypeArray => {
                let length = literal_operand(self.get(id_operand(ty, 1)?)?, 0)?;
                let stride = match self.decoration(id, None, Decoration::ArrayStride) {
                    Some(stride) => stride,
                    None => self.size(id_operand(ty, 0)?)?,
                };
                length * stride
            }
            Op::TypeStruct => {
                let mut end = 0;
                for member in 0..ty.operands.len() {
                    let offset = self
                        .decoration(id, Some(member), Decoration::Offset)
                        .with_context(|| format!("member {member} of type {id} has no offset"))?;
                    end = end.max(offset + self.size(id_operand(ty, member)?)?);
                }
                end
            }
            opcode => bail!("type {id} ({opcode:?}) cannot be push data"),
        })
    }

    /// The type or constant instruction defining `id`.
    fn get(&self, id: u32) -> anyhow::Result<&Instruction> {
        self.0
            .types_global_values
            .iter()
            .find(|instruction| instruction.result_id == Some(id))
            .with_context(|| format!("id {id} is not a type or constant"))
    }

    /// The literal of `decoration` on `id`, or on its `member` when given.
    fn decoration(&self, id: u32, member: Option<usize>, decoration: Decoration) -> Option<u32> {
        self.0.annotations.iter().find_map(|instruction| {
            let (opcode, rest) = match member {
                None => (Op::Decorate, 1),
                Some(_) => (Op::MemberDecorate, 2),
            };
            let target = instruction.operands.first() == Some(&Operand::IdRef(id))
                && member.is_none_or(|member| {
                    instruction.operands.get(1)
                        == u32::try_from(member)
                            .ok()
                            .map(Operand::LiteralBit32)
                            .as_ref()
                });
            (instruction.class.opcode == opcode
                && target
                && instruction.operands.get(rest) == Some(&Operand::Decoration(decoration)))
            .then(|| literal_operand(instruction, rest + 1).ok())
            .flatten()
        })
    }
}

/// What each element of one of the heap's arrays is, as shaders declare it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Element {
    /// Sampled images, declared as depth images or not.
    Texture {
        dim: Dim,
        arrayed: bool,
        depth: bool,
    },
    /// Storage images, declared without a format.
    StorageImage { dim: Dim, arrayed: bool },
    /// Samplers.
    Sampler,
    /// Storage buffers.
    StorageBuffer,
}

impl Element {
    /// The elements of the heap's array at `binding`, if it has one there.
    fn at(binding: u32) -> Option<Self> {
        let texture = |dim, arrayed| Self::Texture {
            dim,
            arrayed,
            depth: false,
        };
        let storage = |dim, arrayed| Self::StorageImage { dim, arrayed };
        Some(match binding {
            heap::TEXTURES_2D => texture(Dim::Dim2D, false),
            heap::TEXTURES_2D_ARRAY => texture(Dim::Dim2D, true),
            heap::TEXTURES_CUBE => texture(Dim::DimCube, false),
            heap::TEXTURES_CUBE_ARRAY => texture(Dim::DimCube, true),
            heap::TEXTURES_3D => texture(Dim::Dim3D, false),
            heap::STORAGE_IMAGES_2D => storage(Dim::Dim2D, false),
            heap::STORAGE_IMAGES_2D_ARRAY => storage(Dim::Dim2D, true),
            heap::STORAGE_IMAGES_3D => storage(Dim::Dim3D, false),
            heap::SAMPLERS => Self::Sampler,
            heap::BUFFERS => Self::StorageBuffer,
            _ => return None,
        })
    }
}

/// The id operand at `index` of `instruction`.
fn id_operand(instruction: &Instruction, index: usize) -> anyhow::Result<u32> {
    match instruction.operands.get(index) {
        Some(Operand::IdRef(id)) => Ok(*id),
        operand => bail!("expected an id at operand {index}, found {operand:?}"),
    }
}

/// The 32-bit literal operand at `index` of `instruction`.
fn literal_operand(instruction: &Instruction, index: usize) -> anyhow::Result<u32> {
    match instruction.operands.get(index) {
        Some(Operand::LiteralBit32(value)) => Ok(*value),
        operand => bail!("expected a literal at operand {index}, found {operand:?}"),
    }
}

/// The `PascalCase` type name of `entry_point`.
fn type_name(entry_point: &str) -> String {
    entry_point
        .split(['_', ':'])
        .filter(|part| !part.is_empty())
        .flat_map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .into_iter()
                .flat_map(char::to_uppercase)
                .chain(chars)
        })
        .collect()
}

#[cfg(test)]
mod tests;
