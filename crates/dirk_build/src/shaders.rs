//! Builds rust-gpu shader crates for `dirk_rhi`.

mod msl;

use std::{
    collections::BTreeMap,
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
use rspirv::{
    dr::{Instruction, Module, Operand},
    spirv::{Decoration, ExecutionModel, Op, StorageClass},
};

/// Compiles the rust-gpu shader crate at `shader_crate` (relative to the
/// calling crate) into one SPIR-V module per entry point, and writes
/// `shaders.rs` to `OUT_DIR`. Include it with
/// `include!(concat!(env!("OUT_DIR"), "/shaders.rs"))`.
///
/// The generated file declares a type per entry point, named after it in
/// `PascalCase` (`main_vs` becomes `MainVs`), implementing `dirk_rhi::Shader`
/// with the stage and push data size reflected from its module. Creating a
/// shader with push data of another size then fails the build. For Apple
/// targets, which `dirk_rhi` runs on Metal, each module is translated to the
/// Metal Shading Language; see [`msl`].
///
/// # Errors
///
/// Returns an error if the rust-gpu backend cannot be installed, the crate
/// does not compile, or a module cannot be read, reflected or translated.
pub fn build_shaders(shader_crate: impl AsRef<Path>) -> anyhow::Result<()> {
    let shader_crate = shader_crate.as_ref();
    let out_dir = PathBuf::from(env::var("OUT_DIR").context("OUT_DIR is not set")?);
    let metal = env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|vendor| vendor == "apple");
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

    let mut generated = String::new();
    for (entry_point, module) in modules.into_iter().collect::<BTreeMap<_, _>>() {
        let spirv =
            fs::read(&module).with_context(|| format!("reading the SPIR-V of `{entry_point}`"))?;
        let reflected = Reflected::new(&entry_point, &spirv)?;
        let (extension, code) = if metal {
            ("metal", msl::translate(&entry_point, &spirv, reflected.stage)?.into_bytes())
        } else {
            ("spv", spirv)
        };
        let path = out_dir.join(format!("{}.{extension}", entry_point.replace("::", "-")));
        fs::write(&path, code)?;
        let path = path.to_str().context("OUT_DIR is not valid UTF-8")?;
        writeln!(
            generated,
            "/// The `{entry_point}` shader entry point.\n\
             pub struct {name};\n\
             // SAFETY: rust-gpu compiled this entry point for Vulkan 1.3, the\n\
             // shader build reflected its stage and push data size, and\n\
             // translated it for Metal on Apple targets.\n\
             unsafe impl dirk_rhi::Shader for {name} {{\n    \
                 const ENTRY_POINT: &'static str = {entry_point:?};\n    \
                 const STAGE: dirk_rhi::ShaderStage = dirk_rhi::ShaderStage::{stage:?};\n    \
                 const PUSH_SIZE: usize = {push_size};\n    \
                 const CODE: &'static [u8] = include_bytes!({path:?});\n\
             }}\n",
            name = type_name(&entry_point),
            stage = reflected.stage,
            push_size = reflected.push_size,
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
        Ok(Self { stage, push_size })
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
mod tests {
    use super::type_name;

    #[test]
    fn entry_points_become_pascal_case_types() {
        assert_eq!(type_name("fill_vs"), "FillVs");
        assert_eq!(type_name("mesh::shadow_fs"), "MeshShadowFs");
    }
}
