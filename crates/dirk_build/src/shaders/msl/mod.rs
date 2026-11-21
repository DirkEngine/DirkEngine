//! Translation of SPIR-V entry points to the Metal Shading Language, for
//! Apple targets, which `dirk_rhi` runs on Metal. SPIRV-Cross translates them
//! on macOS hosts only, so other hosts never compile it.
//!
//! - The bindless heap, descriptor set 0, becomes a tier-2 argument buffer at
//!   [`HEAP_BUFFER`]. Metal lays argument buffers out like C structs, eight
//!   bytes per texture, sampler or buffer, so every shader declares every
//!   array in binding order (see [`lay_out_heap`]): slot `s` of the array at
//!   binding `b` is element `first_element(b) + s`. The storage buffer
//!   lengths that bounds checks read follow, at [`BUFFER_LENGTHS`].
//! - Push data is a buffer at [`PUSH_BUFFER`].
//! - Vertex shaders flip y: the RHI's clip space is Vulkan's, whose y points
//!   down, and Metal's points up.

#[cfg(target_os = "macos")]
pub(super) use apple::translate;

/// Without SPIRV-Cross, other hosts cannot build shaders for Apple targets.
#[cfg(not(target_os = "macos"))]
pub(super) fn translate(entry_point: &str, _: &[u8], _: super::Stage) -> anyhow::Result<String> {
    anyhow::bail!("translating `{entry_point}` for an Apple target needs a macOS host")
}

#[cfg(target_os = "macos")]
mod apple {
    use std::{collections::BTreeMap, fmt::Write as _, num::NonZeroU32};

    use anyhow::{Context, bail, ensure};
    use dirk_shaders::bindless::heap::{
        CAPACITIES, first_element,
        metal::{BUFFER_LENGTHS, HEAP_BUFFER, PUSH_BUFFER},
    };
    use spirv_cross2::{
        Compiler, Module,
        compile::msl::{
            ArgumentBuffersTier, BindTarget, CompilerOptions, MslVersion, ResourceBinding,
        },
        spirv::ExecutionModel,
        targets::Msl,
    };

    use super::super::Stage;

    /// Translates the module `spirv`, which holds only `entry_point`.
    pub(in super::super) fn translate(
        entry_point: &str,
        spirv: &[u8],
        stage: Stage,
    ) -> anyhow::Result<String> {
        let context = || format!("translating `{entry_point}` to MSL");
        let (words, []) = spirv.as_chunks::<4>() else {
            bail!("the SPIR-V of `{entry_point}` is not a whole number of words");
        };
        let words: Vec<u32> = words.iter().map(|word| u32::from_le_bytes(*word)).collect();
        let mut compiler =
            Compiler::<Msl>::new(Module::from_words(&words)).with_context(context)?;
        let model = match stage {
            Stage::Vertex => ExecutionModel::Vertex,
            Stage::Fragment => ExecutionModel::Fragment,
            Stage::Compute => ExecutionModel::GLCompute,
        };
        let target = |id, count| BindTarget {
            buffer: id,
            texture: id,
            sampler: id,
            count: NonZeroU32::new(count),
        };
        for (binding, &capacity) in (0..).zip(&CAPACITIES) {
            compiler
                .add_resource_binding(
                    model,
                    ResourceBinding::from_qualified(0, binding),
                    &target(first_element(binding), capacity),
                )
                .with_context(context)?;
        }
        let bindings = [
            (ResourceBinding::ArgumentBuffer(0), HEAP_BUFFER),
            (ResourceBinding::PushConstantBuffer, PUSH_BUFFER),
            (ResourceBinding::BufferSizeBuffer(0), BUFFER_LENGTHS),
        ];
        for (binding, id) in bindings {
            compiler
                .add_resource_binding(model, binding, &target(id, 0))
                .with_context(context)?;
        }
        // The heap is far larger than the constant address space allows.
        compiler
            .set_argument_buffer_device_address_space(0, true)
            .with_context(context)?;
        let mut options = CompilerOptions::default();
        options.common.flip_vertex_y = true;
        options.version = MslVersion::new(3, 0, 0);
        options.argument_buffers = true;
        options.argument_buffers_tier = ArgumentBuffersTier::Tier2;
        let msl = compiler
            .compile(&options)
            .with_context(context)?
            .to_string();
        lay_out_heap(&msl).with_context(context)
    }

    /// Declares every array of the heap's argument buffer in binding order,
    /// with unused padding for those `msl` does not read, so that every
    /// shader lays the heap out alike. Shaders reading no heap are returned
    /// unchanged.
    pub(super) fn lay_out_heap(msl: &str) -> anyhow::Result<String> {
        const DECLARATION: &str = "struct spvDescriptorSetBuffer0\n{\n";
        let Some(start) = msl.find(DECLARATION).map(|start| start + DECLARATION.len()) else {
            return Ok(msl.to_owned());
        };
        let end = start
            + msl[start..]
                .find("};")
                .context("unterminated heap struct")?;
        let mut members = BTreeMap::new();
        // Shaders may read one array as several types, such as storage
        // buffers of different elements. SPIRV-Cross then declares the first
        // and comments out the others, which it reads through casts of it.
        let declarations = msl[start..end]
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"));
        for line in declarations {
            let id: u32 = line
                .split_once("[[id(")
                .and_then(|(_, rest)| rest.split_once(")]]"))
                .and_then(|(id, _)| id.parse().ok())
                .with_context(|| format!("heap member `{line}` has no id"))?;
            if let Some(other) = members.insert(id, line) {
                bail!("heap members `{other}` and `{line}` share id {id}");
            }
        }
        let mut body = String::new();
        for (binding, capacity) in (0..).zip(CAPACITIES) {
            let id = first_element(binding);
            match members.remove(&id) {
                Some(line) => writeln!(body, "{line}")?,
                None => writeln!(
                    body,
                    "    array<texture2d<float>, {capacity}> dirk_unused_{binding} [[id({id})]];"
                )?,
            }
        }
        if let Some(line) = members.remove(&BUFFER_LENGTHS) {
            writeln!(body, "{line}")?;
        }
        ensure!(
            members.is_empty(),
            "heap members {members:?} are not at the start of an array"
        );
        Ok(format!("{}{body}{}", &msl[..start], &msl[end..]))
    }
}

// SPIRV-Cross, which the tests need, builds only on macOS hosts.
#[cfg(all(test, target_os = "macos"))]
mod tests;
