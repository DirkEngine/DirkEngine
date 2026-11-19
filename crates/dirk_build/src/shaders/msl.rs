//! Translation of SPIR-V entry points to the Metal Shading Language, for
//! Apple targets, which `dirk_rhi` runs on Metal. SPIRV-Cross translates them
//! on Apple hosts only, so other hosts never compile it.
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

#[cfg(target_vendor = "apple")]
pub(super) use apple::translate;

/// Without SPIRV-Cross, other hosts cannot build shaders for Apple targets.
#[cfg(not(target_vendor = "apple"))]
pub(super) fn translate(entry_point: &str, _: &[u8], _: super::Stage) -> anyhow::Result<String> {
    anyhow::bail!("translating `{entry_point}` for an Apple target needs an Apple host")
}

#[cfg(target_vendor = "apple")]
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
    fn lay_out_heap(msl: &str) -> anyhow::Result<String> {
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

    #[cfg(test)]
    mod tests {
        use super::lay_out_heap;

        #[test]
        fn every_shader_declares_the_whole_heap_in_binding_order() {
            let msl = "struct spvDescriptorSetBuffer0\n{\n    \
                       const device uint* m_4 [[id(132096)]][65536];\n    \
                       constant uint* spvBufferSizeConstants [[id(197632)]];\n    \
                       array<sampler, 1024> m_5 [[id(131072)]];\n};\nvoid f();\n";
            let laid_out = lay_out_heap(msl).expect("the heap is laid out");
            let ids: Vec<_> = laid_out
                .lines()
                .filter_map(|line| line.split_once("[[id(")?.1.split_once(')'))
                .map(|(id, _)| id)
                .collect();
            assert_eq!(
                ids,
                [
                    "0", "16384", "32768", "49152", "65536", "81920", "98304", "114688", "131072",
                    "132096", "197632"
                ]
            );
            assert!(
                laid_out.contains("m_5 [[id(131072)]]") && laid_out.ends_with("};\nvoid f();\n")
            );
            assert_eq!(
                lay_out_heap("void f();").expect("nothing to lay out"),
                "void f();"
            );
        }

        #[test]
        fn arrays_read_as_several_types_are_declared_once() {
            let aliased = "struct spvDescriptorSetBuffer0\n{\n    \
                           const device uint* m_4 [[id(132096)]][65536];\n    \
                           // Overlapping binding: const device float4* m_6 [[id(132096)]][65536];\n\
                           };\n";
            let laid_out = lay_out_heap(aliased).expect("the heap is laid out");
            assert!(laid_out.contains("m_4 [[id(132096)]]") && !laid_out.contains("m_6"));
            let clashing = "struct spvDescriptorSetBuffer0\n{\n    \
                            array<sampler, 1024> m_5 [[id(131072)]];\n    \
                            array<sampler, 1024> m_6 [[id(131072)]];\n};\n";
            assert!(
                lay_out_heap(clashing).is_err(),
                "members sharing an id fail"
            );
        }
    }
}
