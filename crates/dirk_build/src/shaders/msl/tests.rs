use super::apple::lay_out_heap;

#[test]
fn every_shader_declares_the_whole_heap_in_binding_order() {
    let msl = "struct spvDescriptorSetBuffer0\n{\n    \
               const device uint* m_4 [[id(132096)]][65536];\n    \
               constant uint* spvBufferSizeConstants [[id(263168)]];\n    \
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
            "132096", "197632", "214016", "230400", "246784", "263168"
        ]
    );
    assert!(laid_out.contains("m_5 [[id(131072)]]") && laid_out.ends_with("};\nvoid f();\n"));
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
