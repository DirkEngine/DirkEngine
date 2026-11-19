//! Barrier derivation, checked on the CPU without executing graphs.

use dirk_rhi::{
    Buffer, BufferAccess, BufferDesc, BufferUsage, Error, Extent3d, Format, Image, ImageAccess,
    ImageDesc, ImageDimension, ImageRange, ImageUsage, LoadOp, Memory, SampleCount, ShaderStages,
    StoreOp, Subresource,
};

use super::{Handle, RenderGraph};

const FRAGMENT: ShaderStages = ShaderStages::FRAGMENT;
const VERTEX: ShaderStages = ShaderStages::VERTEX;
const CLEAR: LoadOp<[f32; 4]> = LoadOp::Clear([0.0; 4]);

/// A 64x64 color image whose usage the graph derives.
fn color_desc() -> ImageDesc<'static> {
    ImageDesc {
        label: "color",
        dimension: ImageDimension::D2,
        extent: Extent3d::new_2d(64, 64),
        format: Format::Rgba8Unorm,
        usage: ImageUsage::empty(),
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::X1,
    }
}

/// Adds a transient color image.
fn color(graph: &mut RenderGraph<'static>) -> Handle<Image> {
    graph.create_image(&color_desc())
}

/// Adds a transient 64-byte buffer.
fn buffer(graph: &mut RenderGraph<'static>) -> Handle<Buffer> {
    graph.create_buffer(&BufferDesc {
        label: "buffer",
        size: 64,
        usage: BufferUsage::empty(),
        memory: Memory::Device,
    })
}

/// Each image barrier's pass and accesses.
fn image_barriers(graph: &RenderGraph<'_>) -> Vec<(usize, ImageAccess, ImageAccess)> {
    graph
        .images
        .barriers
        .iter()
        .map(|barrier| (barrier.pass, barrier.before, barrier.after))
        .collect()
}

/// Adds a pass rendering to `image`.
fn render(graph: &mut RenderGraph<'_>, image: Handle<Image>) {
    graph
        .pass("render")
        .color(image, CLEAR)
        .record(|_| Ok(()))
        .expect("rendering is valid");
}

/// The reason a pass was rejected.
fn rejection(result: dirk_rhi::Result<()>) -> String {
    match result {
        Err(Error::Invalid { reason, .. }) => reason,
        other => panic!("expected a rejected pass, got {other:?}"),
    }
}

#[test]
fn attachment_then_sampled_then_copied_transitions_each_pass() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    render(&mut graph, image);
    graph
        .pass("sample")
        .sampled(image, FRAGMENT)
        .record(|_| Ok(()))
        .expect("sampling what was rendered is valid");
    graph
        .pass("copy")
        .image(image, ImageAccess::CopySource)
        .record(|_| Ok(()))
        .expect("copying what was rendered is valid");
    assert_eq!(
        image_barriers(&graph),
        [
            (0, ImageAccess::Undefined, ImageAccess::ColorAttachment),
            (
                1,
                ImageAccess::ColorAttachment,
                ImageAccess::Sampled(FRAGMENT)
            ),
            (2, ImageAccess::Sampled(FRAGMENT), ImageAccess::CopySource),
        ]
    );
    assert_eq!(
        graph.images.list[image.index].usage,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::COPY_SRC
    );
}

#[test]
fn repeated_writes_keep_a_dependency() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    render(&mut graph, image);
    render(&mut graph, image);
    assert_eq!(
        image_barriers(&graph)[1],
        (
            1,
            ImageAccess::ColorAttachment,
            ImageAccess::ColorAttachment
        )
    );
}

#[test]
fn later_reads_widen_the_barrier_before_the_first() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    render(&mut graph, image);
    for stages in [FRAGMENT, VERTEX, FRAGMENT] {
        graph
            .pass("sample")
            .sampled(image, stages)
            .record(|_| Ok(()))
            .expect("sampling is valid");
    }
    render(&mut graph, image);
    let both = ImageAccess::Sampled(FRAGMENT | VERTEX);
    assert_eq!(
        image_barriers(&graph),
        [
            (0, ImageAccess::Undefined, ImageAccess::ColorAttachment),
            (1, ImageAccess::ColorAttachment, both),
            (4, both, ImageAccess::ColorAttachment),
        ]
    );
}

#[test]
fn reads_in_one_pass_merge_into_one_barrier() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    render(&mut graph, image);
    graph
        .pass("sample twice")
        .sampled(image, VERTEX)
        .sampled(image, FRAGMENT)
        .record(|_| Ok(()))
        .expect("sampling in two stages is valid");
    assert_eq!(
        image_barriers(&graph)[1],
        (
            1,
            ImageAccess::ColorAttachment,
            ImageAccess::Sampled(VERTEX | FRAGMENT)
        )
    );
}

#[test]
fn conflicting_accesses_in_one_pass_are_rejected() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    render(&mut graph, image);
    let reason = rejection(
        graph
            .pass("sample and copy")
            .sampled(image, FRAGMENT)
            .image(image, ImageAccess::CopySource)
            .record(|_| Ok(())),
    );
    assert!(reason.contains("accessed both as"), "{reason}");
    let reason = rejection(
        graph
            .pass("feedback")
            .sampled(image, FRAGMENT)
            .color(image, LoadOp::Load)
            .record(|_| Ok(())),
    );
    assert!(reason.contains("accessed both as"), "{reason}");
    // Rejected passes leave the graph unchanged.
    assert_eq!(graph.passes.len(), 1);
    assert_eq!(image_barriers(&graph).len(), 1);
}

#[test]
fn first_reads_of_undefined_contents_are_rejected() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    let contents = buffer(&mut graph);
    let read = "is read before anything writes it";
    let sampled = graph
        .pass("sample")
        .sampled(image, FRAGMENT)
        .record(|_| Ok(()));
    assert!(rejection(sampled).contains(read));
    let loaded = graph
        .pass("load")
        .color(image, LoadOp::Load)
        .record(|_| Ok(()));
    assert!(rejection(loaded).contains(read));
    let indices = graph
        .pass("draw")
        .buffer(contents, BufferAccess::Index)
        .record(|_| Ok(()));
    assert!(rejection(indices).contains(read));
}

#[test]
fn invalid_declarations_are_rejected() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    let contents = buffer(&mut graph);
    let stageless = graph
        .pass("stageless")
        .buffer(contents, BufferAccess::ShaderWrite(ShaderStages::empty()))
        .record(|_| Ok(()));
    assert!(rejection(stageless).contains("names no shader stage"));
    let attachment = graph
        .pass("attachment")
        .image(image, ImageAccess::ColorAttachment)
        .record(|_| Ok(()));
    assert!(rejection(attachment).contains("without an attachment"));
    let host = graph
        .pass("host")
        .buffer(contents, BufferAccess::HostRead)
        .record(|_| Ok(()));
    assert!(rejection(host).contains("passes do not declare"));
    let other = RenderGraph::new().create_image(&ImageDesc {
        label: "elsewhere",
        ..color_desc()
    });
    let foreign = graph
        .pass("foreign")
        .sampled(other, FRAGMENT)
        .record(|_| Ok(()));
    assert!(rejection(foreign).contains("belongs to another graph"));
}

#[test]
fn buffers_need_no_barrier_before_their_first_write() {
    let mut graph = RenderGraph::new();
    let contents = buffer(&mut graph);
    graph
        .pass("fill")
        .buffer(contents, BufferAccess::CopyDestination)
        .record(|_| Ok(()))
        .expect("filling a buffer is valid");
    graph
        .pass("read")
        .buffer(contents, BufferAccess::ShaderRead(VERTEX))
        .record(|_| Ok(()))
        .expect("reading what was filled is valid");
    let barriers: Vec<_> = graph
        .buffers
        .barriers
        .iter()
        .map(|barrier| (barrier.pass, barrier.before, barrier.after))
        .collect();
    assert_eq!(
        barriers,
        [(
            1,
            BufferAccess::CopyDestination,
            BufferAccess::ShaderRead(VERTEX)
        )]
    );
}

#[test]
fn transients_store_attachments_only_for_later_passes() {
    let mut graph = RenderGraph::new();
    let image = color(&mut graph);
    let unused = color(&mut graph);
    render(&mut graph, image);
    render(&mut graph, image);
    assert_eq!(graph.images.store(image.index, 0), StoreOp::Store);
    assert_eq!(graph.images.store(image.index, 1), StoreOp::Discard);
    // Transients no pass declares are never created.
    assert!(graph.images.list[unused.index].usage.is_empty());
}

/// Mip `mip_level` of layer `layer`.
const fn at(mip_level: u32, layer: u32) -> Subresource {
    Subresource { mip_level, layer }
}

#[test]
fn passes_render_one_mip_while_sampling_the_previous() {
    let mut graph = RenderGraph::new();
    let image = graph.create_image(&ImageDesc {
        mip_levels: 3,
        ..color_desc()
    });
    graph
        .pass("mip 0")
        .color(image.subresource(at(0, 0)), CLEAR)
        .record(|_| Ok(()))
        .expect("rendering one mip is valid");
    for mip in 1..3 {
        graph
            .pass("next mip")
            .sampled(image.slice(ImageRange::mip(mip - 1)), FRAGMENT)
            .color(image.subresource(at(mip, 0)), CLEAR)
            .record(|_| Ok(()))
            .expect("sampling one mip while rendering the next is valid");
    }
    let barriers: Vec<_> = graph
        .images
        .barriers
        .iter()
        .map(|barrier| {
            (
                barrier.pass,
                barrier.subresources.mips.clone(),
                barrier.after,
            )
        })
        .collect();
    let sampled = ImageAccess::Sampled(FRAGMENT);
    let color = ImageAccess::ColorAttachment;
    assert_eq!(
        barriers,
        [
            (0, 0..1, color),
            (1, 0..1, sampled),
            (1, 1..2, color),
            (2, 1..2, sampled),
            (2, 2..3, color),
        ]
    );
}

#[test]
fn slices_are_validated() {
    let mut graph = RenderGraph::new();
    let image = graph.create_image(&ImageDesc {
        mip_levels: 2,
        array_layers: 6,
        dimension: ImageDimension::Cube,
        ..color_desc()
    });
    let reason = rejection(
        graph
            .pass("whole cube")
            .color(image, CLEAR)
            .record(|_| Ok(())),
    );
    assert!(reason.contains("not one mip of one layer"), "{reason}");
    let reason = rejection(
        graph
            .pass("outside")
            .image(image.slice(ImageRange::layer(6)), ImageAccess::CopySource)
            .record(|_| Ok(())),
    );
    assert!(reason.contains("no subresources"), "{reason}");
    let reason = rejection(
        graph
            .pass("overlapping")
            .image(image.slice(ImageRange::layer(2)), ImageAccess::CopySource)
            .image(
                image.slice(ImageRange::mip(0)),
                ImageAccess::CopyDestination,
            )
            .record(|_| Ok(())),
    );
    assert!(reason.contains("overlapping"), "{reason}");
    graph
        .pass("faces")
        .color(image.subresource(at(0, 2)), CLEAR)
        .image(
            image.slice(ImageRange::layer(3)),
            ImageAccess::CopyDestination,
        )
        .record(|_| Ok(()))
        .expect("rendering one face while copying into another is valid");
}
