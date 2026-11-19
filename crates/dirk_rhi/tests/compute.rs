//! Compute pipelines and dispatches writing storage buffers and images
//! through the bindless heap.

mod support;

use std::time::Duration;

use dirk_rhi::{
    Buffer, BufferAccess, BufferBarrier, BufferDesc, BufferImageCopy, BufferUsage, ComputePipeline,
    ComputePipelineDesc, Error, Extent3d, Format, Graphics, ImageAccess, ImageBarrier, ImageDesc,
    ImageDimension, ImageRange, ImageUsage, ImageView, Memory, PushData, Result, Rhi, SampleCount,
    Shader, ShaderStages,
};
use dirk_rhi_test_shaders::{Count, CountCs, FillVs, Paint, PaintCs};
use support::with_rhi;

/// A compute pipeline running shader `S`.
fn compute_pipeline<S: Shader, P: PushData>(rhi: &Rhi) -> Result<ComputePipeline<P>> {
    rhi.create_compute_pipeline(&ComputePipelineDesc {
        label: S::ENTRY_POINT,
        shader: &rhi.create_shader::<S, P>()?,
    })
}

/// A readback buffer of `size` bytes, written by `usage`.
fn readback(rhi: &Rhi, size: u64, usage: BufferUsage) -> Result<Buffer> {
    rhi.create_buffer(&BufferDesc {
        label: "readback",
        size,
        usage,
        memory: Memory::Readback,
    })
}

#[test]
fn dispatches_write_storage_buffers_and_images() -> Result<()> {
    with_rhi(|rhi| {
        let count = compute_pipeline::<CountCs, Count>(rhi)?;
        assert_eq!(count.workgroup_size(), [64, 1, 1]);
        let paint = compute_pipeline::<PaintCs, Paint>(rhi)?;
        // 200 elements, in four workgroups of 64.
        let numbers = readback(rhi, 800, BufferUsage::STORAGE)?;
        let image = rhi.create_image(&ImageDesc {
            label: "painted",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(8, 8),
            format: Format::Rgba8Unorm,
            usage: ImageUsage::STORAGE | ImageUsage::COPY_SRC,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
        })?;
        let texels = readback(rhi, 256, BufferUsage::COPY_DST)?;

        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("compute")?;
        let written = |buffer| BufferBarrier {
            buffer,
            before: BufferAccess::ShaderWrite(ShaderStages::COMPUTE),
            after: BufferAccess::HostRead,
            transfer: None,
        };
        let image_barrier = |before, after| ImageBarrier {
            image: &image,
            range: ImageRange::ALL,
            before,
            after,
            transfer: None,
        };
        let storage = ImageAccess::StorageWrite(ShaderStages::COMPUTE);
        // SAFETY: the image is new.
        unsafe { encoder.barrier(&[], &[image_barrier(ImageAccess::Undefined, storage)])? };
        let push = Count {
            output: numbers.index().expect("storage buffers have an index"),
            base: 7,
        };
        encoder.dispatch(&count, &push, [4, 1, 1])?;
        let push = Paint {
            image: image
                .view()
                .and_then(ImageView::storage_index)
                .expect("storage images have an index"),
            color: [0.0, 1.0, 0.0, 1.0],
        };
        encoder.dispatch(&paint, &push, [1, 1, 1])?;
        // SAFETY: the dispatches wrote the image last.
        unsafe { encoder.barrier(&[], &[image_barrier(storage, ImageAccess::CopySource)])? };
        encoder.copy_image_to_buffer(
            &image,
            &texels,
            &[BufferImageCopy::packed(
                Format::Rgba8Unorm,
                0,
                image.info().mip_region(0),
            )],
        )?;
        let copied = BufferBarrier {
            before: BufferAccess::CopyDestination,
            ..written(&texels)
        };
        // SAFETY: the dispatch and copy wrote these buffers last.
        unsafe { encoder.barrier(&[written(&numbers), copied], &[])? };
        let done = frame.submit(encoder)?;
        frame.finish();
        rhi.wait(done, Duration::from_secs(10))?;

        let mut values = [0_u32; 200];
        let mut pixels = [[0_u8; 4]; 64];
        // SAFETY: the work writing both buffers completed, followed by
        // barriers to host reads.
        unsafe {
            numbers.read(0, bytemuck::cast_slice_mut(&mut values))?;
            texels.read(0, pixels.as_flattened_mut())?;
        }
        assert!(
            values
                .iter()
                .zip(7..)
                .all(|(&value, expected)| value == expected)
        );
        assert!(pixels.iter().all(|&pixel| pixel == [0, 255, 0, 255]));
        Ok(())
    })
}

#[test]
fn compute_pipelines_are_validated() -> Result<()> {
    with_rhi(|rhi| {
        let vertex = rhi.create_shader::<FillVs, dirk_rhi_test_shaders::Fill>()?;
        match rhi.create_compute_pipeline(&ComputePipelineDesc {
            label: "vertex",
            shader: &vertex,
        }) {
            Err(Error::Invalid { reason, .. }) => {
                assert!(reason.contains("not a compute shader"), "{reason}");
            }
            _ => panic!("a vertex shader made a compute pipeline"),
        }
        let count = compute_pipeline::<CountCs, Count>(rhi)?;
        let numbers = readback(rhi, 4, BufferUsage::STORAGE)?;
        let push = Count {
            output: numbers.index().expect("storage buffers have an index"),
            base: 0,
        };
        // Some devices dispatch as many workgroups as a `u32` counts.
        let Some(too_many) = rhi.capabilities().limits.max_compute_workgroups[0].checked_add(1)
        else {
            return Ok(());
        };
        let frame = rhi.begin_frame()?;
        let mut encoder = frame.encoder::<Graphics>("too many workgroups")?;
        match encoder.dispatch(&count, &push, [too_many, 1, 1]) {
            Err(Error::Invalid { reason, .. }) => assert!(reason.contains("exceed"), "{reason}"),
            _ => panic!("a dispatch beyond the device's limit was accepted"),
        }
        Ok(())
    })
}
