use std::time::Duration;

use dirk_rhi::{Extent3d, Format, ImageDimension, SampleCount};
use dirk_rhi_test_shaders::support::with_rhi;

use super::*;

/// Records on the copy queue even without a dedicated one, which runs it
/// on the graphics queue's family, so every device covers that path.
#[test]
fn failed_image_uploads_on_the_copy_queue_leave_the_batch_to_submit() -> Result<()> {
    with_rhi(|rhi| {
        let mut frame = rhi.begin_frame()?;
        let mut uploads = Uploads::new(&mut frame);
        uploads.recording = Some(Recording::Transfer {
            copies: uploads.frame.encoder("uploads")?,
            acquires: uploads.frame.encoder("upload acquisitions")?,
        });
        uploads.buffer(
            "before",
            BufferUsage::COPY_SRC,
            BufferAccess::CopySource,
            &[1; 16],
        )?;
        let depth = ImageDesc {
            label: "depth",
            dimension: ImageDimension::D2,
            extent: Extent3d::new_2d(4, 4),
            format: Format::Depth32Float,
            usage: ImageUsage::SAMPLED,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::X1,
            rest: None,
        };
        assert!(matches!(
            uploads.image(&depth, &[&[0; 64]]),
            Err(Error::Invalid { .. })
        ));
        let done = uploads.submit()?.expect("the buffer upload was recorded");
        frame.finish();
        rhi.wait(done, Duration::from_secs(10))
    })
}
