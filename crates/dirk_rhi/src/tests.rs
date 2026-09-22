use crate::*;

fn capabilities() -> Capabilities {
    Capabilities {
        limits: crate::Limits::default(),
        depth_bias_clamp: false,
        max_sampler_anisotropy: 1,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 16,
        buffer_copy_offset_alignment: 512,
        buffer_copy_row_pitch_alignment: 256,
        dedicated_compute_queue: false,
        dedicated_copy_queue: false,
    }
}

#[test]
fn padded_upload_rows_preserve_pixels_and_zero_padding() -> Result<()> {
    let caps = capabilities();
    let layout = crate::UploadLayout::new(3, 2, TextureFormat::Rgba8Unorm, caps)?;
    let pixels: Vec<u8> = (0..24).collect();
    let padded = layout.pack(&pixels)?;
    assert_eq!(padded.len(), 512);
    assert_eq!(&padded[..12], &pixels[..12]);
    assert_eq!(&padded[256..268], &pixels[12..]);
    assert!(
        padded[12..256]
            .iter()
            .chain(padded[268..].iter())
            .all(|byte| *byte == 0)
    );
    assert!(layout.pack(&pixels[..23]).is_err());
    Ok(())
}

#[test]
fn shader_binding_maps_agree_for_sparse_stage_specific_layouts() -> Result<()> {
    use crate::{BindGroupLayoutEntry as E, BindingType as T, ShaderStage, ShaderStages as S};
    let vertex_only = E {
        binding: 7,
        ty: T::UniformBuffer {
            dynamic_offset: false,
        },
        visibility: S::VERTEX,
    };
    let fragment_only = E {
        binding: 13,
        ty: T::SampledImage,
        visibility: S::FRAGMENT,
    };
    let shared = E {
        binding: 42,
        ty: T::StorageBuffer {
            read_only: true,
            dynamic_offset: false,
        },
        visibility: S::VERTEX | S::FRAGMENT,
    };
    let merged = [vertex_only, fragment_only, shared];
    let fragment = [fragment_only, shared];
    let pipeline = crate::BindingMap::new(&[&[], &merged], ShaderStage::Fragment)?;
    let shader = crate::BindingMap::new(&[&[], &fragment], ShaderStage::Fragment)?;
    assert_eq!(pipeline.get(1, 13), shader.get(1, 13));
    assert_eq!(pipeline.get(1, 42), shader.get(1, 42));
    assert_eq!(pipeline.get(1, 7), None);
    assert_eq!(pipeline.get(1, 42).and_then(|s| s.buffer), Some(0));
    Ok(())
}
