//! Batched uploads with an explicit transfer-to-graphics handoff.
use dirk_rhi::{
    Buffer, BufferCopy, BufferDesc, BufferImageCopy, BufferUsages, CommandEncoder, Completion,
    CopyQueue, DependencyInfo, Graphics, Image, ImageBarrier, ImageState, ImageSubresourceRange,
    InvalidResourceKind, MemoryDomain, Origin3d, QueueTransfer, QueueType, RecordedCommands,
    ResourceAccess, Result, Rhi, ShaderStages, SubmitInfo, UploadLayout,
};

/// One tick's uploads. Native copies record immediately; staging drops into the current cycle.
#[derive(Default)]
pub struct UploadBatch {
    encoders: Option<UploadEncoders>,
}

struct UploadEncoders {
    transfer: CommandEncoder<CopyQueue>,
    acquire: CommandEncoder<Graphics>,
}
impl UploadBatch {
    /// Starts a batch independently of renderer state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn encoders(&mut self, rhi: &Rhi) -> Result<&mut UploadEncoders> {
        let encoders = match self.encoders.take() {
            Some(encoders) => encoders,
            None => UploadEncoders {
                transfer: rhi.create_encoder("uploads")?,
                acquire: rhi.create_encoder("upload acquisitions")?,
            },
        };
        Ok(self.encoders.insert(encoders))
    }
    /// Allocates and uploads immutable buffer data for subsequent graphics use.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details,
    /// and rejects shader accesses naming no shader stages.
    pub fn buffer(
        &mut self,
        rhi: &Rhi,
        bytes: &[u8],
        usage: BufferUsages,
        access: ResourceAccess,
    ) -> Result<Buffer> {
        ensure_shader_stages(access)?;
        let mut staging = Self::staging(rhi, bytes)?;
        // SAFETY: this allocation has not been submitted and is exclusively borrowed.
        unsafe {
            staging.write(0, bytes)?;
        }
        let buffer = rhi.create_buffer(&BufferDesc {
            label: "uploaded buffer",
            size: bytes.len() as u64,
            usage: usage | BufferUsages::COPY_DST,
            memory: MemoryDomain::Device,
        })?;
        let encoders = self.encoders(rhi)?;
        // SAFETY: both ranges are fresh allocations, and both recordings submit in this cycle.
        unsafe {
            encoders.transfer.copy_buffer(
                &staging,
                &buffer,
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes.len() as u64,
                }],
            )?;
            let barrier = [dirk_rhi::BufferBarrier {
                buffer: &buffer,
                offset: 0,
                size: buffer.size(),
                old_state: ImageState::CopyDestination,
                new_state: access,
                queue_transfer: Some(Self::handoff()),
            }];
            let dependency = DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &barrier,
                image_barriers: &[],
            };
            encoders.transfer.barrier(&dependency)?;
            encoders.acquire.barrier(&dependency)?;
        }
        Ok(buffer)
    }
    /// Uploads every mip and array layer of a new color image and makes it
    /// readable by fragment shaders. See [`Self::image_for_stages`].
    ///
    /// # Safety
    /// The image must be uninitialized, not in GPU use, and remain alive through all subsequent uses.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn image(&mut self, rhi: &Rhi, image: &Image, levels: &[&[u8]]) -> Result<()> {
        unsafe { self.image_for_stages(rhi, image, levels, ShaderStages::FRAGMENT) }
    }

    /// Uploads every mip and array layer of a new color image and makes it
    /// readable by shaders in `stages`.
    ///
    /// `levels` holds one entry per mip. Each entry contains every array layer's
    /// tightly packed texels, layer after layer.
    ///
    /// # Safety
    /// The image must be uninitialized, not in GPU use, and remain alive through all subsequent uses.
    ///
    /// # Errors
    /// Rejects empty `stages`, volume images, a level count differing from the
    /// image's mips, or level sizes differing from all of its layers before
    /// recording anything. Returns allocation, interface validation, or native
    /// device errors with their details.
    pub unsafe fn image_for_stages(
        &mut self,
        rhi: &Rhi,
        image: &Image,
        levels: &[&[u8]],
        stages: ShaderStages,
    ) -> Result<()> {
        let final_state = ImageState::ShaderRead(stages);
        ensure_shader_stages(final_state)?;
        let info = image.description();
        validate_levels(&info, levels)?;
        let encoders = self.encoders(rhi)?;
        unsafe {
            Self::image_barrier(
                &mut encoders.transfer,
                image,
                ImageState::Undefined,
                ImageState::CopyDestination,
                None,
            )?;
        }
        for (mip, pixels) in levels.iter().enumerate() {
            let mip = u32::try_from(mip).map_err(|_| InvalidResourceKind::OutOfRange)?;
            let extent = info.mip_extent(mip)?;
            unsafe {
                ImageUpload {
                    image,
                    mip,
                    origin: Origin3d::default(),
                    extent,
                    pixels,
                }
                .record_layers(
                    rhi,
                    &mut encoders.transfer,
                    0,
                    info.array_layers,
                )?;
            }
        }
        unsafe {
            Self::image_barrier(
                &mut encoders.transfer,
                image,
                ImageState::CopyDestination,
                final_state,
                Some(Self::handoff()),
            )?;
            Self::image_barrier(
                &mut encoders.acquire,
                image,
                ImageState::CopyDestination,
                final_state,
                Some(Self::handoff()),
            )?;
        }
        Ok(())
    }
    /// Submits the transfer batch and returns graphics acquisitions and their GPU dependency.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn submit(self, rhi: &mut Rhi) -> Result<Option<(RecordedCommands, Completion)>> {
        let Some(encoders) = self.encoders else {
            return Ok(None);
        };
        let transfer = encoders.transfer.finish()?;
        let acquire = encoders.acquire.finish()?;
        // SAFETY: this batch records complete copy dependencies and submits once in its recording cycle.
        let completion = unsafe {
            rhi.queue::<CopyQueue>()
                .submit(vec![transfer], &SubmitInfo::default())?
        };
        Ok(Some((acquire, completion)))
    }
    /// Finishes initialization uploads, including ownership acquisition, before returning.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub fn finish(self, rhi: &mut Rhi) -> Result<()> {
        if let Some((acquire, transfer)) = self.submit(rhi)? {
            // SAFETY: acquisition waits for the matching, already-submitted release.
            unsafe {
                rhi.queue::<Graphics>().submit(
                    vec![acquire],
                    &SubmitInfo {
                        surface_frames: &[],
                        wait_for: &[&transfer],
                    },
                )?
            }
            .wait(u64::MAX)?;
        }
        Ok(())
    }
    fn staging(rhi: &Rhi, bytes: &[u8]) -> Result<Buffer> {
        rhi.create_buffer(&BufferDesc {
            label: "upload staging",
            size: bytes.len() as u64,
            usage: BufferUsages::COPY_SRC,
            memory: MemoryDomain::Upload,
        })
    }
    fn handoff() -> QueueTransfer {
        QueueTransfer {
            source: QueueType::Copy,
            destination: QueueType::Graphics,
        }
    }
    unsafe fn image_barrier<Q: dirk_rhi::QueueKind>(
        cmd: &mut CommandEncoder<Q>,
        image: &Image,
        old_state: ImageState,
        new_state: ImageState,
        queue_transfer: Option<QueueTransfer>,
    ) -> Result<()> {
        let range = ImageSubresourceRange::WHOLE.resolve(&image.description())?;
        unsafe {
            cmd.barrier(&DependencyInfo {
                memory_barriers: &[],
                buffer_barriers: &[],
                image_barriers: &[ImageBarrier {
                    image,
                    old_state,
                    new_state,
                    aspects: range.aspects,
                    base_mip_level: 0,
                    mip_level_count: range.mip_level_count,
                    base_array_layer: 0,
                    array_layer_count: range.array_layer_count,
                    queue_transfer,
                }],
            })
        }
    }
}

/// One packed color region; the caller supplies its explicit image dependencies.
///
/// [`Self::record`] writes array layer 0; [`Self::record_layers`] writes several
/// layers from consecutive tightly packed layer payloads.
pub struct ImageUpload<'a> {
    /// Destination allocation.
    pub image: &'a Image,
    /// Destination mip.
    pub mip: u32,
    /// Destination texel origin.
    pub origin: Origin3d,
    /// Region extent.
    pub extent: dirk_rhi::Extent3d,
    /// Tightly packed source texels, layer after layer when recording several layers.
    pub pixels: &'a [u8],
}
impl ImageUpload<'_> {
    /// Packs aligned staging memory and records a copy into layer 0 of an
    /// already transitioned image.
    ///
    /// # Safety
    /// The image must be in `CopyDestination` state, ordered against prior uses, and alive through GPU use.
    ///
    /// # Errors
    /// Returns allocation, interface validation, or native device errors with their details.
    pub unsafe fn record<Q: dirk_rhi::QueueKind>(
        &self,
        rhi: &Rhi,
        command: &mut CommandEncoder<Q>,
    ) -> Result<()> {
        unsafe { self.record_layers(rhi, command, 0, 1) }
    }

    /// Packs aligned staging memory and records one copy into `layer_count`
    /// consecutive layers starting at `base_layer`.
    ///
    /// # Safety
    /// The layers must be in `CopyDestination` state, ordered against prior uses, and alive through GPU use.
    ///
    /// # Errors
    /// Rejects pixel data that is not exactly `layer_count` tightly packed
    /// layers. Returns allocation, interface validation, or native device errors
    /// with their details.
    pub unsafe fn record_layers<Q: dirk_rhi::QueueKind>(
        &self,
        rhi: &Rhi,
        command: &mut CommandEncoder<Q>,
        base_layer: u32,
        layer_count: u32,
    ) -> Result<()> {
        let layout = UploadLayout::new(
            self.extent.width,
            self.extent.height,
            self.image.description().format,
            rhi.capabilities(),
        )?;
        let pixels = pack_layers(layout, self.pixels, layer_count)?;
        let mut staging = UploadBatch::staging(rhi, &pixels)?;
        unsafe {
            staging.write(0, &pixels)?;
            command.copy_buffer_to_image(
                &staging,
                self.image,
                &[BufferImageCopy {
                    buffer_offset: 0,
                    buffer_bytes_per_row: layout.bytes_per_row,
                    buffer_rows_per_image: layout.rows,
                    mip_level: self.mip,
                    base_array_layer: base_layer,
                    array_layer_count: layer_count,
                    image_origin: self.origin,
                    extent: self.extent,
                    aspects: dirk_rhi::ImageAspects::COLOR,
                }],
            )?;
        }
        Ok(())
    }
}

/// Checks one tightly packed payload per mip covering every array layer.
fn validate_levels(info: &dirk_rhi::ImageInfo, levels: &[&[u8]]) -> Result<()> {
    if info.dimension == dirk_rhi::ImageDimension::ThreeD {
        return Err(InvalidResourceKind::Mismatch
            .with_detail("volume image uploads are unsupported")
            .into());
    }
    if levels.len() != info.mip_levels as usize {
        return Err(InvalidResourceKind::Mismatch
            .with_detail("upload must provide exactly one level per image mip")
            .into());
    }
    for (mip, pixels) in (0..info.mip_levels).zip(levels) {
        let extent = info.mip_extent(mip)?;
        let expected = u64::from(extent.width)
            .checked_mul(u64::from(extent.height))
            .and_then(|n| n.checked_mul(u64::from(info.format.texel_size())))
            .and_then(|n| n.checked_mul(u64::from(info.array_layers)))
            .ok_or(InvalidResourceKind::OutOfRange)?;
        if u64::try_from(pixels.len()).ok() != Some(expected) {
            return Err(InvalidResourceKind::Mismatch
                .with_detail(format!(
                    "upload mip {mip} must hold {expected} bytes covering all {} layers",
                    info.array_layers
                ))
                .into());
        }
    }
    Ok(())
}

/// Packs `layer_count` consecutive tightly packed layers with aligned row pitches.
fn pack_layers(layout: UploadLayout, pixels: &[u8], layer_count: u32) -> Result<Vec<u8>> {
    let layer_bytes = usize::try_from(u64::from(layout.row_bytes) * u64::from(layout.rows.get()))
        .map_err(|_| InvalidResourceKind::OutOfRange)?;
    let layers = usize::try_from(layer_count).map_err(|_| InvalidResourceKind::OutOfRange)?;
    if layers == 0 || layer_bytes.checked_mul(layers) != Some(pixels.len()) {
        return Err(InvalidResourceKind::Mismatch
            .with_detail("upload pixels must hold every selected layer, tightly packed")
            .into());
    }
    let mut packed = Vec::new();
    for layer in pixels.chunks_exact(layer_bytes) {
        packed.extend(layout.pack(layer)?);
    }
    Ok(packed)
}

/// Rejects shader accesses whose empty stage set would synchronize nothing.
fn ensure_shader_stages(access: ResourceAccess) -> Result<()> {
    if crate::graph::names_shader_stages(access) {
        Ok(())
    } else {
        Err(InvalidResourceKind::Mismatch
            .with_detail("upload destination must name at least one shader stage")
            .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirk_rhi::{Extent3d, ImageDimension, ImageInfo, ImageUsages, SampleCount, TextureFormat};
    use std::num::NonZeroU32;

    fn layered(dimension: ImageDimension, array_layers: u32) -> ImageInfo {
        ImageInfo {
            dimension,
            extent: Extent3d::new_2d(4, 2),
            format: TextureFormat::Rgba8Unorm,
            usage: ImageUsages::COPY_DST | ImageUsages::SAMPLED,
            mip_levels: 2,
            array_layers,
            samples: SampleCount::One,
        }
    }

    #[test]
    fn levels_must_cover_every_layer_of_every_mip() {
        let info = layered(ImageDimension::TwoD, 3);
        let (mip0, mip1) = (vec![0; 4 * 2 * 4 * 3], vec![0; 2 * 4 * 3]);
        assert!(validate_levels(&info, &[&mip0, &mip1]).is_ok());
        let single_layer = vec![0; 4 * 2 * 4];
        assert!(validate_levels(&info, &[&single_layer, &mip1]).is_err());
        assert!(validate_levels(&info, &[&mip0]).is_err());
        let volume = layered(ImageDimension::ThreeD, 1);
        assert!(validate_levels(&volume, &[&single_layer, &mip1[..8]]).is_err());
    }

    #[test]
    fn layers_pack_consecutively_with_padded_rows() -> Result<()> {
        let layout = UploadLayout {
            row_bytes: 2,
            bytes_per_row: NonZeroU32::new(4).ok_or(InvalidResourceKind::Empty)?,
            rows: NonZeroU32::new(2).ok_or(InvalidResourceKind::Empty)?,
        };
        let packed = pack_layers(layout, &[1, 2, 3, 4, 5, 6, 7, 8], 2)?;
        assert_eq!(packed, [1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0]);
        assert!(pack_layers(layout, &[1, 2, 3, 4], 2).is_err());
        assert!(pack_layers(layout, &[], 0).is_err());
        Ok(())
    }

    #[test]
    fn upload_destinations_must_name_shader_stages() {
        assert!(ensure_shader_stages(ResourceAccess::Uniform(ShaderStages::NONE)).is_err());
        assert!(ensure_shader_stages(ResourceAccess::ShaderRead(ShaderStages::VERTEX)).is_ok());
        assert!(ensure_shader_stages(ResourceAccess::Index).is_ok());
    }
}
