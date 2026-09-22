//! CPU preparation of asset mipmaps.

/// CPU fallback for sRGB asset mipmaps when exact filtered blits are unavailable.
pub(super) struct RgbaMip {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
impl RgbaMip {
    // Dimensions are constrained by image allocation limits; conversion back to
    // bytes follows clamping to the representable range.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn next(&self) -> Self {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0_f32; 4];
                let mut count = 0;
                for sy in y * self.height / height..(y + 1) * self.height / height {
                    for sx in x * self.width / width..(x + 1) * self.width / width {
                        let offset = (sy as usize * self.width as usize + sx as usize) * 4;
                        for (channel, sum) in sum.iter_mut().enumerate() {
                            let value = f32::from(self.pixels[offset + channel]) / 255.0;
                            *sum += if channel == 3 {
                                value
                            } else {
                                Self::linear(value)
                            };
                        }
                        count += 1;
                    }
                }
                for (channel, sum) in sum.into_iter().enumerate() {
                    let value = sum / count as f32;
                    let value = if channel == 3 {
                        value
                    } else {
                        Self::srgb(value)
                    };
                    pixels.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
    fn linear(value: f32) -> f32 {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }
    fn srgb(value: f32) -> f32 {
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RgbaMip;
    #[test]
    fn odd_srgb_mips_include_edge_pixels_and_average_in_linear_light() {
        let mip = RgbaMip {
            width: 3,
            height: 1,
            pixels: vec![0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255],
        }
        .next();
        assert_eq!((mip.width, mip.height), (1, 1));
        assert_eq!(mip.pixels, [156, 156, 156, 255]);
    }
}
