use ash::vk;

use crate::{
    Result,
    resources::{command_pool::CommandBuffer, image::Image},
};

impl Image {
    pub fn mip_levels(width: u32, height: u32) -> u32 {
        // How many times can we halve the larger dimension before hitting 1px?
        width.max(height).bit_width()
    }
    pub fn generate_mipmaps(
        &self,
        cmd: &CommandBuffer,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<()> {
        // heigh & width never get anywhere near i32::MAX, so no real problem
        #[allow(clippy::cast_possible_wrap)]
        let mut mip_width = width as i32;
        #[allow(clippy::cast_possible_wrap)]
        let mut mip_height = height as i32;

        for level in 1..mip_levels {
            let base_mip = level - 1;
            // Transition previous level: TRANSFER_DST → TRANSFER_SRC
            self.transition_image_layout(
                cmd,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                1,
                base_mip,
            )?;

            let next_w = (mip_width / 2).max(1);
            let next_h = (mip_height / 2).max(1);

            let blit = vk::ImageBlit::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_w,
                        y: next_h,
                        z: 1,
                    },
                ]);

            cmd.blit_image(
                self.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );

            // Previous level is fully consumed — transition to shader-readable
            self.transition_image_layout(
                cmd,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                1,
                base_mip,
            )?;

            mip_width = next_w;
            mip_height = next_h;
        }

        // Transition the final mip level (never used as a blit source)
        self.transition_image_layout(
            cmd,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
            mip_levels - 1,
        )
    }
}
