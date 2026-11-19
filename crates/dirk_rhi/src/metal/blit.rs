//! Scaling blits. Metal's blit encoders copy without scaling, so blits draw
//! the source region into the destination's, one layer or depth slice at a
//! time, sampling a view of the source's mip and layer.

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLFunction, MTLLibrary, MTLLoadAction,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResourceUsage,
    MTLSamplerState, MTLStoreAction, MTLTexture, MTLTextureType, MTLViewport,
};
use parking_lot::Mutex;

use super::{Check, CommandBuffer, Device, Image, Object, Sampler};
use crate::{Filter, Format, ImageCopy, Result, SamplerDesc};

/// Draws a triangle covering the viewport, and samples the source region
/// it maps to: `origin` and `size` in normalized coordinates of the source
/// mip, and the depth `z` of 3D sources.
const SHADERS: &str = r"
#include <metal_stdlib>
using namespace metal;

struct Region { float2 origin; float2 size; float z; };
struct Varyings { float4 position [[position]]; float2 t; };

vertex Varyings blit_vs(uint vertex_id [[vertex_id]]) {
    float2 t = float2((vertex_id << 1) & 2, vertex_id & 2);
    return { float4(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0), t };
}

fragment float4 blit_2d(Varyings in [[stage_in]], texture2d<float> source [[texture(0)]],
                        sampler filter [[sampler(0)]], constant Region& region [[buffer(0)]]) {
    return source.sample(filter, region.origin + in.t * region.size, level(0));
}

fragment float4 blit_3d(Varyings in [[stage_in]], texture3d<float> source [[texture(0)]],
                        sampler filter [[sampler(0)]], constant Region& region [[buffer(0)]]) {
    return source.sample(filter, float3(region.origin + in.t * region.size, region.z), level(0));
}
";

/// The source region of a blit, as the shaders read it.
#[repr(C)]
struct Region {
    origin: [f32; 2],
    size: [f32; 2],
    z: f32,
    _padding: f32,
}

/// What blits draw with, created on the first blit.
struct State {
    /// Pipelines by format and source dimension: 2D, then 3D.
    pipelines: [[Option<Object<dyn MTLRenderPipelineState>>; 2]; Format::COUNT],
    vertex: Object<dyn MTLFunction>,
    fragments: [Object<dyn MTLFunction>; 2],
    /// Nearest and linear samplers.
    samplers: [Object<dyn MTLSamplerState>; 2],
}

/// Pipelines and samplers for blits, created when first needed and kept
/// for the device's lifetime.
#[derive(Default)]
pub(super) struct Blitter {
    state: Mutex<Option<State>>,
}

impl Blitter {
    /// The pipeline blitting into `format` from 2D or 3D sources, and the
    /// sampler filtering with `filter`.
    fn prepare(
        &self,
        device: &Device,
        format: Format,
        three_d: bool,
        filter: Filter,
    ) -> Result<(
        Object<dyn MTLRenderPipelineState>,
        Object<dyn MTLSamplerState>,
    )> {
        let mut state = self.state.lock();
        let state = match &mut *state {
            Some(state) => state,
            empty @ None => empty.insert(State::new(device)?),
        };
        let sampler = state.samplers[filter as usize].clone();
        let pipeline = &mut state.pipelines[format as usize][usize::from(three_d)];
        if let Some(pipeline) = pipeline {
            return Ok((pipeline.clone(), sampler));
        }
        let desc = MTLRenderPipelineDescriptor::new();
        desc.setLabel(Some(&NSString::from_str("blit")));
        desc.setVertexFunction(Some(&state.vertex));
        desc.setFragmentFunction(Some(&state.fragments[usize::from(three_d)]));
        // SAFETY: every pipeline has a first color attachment.
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }.setPixelFormat(format.mtl());
        let created: Object<_> = device
            .raw
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .check("creating a blit pipeline")?
            .into();
        Ok((pipeline.insert(created).clone(), sampler))
    }
}

impl State {
    fn new(device: &Device) -> Result<Self> {
        let library = device.library(SHADERS)?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .check("newFunctionWithName")
                .map(Object::from)
        };
        let sampler = |filter: Filter| {
            Sampler::create(
                &device.raw,
                &SamplerDesc {
                    label: "blit",
                    mag_filter: filter,
                    min_filter: filter,
                    mipmap_filter: filter,
                    ..SamplerDesc::default()
                },
            )
        };
        Ok(Self {
            pipelines: Default::default(),
            vertex: function("blit_vs")?,
            fragments: [function("blit_2d")?, function("blit_3d")?],
            samplers: [sampler(Filter::Nearest)?, sampler(Filter::Linear)?],
        })
    }
}

impl CommandBuffer {
    /// Blits `regions` of `src` into `dst`, both of `format`, drawing each
    /// destination layer or depth slice in its own pass.
    #[expect(
        clippy::cast_precision_loss,
        reason = "image extents are far below f32's exact integer range"
    )]
    pub(crate) fn blit_image(
        &mut self,
        device: &Device,
        src: &Image,
        dst: &Image,
        format: Format,
        regions: &[ImageCopy],
        filter: Filter,
    ) -> Result<()> {
        let three_d = src.raw.textureType() == MTLTextureType::Type3D;
        let (pipeline, sampler) = device.blitter.prepare(device, format, three_d, filter)?;
        for region in regions {
            let (source, destination) = (region.src, region.dst);
            let mip = |size: usize| (size >> source.mip_level).max(1) as f32;
            let size = (
                mip(src.raw.width()),
                mip(src.raw.height()),
                mip(src.raw.depth()),
            );
            for layer in 0..source.layer_count {
                let view = src.view(source.mip_level, source.base_layer + layer, three_d)?;
                for z in 0..destination.extent.depth {
                    // Sample the source at the middle of this slice's share
                    // of its depth.
                    let depth = (z as f32 + 0.5) / destination.extent.depth as f32;
                    let region = Region {
                        origin: [
                            source.origin.x as f32 / size.0,
                            source.origin.y as f32 / size.1,
                        ],
                        size: [
                            source.extent.width as f32 / size.0,
                            source.extent.height as f32 / size.1,
                        ],
                        z: (source.origin.z as f32 + depth * source.extent.depth as f32) / size.2,
                        _padding: 0.0,
                    };
                    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
                    // SAFETY: every pass has a first color attachment.
                    let target = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
                    target.setTexture(Some(&dst.raw));
                    target.setLevel(destination.mip_level as usize);
                    if three_d {
                        target.setDepthPlane((destination.origin.z + z) as usize);
                    } else {
                        target.setSlice((destination.base_layer + layer) as usize);
                    }
                    // Keep the texels outside the region.
                    target.setLoadAction(MTLLoadAction::Load);
                    target.setStoreAction(MTLStoreAction::Store);
                    let encoder = self.begin_render(|commands| {
                        commands.renderCommandEncoderWithDescriptor(&pass)
                    });
                    encoder.setLabel(Some(&NSString::from_str("blit")));
                    encoder.setRenderPipelineState(&pipeline);
                    // Views are tracked apart from their texture, so the
                    // texture itself is declared for Metal to order this
                    // read after earlier writes.
                    encoder.useResource_usage_stages(
                        ProtocolObject::from_ref(&*src.raw),
                        MTLResourceUsage::Read,
                        MTLRenderStages::Fragment,
                    );
                    // Primitives are clipped to the viewport, the region.
                    encoder.setViewport(MTLViewport {
                        originX: f64::from(destination.origin.x),
                        originY: f64::from(destination.origin.y),
                        width: f64::from(destination.extent.width),
                        height: f64::from(destination.extent.height),
                        znear: 0.0,
                        zfar: 1.0,
                    });
                    // SAFETY: the view, sampler and region outlive the
                    // calls, and Metal retains the view.
                    unsafe {
                        encoder.setFragmentTexture_atIndex(Some(&view), 0);
                        encoder.setFragmentSamplerState_atIndex(Some(&sampler), 0);
                        encoder.setFragmentBytes_length_atIndex(
                            NonNull::from(&region).cast(),
                            size_of::<Region>(),
                            0,
                        );
                        encoder.drawPrimitives_vertexStart_vertexCount(
                            MTLPrimitiveType::Triangle,
                            0,
                            3,
                        );
                    }
                }
            }
        }
        self.end_encoding();
        Ok(())
    }
}

impl Image {
    /// A view of mip `level` of layer `layer`: 2D, or the whole mip of a 3D
    /// image.
    fn view(&self, level: u32, layer: u32, three_d: bool) -> Result<Object<dyn MTLTexture>> {
        let (texture_type, slice) = if three_d {
            (MTLTextureType::Type3D, 0)
        } else {
            (MTLTextureType::Type2D, layer as usize)
        };
        // SAFETY: the level and layer are inside the texture, and a 2D view
        // of one slice is valid for every texture type but 3D.
        unsafe {
            self.raw
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    self.raw.pixelFormat(),
                    texture_type,
                    NSRange::new(level as usize, 1),
                    NSRange::new(slice, 1),
                )
        }
        .check("newTextureView")
        .map(Object::from)
    }
}
