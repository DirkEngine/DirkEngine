//! The bindless resource heap: one tier-2 argument buffer every pipeline
//! binds, laid out as the shader build declares it (see
//! `dirk_build::shaders`): every array back to back in binding order, eight
//! bytes per element, then a pointer to one `u32` byte length per element,
//! which shaders bound their storage buffer reads with.

use std::collections::VecDeque;

use dirk_shaders::bindless::heap::{self, CAPACITIES, first_element, metal::BUFFER_LENGTHS};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLResource, MTLResourceID, MTLResourceOptions,
    MTLSamplerState, MTLSize, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use parking_lot::Mutex;

use super::{Check, Device, Object, Sampler, allocator::Allocator};
use crate::{Error, Memory, Result, SamplerDesc};

/// A slot in one of the heap's arrays.
#[derive(Clone, Copy, Debug)]
pub(super) struct Slot {
    pub(super) binding: u32,
    pub(super) index: u32,
}

/// The resource a slot holds.
pub(super) enum Descriptor<'a> {
    Texture(&'a ProtocolObject<dyn MTLTexture>),
    Sampler(&'a ProtocolObject<dyn MTLSamplerState>),
    Buffer(&'a ProtocolObject<dyn MTLBuffer>),
}

/// Free slots of each array.
#[derive(Default)]
struct Slots {
    /// Slots never used yet start here.
    next: [u32; CAPACITIES.len()],
    /// Slots freed since, reused first, oldest first, so a stale index
    /// reads the empty resource for as long as possible.
    free: [VecDeque<u32>; CAPACITIES.len()],
}

/// The heap's argument buffer and what fills its empty slots.
pub(super) struct Heap {
    /// The argument buffer every pipeline reads.
    pub(super) arguments: Object<dyn MTLBuffer>,
    /// Storage buffer lengths, one `u32` per element of the arguments.
    lengths: Object<dyn MTLBuffer>,
    /// Zeroed 1x1 textures filling empty texture slots, by binding; `None`
    /// for the sampler and buffer arrays.
    empty: Vec<Option<Object<dyn MTLTexture>>>,
    /// Fills empty sampler slots.
    default_sampler: Object<dyn MTLSamplerState>,
    /// Locked to assign, write and empty slots.
    slots: Mutex<Slots>,
}

impl Heap {
    /// Creates the heap with every slot empty: zeroed textures of the
    /// array's kind, the default sampler, or an empty buffer. Its resources
    /// come from `allocator`, so encoders make them resident with the rest.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        allocator: &Allocator,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        let buffer = |elements: u32, size: usize| {
            allocator
                .buffer(device, elements as usize * size, Memory::Upload)
                .map(Object::from)
        };
        let arguments = buffer(BUFFER_LENGTHS + 1, 8)?;
        arguments.setLabel(Some(&NSString::from_str("bindless heap")));
        let lengths = buffer(BUFFER_LENGTHS, 4)?;
        let empty = (0..)
            .zip(CAPACITIES)
            .map(|(binding, _)| Self::empty_texture(device, allocator, binding))
            .collect::<Result<Vec<_>>>()?;
        Self::zero(queue, empty.iter().flatten())?;
        let heap = Self {
            arguments,
            lengths,
            empty,
            default_sampler: Sampler::create(device, &SamplerDesc::default())?,
            slots: Mutex::default(),
        };
        // SAFETY: the element after the arrays holds the lengths' address.
        unsafe {
            heap.element::<u64>(BUFFER_LENGTHS)
                .write(heap.lengths.gpuAddress());
        }
        for (binding, capacity) in (0..).zip(CAPACITIES) {
            for index in 0..capacity {
                heap.write(Slot { binding, index }, None);
            }
        }
        Ok(heap)
    }

    /// A 1x1 texture of the kind the array at `binding` holds, or `None` if
    /// it holds samplers or buffers.
    fn empty_texture(
        device: &ProtocolObject<dyn MTLDevice>,
        allocator: &Allocator,
        binding: u32,
    ) -> Result<Option<Object<dyn MTLTexture>>> {
        let texture_type = match binding {
            heap::TEXTURES_2D | heap::STORAGE_IMAGES_2D | heap::DEPTH_TEXTURES_2D => {
                MTLTextureType::Type2D
            }
            heap::TEXTURES_2D_ARRAY
            | heap::STORAGE_IMAGES_2D_ARRAY
            | heap::DEPTH_TEXTURES_2D_ARRAY => MTLTextureType::Type2DArray,
            heap::TEXTURES_CUBE | heap::DEPTH_TEXTURES_CUBE => MTLTextureType::TypeCube,
            heap::TEXTURES_CUBE_ARRAY | heap::DEPTH_TEXTURES_CUBE_ARRAY => {
                MTLTextureType::TypeCubeArray
            }
            heap::TEXTURES_3D | heap::STORAGE_IMAGES_3D => MTLTextureType::Type3D,
            _ => return Ok(None),
        };
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(texture_type);
        // Depth arrays are declared as depth textures, which need a depth
        // format.
        let depth = matches!(
            binding,
            heap::DEPTH_TEXTURES_2D..=heap::DEPTH_TEXTURES_CUBE_ARRAY
        );
        desc.setPixelFormat(if depth {
            MTLPixelFormat::Depth32Float
        } else {
            MTLPixelFormat::RGBA8Unorm
        });
        let storage = matches!(
            binding,
            heap::STORAGE_IMAGES_2D..=heap::STORAGE_IMAGES_3D
        );
        desc.setUsage(if storage {
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
        } else {
            MTLTextureUsage::ShaderRead
        });
        // SAFETY: a 1x1 texture with one mip and layer (one cube) is valid
        // for every type.
        unsafe {
            desc.setWidth(1);
            desc.setHeight(1);
            desc.setDepth(1);
            desc.setArrayLength(1);
        }
        allocator
            .texture(device, &desc)
            .map(|texture| Some(texture.into()))
    }

    /// Zeroes `textures`, waiting for the GPU to finish.
    fn zero<'a>(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        textures: impl Iterator<Item = &'a Object<dyn MTLTexture>>,
    ) -> Result<()> {
        let zeros = queue
            .device()
            .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            .check("newBufferWithLength")?;
        // SAFETY: the buffer holds four bytes the GPU does not use yet.
        unsafe { zeros.contents().cast::<u32>().write(0) };
        let commands = queue.commandBuffer().check("commandBuffer")?;
        let blit = commands.blitCommandEncoder().check("blitCommandEncoder")?;
        for texture in textures {
            let slices = if texture.textureType() == MTLTextureType::TypeCube
                || texture.textureType() == MTLTextureType::TypeCubeArray
            {
                6
            } else {
                1
            };
            for slice in 0..slices {
                // SAFETY: the buffer holds one texel, copied into each slice.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &zeros,
                        0,
                        4,
                        4,
                        MTLSize { width: 1, height: 1, depth: 1 },
                        texture,
                        slice,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
            }
        }
        blit.endEncoding();
        commands.commit();
        commands.waitUntilCompleted();
        match commands.error() {
            Some(error) => Err(Error::native(
                "zeroing the heap's empty textures",
                error.localizedDescription().to_string(),
            )),
            None => Ok(()),
        }
    }

    /// Pointer to element `element` of the argument buffer, as a `T`.
    ///
    /// # Safety
    ///
    /// `element` is at most `BUFFER_LENGTHS`, and `T` is eight bytes.
    unsafe fn element<T>(&self, element: u32) -> *mut T {
        // SAFETY: the caller keeps the element inside the buffer.
        unsafe {
            self.arguments
                .contents()
                .cast::<u64>()
                .add(element as usize)
                .cast()
                .as_ptr()
        }
    }

    /// Writes `descriptor`, or the empty one, to `slot`. The caller holds the
    /// slot lock, or the heap is not shared yet.
    fn write(&self, slot: Slot, descriptor: Option<&Descriptor<'_>>) {
        let element = first_element(slot.binding) + slot.index;
        let id: MTLResourceID = match descriptor {
            Some(Descriptor::Texture(texture)) => texture.gpuResourceID(),
            Some(Descriptor::Sampler(sampler)) => sampler.gpuResourceID(),
            Some(Descriptor::Buffer(buffer)) => {
                // Lengths of storage buffers fit, as creation checks.
                #[expect(clippy::cast_possible_truncation, reason = "the length fits")]
                let length = buffer.length() as u32;
                self.write_buffer(element, buffer.gpuAddress(), length);
                return;
            }
            None => match &self.empty[slot.binding as usize] {
                Some(texture) => texture.gpuResourceID(),
                None if slot.binding == heap::SAMPLERS => self.default_sampler.gpuResourceID(),
                None => {
                    // Shaders read nothing from a buffer of zero bytes.
                    self.write_buffer(element, 0, 0);
                    return;
                }
            },
        };
        // SAFETY: `element` is inside the arrays, and resource IDs are eight
        // bytes; no pending work reads this slot.
        unsafe { self.element::<MTLResourceID>(element).write(id) };
    }

    /// Writes a buffer at `address` of `length` bytes to `element`.
    fn write_buffer(&self, element: u32, address: u64, length: u32) {
        // SAFETY: `element` is inside both buffers, and no pending work reads
        // it.
        unsafe {
            self.element::<u64>(element).write(address);
            self.lengths
                .contents()
                .cast::<u32>()
                .add(element as usize)
                .write(length);
        }
    }
}

impl Device {
    /// Assigns `descriptor` a slot in the array at `binding`.
    pub(super) fn add_to_heap(&self, binding: u32, descriptor: &Descriptor<'_>) -> Result<Slot> {
        let heap = &self.heap;
        let mut slots = heap.slots.lock();
        let capacity = CAPACITIES[binding as usize];
        let index = match slots.free[binding as usize].pop_front() {
            Some(index) => index,
            None if slots.next[binding as usize] < capacity => {
                slots.next[binding as usize] += 1;
                slots.next[binding as usize] - 1
            }
            None => {
                return Err(Error::invalid(
                    "bindless heap",
                    &format!("binding {binding}"),
                    format!("all {capacity} slots are in use"),
                ));
            }
        };
        let slot = Slot { binding, index };
        heap.write(slot, Some(descriptor));
        Ok(slot)
    }

    /// Empties `slot` and makes it available again. Only call once no work
    /// can use the resource it held.
    pub(super) fn remove_from_heap(&self, slot: Slot) {
        let mut slots = self.heap.slots.lock();
        self.heap.write(slot, None);
        slots.free[slot.binding as usize].push_back(slot.index);
    }
}
