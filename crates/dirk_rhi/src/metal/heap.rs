//! The bindless resource heap: one tier-2 argument buffer every pipeline
//! binds, laid out as the shader build declares it (see
//! `dirk_build::shaders`): every array back to back in binding order, eight
//! bytes per element, then a pointer to one `u32` byte length per element,
//! which shaders bound their storage buffer reads with.

use std::ptr::NonNull;

use dirk_shaders::bindless::heap::{self, CAPACITIES, first_element, metal::BUFFER_LENGTHS};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRenderCommandEncoder, MTLRenderStages, MTLResource,
    MTLResourceID, MTLResourceOptions, MTLResourceUsage, MTLSamplerDescriptor, MTLSamplerState,
    MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use parking_lot::Mutex;

use super::{Check, Device, Object};
use crate::{Error, Result};

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

/// How shaders access the resources of an array: sampled textures are read,
/// storage images and buffers read and written. Indexes `Slots::resident`.
const fn usage(binding: u32) -> usize {
    (binding >= heap::STORAGE_IMAGES_2D) as usize
}

/// Usages of `Slots::resident`, by index.
const USAGES: [MTLResourceUsage; 2] = [
    MTLResourceUsage::Read,
    MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0),
];

/// Resources render passes make resident, and track hazards on, with one
/// `useResources` call.
#[derive(Default)]
struct Resident {
    /// The resources, contiguous for `useResources`.
    resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>>,
    /// The slot each resource fills; `None` for the heap's own resources.
    owners: Vec<Option<Slot>>,
}

/// Free slots of each array, and the resources render passes use.
#[derive(Default)]
struct Slots {
    /// Slots never used yet start here.
    next: [u32; CAPACITIES.len()],
    /// Slots freed since, reused first.
    free: [Vec<u32>; CAPACITIES.len()],
    /// Resources of filled slots, by `usage`.
    resident: [Resident; 2],
    /// Where each filled slot's resource is in `resident`, by binding and
    /// slot.
    positions: [Vec<usize>; CAPACITIES.len()],
}

// SAFETY: the resident pointers are thread-safe Metal resources, kept alive
// by the objects owning their slots until the slot is emptied, or by the heap.
unsafe impl Send for Slots {}

/// The heap's argument buffer and what fills its empty slots.
pub(super) struct Heap {
    /// The argument buffer every pipeline reads.
    pub(super) arguments: Object<dyn MTLBuffer>,
    /// Storage buffer lengths, one `u32` per element of the arguments.
    lengths: Object<dyn MTLBuffer>,
    /// Zeroed 1x1 textures filling empty texture slots, by binding.
    empty: Vec<Object<dyn MTLTexture>>,
    /// Fills empty sampler slots.
    default_sampler: Object<dyn MTLSamplerState>,
    /// Locked to assign, write and empty slots and to make their resources
    /// resident.
    slots: Mutex<Slots>,
}

impl Heap {
    /// Creates the heap with every slot empty: zeroed textures of the
    /// array's kind, the default sampler, or an empty buffer.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Self> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = |elements: u32, size: usize| {
            device
                .newBufferWithLength_options(elements as usize * size, shared)
                .check("newBufferWithLength")
                .map(Object::from)
        };
        let arguments = buffer(BUFFER_LENGTHS + 1, 8)?;
        arguments.setLabel(Some(&objc2_foundation::NSString::from_str("bindless heap")));
        let lengths = buffer(BUFFER_LENGTHS, 4)?;
        let empty = (heap::TEXTURES_2D..heap::SAMPLERS)
            .map(|binding| Self::empty_texture(device, binding))
            .collect::<Result<Vec<_>>>()?;
        Self::zero(queue, &empty)?;
        let sampler = MTLSamplerDescriptor::new();
        sampler.setSupportArgumentBuffers(true);
        let default_sampler = device
            .newSamplerStateWithDescriptor(&sampler)
            .check("newSamplerStateWithDescriptor")?
            .into();
        let mut slots = Slots::default();
        // The lengths and empty textures stay resident.
        for (binding, texture) in (0..).zip(&empty) {
            let resident = &mut slots.resident[usage(binding)];
            resident
                .resources
                .push(NonNull::from(ProtocolObject::from_ref(&**texture)));
            resident.owners.push(None);
        }
        slots.resident[0]
            .resources
            .push(NonNull::from(ProtocolObject::from_ref(&*lengths)));
        slots.resident[0].owners.push(None);
        let heap = Self {
            arguments,
            lengths,
            empty,
            default_sampler,
            slots: Mutex::new(slots),
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

    /// A 1x1 texture of the kind the array at `binding` holds.
    fn empty_texture(
        device: &ProtocolObject<dyn MTLDevice>,
        binding: u32,
    ) -> Result<Object<dyn MTLTexture>> {
        let texture_type = match binding {
            heap::TEXTURES_2D | heap::STORAGE_IMAGES_2D => MTLTextureType::Type2D,
            heap::TEXTURES_2D_ARRAY | heap::STORAGE_IMAGES_2D_ARRAY => MTLTextureType::Type2DArray,
            heap::TEXTURES_CUBE => MTLTextureType::TypeCube,
            heap::TEXTURES_CUBE_ARRAY => MTLTextureType::TypeCubeArray,
            _ => MTLTextureType::Type3D,
        };
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(texture_type);
        desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setUsage(if usage(binding) == 0 {
            MTLTextureUsage::ShaderRead
        } else {
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
        });
        // SAFETY: a 1x1 texture with one mip and layer (one cube) is valid
        // for every type.
        unsafe {
            desc.setWidth(1);
            desc.setHeight(1);
            desc.setDepth(1);
            desc.setArrayLength(1);
        }
        device
            .newTextureWithDescriptor(&desc)
            .check("newTextureWithDescriptor")
            .map(Object::from)
    }

    /// Zeroes `textures`, waiting for the GPU to finish.
    fn zero(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        textures: &[Object<dyn MTLTexture>],
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
            None if slot.binding == heap::BUFFERS => {
                // Shaders read nothing from a buffer of zero bytes.
                self.write_buffer(element, 0, 0);
                return;
            }
            None if slot.binding == heap::SAMPLERS => self.default_sampler.gpuResourceID(),
            None => self.empty[slot.binding as usize].gpuResourceID(),
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

    /// Makes every resource in the heap resident for `encoder`'s pass, and
    /// declares how its shaders access them so Metal tracks their hazards.
    pub(super) fn use_resources(&self, encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        let slots = self.slots.lock();
        for (resident, usage) in slots.resident.iter().zip(USAGES) {
            // SAFETY: the resources are alive while their slots are filled,
            // which the lock keeps them until the call returns.
            unsafe {
                encoder.useResources_count_usage_stages(
                    NonNull::from(resident.resources.as_slice()).cast(),
                    resident.resources.len(),
                    usage,
                    MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                );
            }
        }
    }
}

impl Device {
    /// Assigns `descriptor` a slot in the array at `binding`.
    pub(super) fn add_to_heap(&self, binding: u32, descriptor: &Descriptor<'_>) -> Result<Slot> {
        let heap = &self.heap;
        let mut slots = heap.slots.lock();
        let capacity = CAPACITIES[binding as usize];
        let index = match slots.free[binding as usize].pop() {
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
        let resource = match *descriptor {
            Descriptor::Texture(texture) => ProtocolObject::from_ref(texture),
            Descriptor::Buffer(buffer) => ProtocolObject::from_ref(buffer),
            Descriptor::Sampler(_) => return Ok(slot),
        };
        let resident = &mut slots.resident[usage(binding)];
        let position = resident.resources.len();
        resident.resources.push(NonNull::from(resource));
        resident.owners.push(Some(slot));
        let positions = &mut slots.positions[binding as usize];
        if positions.len() <= index as usize {
            positions.resize(index as usize + 1, 0);
        }
        positions[index as usize] = position;
        Ok(slot)
    }

    /// Empties `slot` and makes it available again. Only call once no work
    /// can use the resource it held.
    pub(super) fn remove_from_heap(&self, slot: Slot) {
        let mut slots = self.heap.slots.lock();
        self.heap.write(slot, None);
        slots.free[slot.binding as usize].push(slot.index);
        if slot.binding == heap::SAMPLERS {
            return;
        }
        let position = slots.positions[slot.binding as usize][slot.index as usize];
        let resident = &mut slots.resident[usage(slot.binding)];
        resident.resources.swap_remove(position);
        resident.owners.swap_remove(position);
        if let Some(Some(moved)) = resident.owners.get(position).copied() {
            slots.positions[moved.binding as usize][moved.index as usize] = position;
        }
    }
}
