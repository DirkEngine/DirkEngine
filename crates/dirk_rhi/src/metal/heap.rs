//! The bindless resource heap: one tier-2 argument buffer every pipeline
//! binds, laid out as the shader build declares it (see
//! `dirk_build::shaders`): every array back to back in binding order, eight
//! bytes per element, then a pointer to one `u32` byte length per element,
//! which shaders bound their storage buffer reads with.
//!
//! Every resource lives in one of the device's heaps, which encoders make
//! resident with `useHeap`. Metal makes heaps resident for reads only, and
//! texture views apart from their heap, so encoders also declare what that
//! does not cover with `useResources`: storage buffers and images, which
//! shaders may write, and the subviews of images.

use std::{
    collections::VecDeque,
    ptr::NonNull,
    sync::atomic::{AtomicU64, Ordering},
};

use dirk_shaders::bindless::heap::{self, CAPACITIES, first_element, metal::BUFFER_LENGTHS};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLResource, MTLResourceID, MTLResourceOptions,
    MTLResourceUsage, MTLSamplerState, MTLSize, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use parking_lot::Mutex;

use super::{Check, Device, Object, Sampler, allocator::Allocator};
use crate::{Error, Memory, Result, SamplerDesc};

/// A slot in one of the heap's arrays.
#[derive(Clone, Copy, Debug)]
pub(super) struct Slot {
    pub(super) binding: u32,
    pub(super) index: u32,
    /// The usage encoders declare its resource with, as an index into
    /// [`USAGES`], if they do.
    declared: Option<usize>,
}

/// The resource a slot holds.
pub(super) enum Descriptor<'a> {
    Texture(&'a ProtocolObject<dyn MTLTexture>),
    Sampler(&'a ProtocolObject<dyn MTLSamplerState>),
    Buffer(&'a ProtocolObject<dyn MTLBuffer>),
}

/// How encoders declare resources `useHeap` does not cover, indexing
/// [`Declared`]: storage resources for writes, subviews for reads.
const USAGES: [MTLResourceUsage; 2] = [
    MTLResourceUsage::Read,
    MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0),
];

/// Resources encoders declare with one `useResources` call per usage, by
/// index into [`USAGES`]; contiguous for those calls.
pub(super) type Declared = [Vec<NonNull<ProtocolObject<dyn MTLResource>>>; 2];

/// Free slots of each array, and the resources of filled ones encoders
/// declare.
#[derive(Default)]
struct Slots {
    /// Slots never used yet start here.
    next: [u32; CAPACITIES.len()],
    /// Slots freed since, reused first, oldest first, so a stale index
    /// reads the empty resource for as long as possible.
    free: [VecDeque<u32>; CAPACITIES.len()],
    declared: Declared,
    /// The slot each declared resource fills, alongside `declared`; `None`
    /// for the heap's own empty storage images.
    owners: [Vec<Option<Slot>>; 2],
    /// Where each declared slot's resource is in `declared`, by binding
    /// and slot.
    positions: [Vec<usize>; CAPACITIES.len()],
}

// SAFETY: the declared pointers are thread-safe Metal resources, kept alive
// by the objects owning their slots until the slot is emptied, or by the
// heap.
unsafe impl Send for Slots {}

impl Slots {
    /// Declares `resource`, filling `owner`, with usage `usage`.
    fn declare(
        &mut self,
        resource: &ProtocolObject<dyn MTLResource>,
        usage: usize,
        owner: Option<Slot>,
    ) {
        if let Some(slot) = owner {
            let positions = &mut self.positions[slot.binding as usize];
            if positions.len() <= slot.index as usize {
                positions.resize(slot.index as usize + 1, 0);
            }
            positions[slot.index as usize] = self.declared[usage].len();
        }
        self.declared[usage].push(NonNull::from(resource));
        self.owners[usage].push(owner);
    }

    /// Stops declaring the resource filling `slot`, with usage `usage`.
    fn forget(&mut self, slot: Slot, usage: usize) {
        let position = self.positions[slot.binding as usize][slot.index as usize];
        self.declared[usage].swap_remove(position);
        self.owners[usage].swap_remove(position);
        if let Some(Some(moved)) = self.owners[usage].get(position).copied() {
            self.positions[moved.binding as usize][moved.index as usize] = position;
        }
    }
}

impl Descriptor<'_> {
    /// The usage encoders declare the resource with, as an index into
    /// [`USAGES`], filling a slot at `binding`; `None` if `useHeap` covers
    /// it. Views are told apart by their parent texture.
    fn declared(&self, binding: u32) -> Option<usize> {
        match self {
            Self::Buffer(_) => Some(1),
            Self::Texture(_) if is_storage(binding) => Some(1),
            Self::Texture(texture) => texture.parentTexture().map(|_| 0),
            Self::Sampler(_) => None,
        }
    }
}

/// Whether the array at `binding` holds storage images.
const fn is_storage(binding: u32) -> bool {
    matches!(binding, heap::STORAGE_IMAGES_2D..=heap::STORAGE_IMAGES_3D)
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
    /// Changes whenever the declared resources do, so command buffers know
    /// when to declare them again without locking.
    generation: AtomicU64,
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
        let mut slots = Slots::default();
        // Shaders may write the empty storage images through stale or
        // forged indices.
        for (binding, texture) in (0..).zip(&empty) {
            if let Some(texture) = texture.as_ref().filter(|_| is_storage(binding)) {
                slots.declare(ProtocolObject::from_ref(&**texture), 1, None);
            }
        }
        let heap = Self {
            arguments,
            lengths,
            empty,
            default_sampler: Sampler::create(device, &SamplerDesc::default())?,
            slots: Mutex::new(slots),
            generation: AtomicU64::new(1),
        };
        // SAFETY: the element after the arrays holds the lengths' address.
        unsafe {
            heap.element::<u64>(BUFFER_LENGTHS)
                .write(heap.lengths.gpuAddress());
        }
        for (binding, capacity) in (0..).zip(CAPACITIES) {
            for index in 0..capacity {
                let slot = Slot {
                    binding,
                    index,
                    declared: None,
                };
                heap.write(slot, None);
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
        desc.setUsage(if is_storage(binding) {
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

impl Heap {
    /// The generation of the declared resources, which changes whenever
    /// they do. Lock-free, for checks before every draw.
    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// The resources encoders declare, and their generation. They stay
    /// alive until the slots they fill are emptied, which happens only
    /// between frames, so until the open frame closes.
    pub(super) fn declared(&self) -> (u64, Declared) {
        let slots = self.slots.lock();
        (self.generation(), slots.declared.clone())
    }

    /// Passes each usage's resources in `declared` to `declare`, as Metal's
    /// `useResources` takes them: `count` at `resources`.
    pub(super) fn declare(
        declared: &Declared,
        mut declare: impl FnMut(
            NonNull<NonNull<ProtocolObject<dyn MTLResource>>>,
            usize,
            MTLResourceUsage,
        ),
    ) {
        for (resources, usage) in declared.iter().zip(USAGES) {
            if !resources.is_empty() {
                declare(
                    NonNull::from(resources.as_slice()).cast(),
                    resources.len(),
                    usage,
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
        let slot = Slot {
            binding,
            index,
            declared: descriptor.declared(binding),
        };
        heap.write(slot, Some(descriptor));
        let resource = match *descriptor {
            Descriptor::Texture(texture) => ProtocolObject::from_ref(texture),
            Descriptor::Buffer(buffer) => ProtocolObject::from_ref(buffer),
            Descriptor::Sampler(_) => return Ok(slot),
        };
        if let Some(usage) = slot.declared {
            slots.declare(resource, usage, Some(slot));
            heap.generation.fetch_add(1, Ordering::Release);
        }
        Ok(slot)
    }

    /// Empties `slot` and makes it available again. Only call once no work
    /// can use the resource it held.
    pub(super) fn remove_from_heap(&self, slot: Slot) {
        let mut slots = self.heap.slots.lock();
        self.heap.write(slot, None);
        slots.free[slot.binding as usize].push_back(slot.index);
        if let Some(usage) = slot.declared {
            slots.forget(slot, usage);
            self.heap.generation.fetch_add(1, Ordering::Release);
        }
    }
}
