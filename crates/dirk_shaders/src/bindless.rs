//! Indices into the bindless resource heap, shared by host code and shaders.
//! Shader crates enable the `gpu` feature to read the heap through them.
//!
//! The heap is descriptor set 0: one array per kind of resource, at the
//! bindings in [`heap`]. Shaders declare the arrays they read as entry point
//! parameters with those binding numbers, written as literals because
//! attributes cannot name constants:
//!
//! ```ignore
//! #[spirv(fragment)]
//! pub fn main_fs(
//!     #[spirv(push_constant)] push: &MaterialPush,
//!     #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image!(2D, type = f32, sampled)>,
//!     #[spirv(descriptor_set = 0, binding = 8)] samplers: &RuntimeArray<Sampler>,
//!     uv: Vec2,
//!     color: &mut Vec4,
//! ) {
//!     let texture = push.albedo.get(textures);
//!     *color = texture.sample(*push.sampler.get(samplers), uv);
//! }
//! ```
//!
//! Each index reads only the arrays of its kind: its `get` accepts only
//! their element types, which the shader build checks against the heap's
//! bindings, and masks the index to their capacity, which is the same for
//! every image array. Every slot holds a valid descriptor of its array's
//! kind and dimension: a live resource, or, once freed, a null one (zeros,
//! or a default sampler). So whatever its value, an index never reads
//! outside its array or a descriptor of another kind; a stale or forged one
//! reads the null descriptor or another live resource of that kind. Freed
//! slots are reused first in, first out, so a stale index reads the null
//! descriptor for as long as possible.
//!
//! Push data and the elements of frame allocations implement
//! [`HeapIndices`], usually derived, which lists the indices they hold. The
//! RHI keeps the resource of every index it sees this way, and its slot,
//! until the work reading it completes, even once its owner dropped it.
//!
//! That is all the RHI tracks. The caller keeps alive the owners of indices
//! it does not see (in storage buffers, written by the GPU, or computed by
//! shaders) until the work reading them completes, since their slots may
//! otherwise be filled again while it runs, which Vulkan forbids. Heap reads
//! also bypass images' accesses: barriers or the render graph order them
//! after writes, and shaders read images only in their resting or a
//! declared shader access, never one the current render pass renders to;
//! otherwise they read undefined values, though never outside the image.

use core::{fmt, marker::PhantomData};

use bytemuck::NoUninit;
/// Derives [`HeapIndices`](trait@HeapIndices) for a struct, listing the
/// indices of every field.
pub use dirk_proc::HeapIndices;

/// Binding numbers of the heap's arrays in descriptor set 0, and their
/// capacities, which are powers of two.
pub mod heap {
    /// Sampled 2D images.
    pub const TEXTURES_2D: u32 = 0;
    /// Sampled 2D array images.
    pub const TEXTURES_2D_ARRAY: u32 = 1;
    /// Sampled cube images.
    pub const TEXTURES_CUBE: u32 = 2;
    /// Sampled cube array images.
    pub const TEXTURES_CUBE_ARRAY: u32 = 3;
    /// Sampled 3D images.
    pub const TEXTURES_3D: u32 = 4;
    /// 2D storage images, declared without a format.
    pub const STORAGE_IMAGES_2D: u32 = 5;
    /// 2D array storage images, declared without a format.
    pub const STORAGE_IMAGES_2D_ARRAY: u32 = 6;
    /// 3D storage images, declared without a format.
    pub const STORAGE_IMAGES_3D: u32 = 7;
    /// Samplers.
    pub const SAMPLERS: u32 = 8;
    /// Storage buffers.
    pub const BUFFERS: u32 = 9;

    /// Slots in each image array, sampled and storage alike, so an image
    /// index masked to it fits every image array.
    pub const IMAGE_CAPACITY: u32 = 1 << 14;
    /// Slots in the sampler array.
    pub const SAMPLER_CAPACITY: u32 = 1 << 10;
    /// Slots in the storage buffer array.
    pub const BUFFER_CAPACITY: u32 = 1 << 16;

    /// Capacity of each array, indexed by binding.
    pub const CAPACITIES: [u32; 10] = [
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        SAMPLER_CAPACITY,
        BUFFER_CAPACITY,
    ];

    /// Index of the first element of the array at `binding` when the arrays
    /// are laid out back to back in binding order, as Metal's argument
    /// buffer holds them. The binding after the last gives their total.
    #[must_use]
    pub const fn first_element(binding: u32) -> u32 {
        let mut first = 0;
        let mut index = 0;
        while index < binding as usize {
            first += CAPACITIES[index];
            index += 1;
        }
        first
    }

    /// How Metal shaders reach the heap, which the shader build's MSL
    /// translation and the RHI's Metal backend agree on.
    pub mod metal {
        /// Buffer index of the heap's argument buffer.
        pub const HEAP_BUFFER: u32 = 0;
        /// Buffer index of push data.
        pub const PUSH_BUFFER: u32 = 1;
        /// Argument buffer element after the arrays: a pointer to one `u32`
        /// per element, the byte length of storage buffers, which shaders
        /// bound their reads with.
        #[expect(clippy::cast_possible_truncation, reason = "there are ten arrays")]
        pub const BUFFER_LENGTHS: u32 = super::first_element(super::CAPACITIES.len() as u32);
    }
}

/// Declares an index into the heap's arrays of one kind, and with the `gpu`
/// feature its shader-side access, to arrays of the `$element` elements.
macro_rules! indices {
    ($($(#[$doc:meta])* $name:ident: $capacity:ident, $element:ident, $variant:ident,)+) => {$(
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, NoUninit)]
        pub struct $name(u32);

        impl $name {
            /// The index of `slot`. The RHI creates the indices of its
            /// resources; one made from any other number is safe, reading a
            /// live or null resource of its kind (see the [module](self)).
            #[must_use]
            pub const fn new(slot: u32) -> Self {
                Self(slot)
            }

            /// The slot, masked to the array's capacity.
            #[must_use]
            pub const fn slot(self) -> u32 {
                self.0 & (heap::$capacity - 1)
            }

            /// This resource in `array`, the heap's array of its kind and
            /// view type.
            #[cfg(feature = "gpu")]
            #[must_use]
            #[expect(
                unsafe_code,
                reason = "array accesses are unchecked, but the index is masked to the array's capacity"
            )]
            pub fn get<R: gpu::$element>(self, array: &spirv_std::RuntimeArray<R>) -> &R {
                // SAFETY: the slot is masked to the capacity of every array
                // whose elements are `$element`s.
                unsafe { array.index(self.slot() as usize) }
            }
        }

        // SAFETY: the index is the only one it holds.
        #[expect(unsafe_code, reason = "implementors promise to list every index")]
        unsafe impl HeapIndices for $name {
            fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
                visit(HeapIndex::$variant(*self));
            }
        }
    )+};
}

indices! {
    /// A sampled image view in the heap: in the texture array of its view
    /// type, such as [`heap::TEXTURES_2D`].
    TextureIndex: IMAGE_CAPACITY, Texture, Texture,
    /// A storage image view in the heap: in the storage image array of its
    /// view type, such as [`heap::STORAGE_IMAGES_2D`].
    StorageImageIndex: IMAGE_CAPACITY, StorageImage, StorageImage,
    /// A sampler in the heap, at [`heap::SAMPLERS`].
    SamplerIndex: SAMPLER_CAPACITY, HeapSampler, Sampler,
}

/// A storage buffer in the heap, at [`heap::BUFFERS`], read by shaders as an
/// array of `T`.
#[repr(transparent)]
#[derive(NoUninit)]
pub struct BufferIndex<T> {
    slot: u32,
    element: PhantomData<T>,
}

impl<T> BufferIndex<T> {
    /// The index of `slot`. The RHI creates the indices of its buffers;
    /// one made from any other number is safe, reading a live or null
    /// buffer (see the [module](self)).
    #[must_use]
    pub const fn new(slot: u32) -> Self {
        Self {
            slot,
            element: PhantomData,
        }
    }

    /// The slot, masked to the array's capacity.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot & (heap::BUFFER_CAPACITY - 1)
    }
}

/// `len` elements of a storage buffer in the heap read as an array of `T`,
/// from element `first`, such as memory a frame allocated. Shaders read
/// only inside it.
#[repr(C)]
pub struct BufferSlice<T> {
    buffer: BufferIndex<T>,
    first: u32,
    len: u32,
}

#[expect(
    unsafe_code,
    reason = "bytemuck cannot derive this for generic structs"
)]
// SAFETY: three 4-byte fields (`BufferIndex` is a transparent `u32`) leave
// no padding, and `T` is only a marker. Deriving cannot see through `T`.
unsafe impl<T: 'static> NoUninit for BufferSlice<T> {}

impl<T> BufferSlice<T> {
    /// The `len` elements of `buffer` from element `first`. The RHI creates
    /// the slices of frame allocations; any other is safe too, since shaders
    /// bound their reads by the buffer as well as by the slice.
    #[must_use]
    pub const fn new(buffer: BufferIndex<T>, first: u32, len: u32) -> Self {
        Self { buffer, first, len }
    }

    /// The buffer holding the elements.
    #[must_use]
    pub const fn buffer(self) -> BufferIndex<T> {
        self.buffer
    }

    /// The index of the first element in the buffer.
    #[must_use]
    pub const fn first(self) -> u32 {
        self.first
    }

    /// The number of elements.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Whether there are no elements.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// A heap index of any kind, as [`HeapIndices`] lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeapIndex {
    /// A sampled image view.
    Texture(TextureIndex),
    /// A storage image view.
    StorageImage(StorageImageIndex),
    /// A sampler.
    Sampler(SamplerIndex),
    /// A storage buffer, whatever its elements.
    Buffer(BufferIndex<()>),
}

/// Values holding heap indices, which list them: push data and the
/// elements of frame allocations. The RHI keeps the resources they index,
/// and their slots, until the work recorded with them completes. Derive it
/// with `#[derive(HeapIndices)]`, which lists the indices of every field.
///
/// The RHI never sees indices in other memory, such as a storage buffer of
/// materials, written by the GPU or computed by shaders, so the work reading
/// them relies on the caller to keep their resources alive until it
/// completes: otherwise their slots may be emptied and filled again while it
/// runs.
///
/// # Safety
///
/// [`heap_indices`](Self::heap_indices) passes every index the value holds
/// to `visit`. The RHI may empty and reuse the slot of an index it is not
/// told about while work reading it still runs.
#[expect(
    unsafe_code,
    reason = "the RHI's soundness relies on implementors listing every index"
)]
pub unsafe trait HeapIndices {
    /// Passes each heap index `self` holds to `visit`.
    fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V);
}

/// Implements [`HeapIndices`] for types holding no index.
macro_rules! no_indices {
    ($($ty:ty),+) => {$(
        // SAFETY: the type holds no index.
        #[expect(unsafe_code, reason = "implementors promise to list every index")]
        unsafe impl HeapIndices for $ty {
            fn heap_indices<V: FnMut(HeapIndex)>(&self, _: &mut V) {}
        }
    )+};
}

no_indices!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, ());

// SAFETY: lists the indices of every element.
#[expect(unsafe_code, reason = "implementors promise to list every index")]
unsafe impl<T: HeapIndices> HeapIndices for [T] {
    fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
        for element in self {
            element.heap_indices(visit);
        }
    }
}

// SAFETY: lists the indices of every element.
#[expect(unsafe_code, reason = "implementors promise to list every index")]
unsafe impl<T: HeapIndices, const N: usize> HeapIndices for [T; N] {
    fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
        self.as_slice().heap_indices(visit);
    }
}

// SAFETY: the buffer is the only index it holds.
#[expect(unsafe_code, reason = "implementors promise to list every index")]
unsafe impl<T> HeapIndices for BufferIndex<T> {
    fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
        visit(HeapIndex::Buffer(BufferIndex::new(self.slot)));
    }
}

// SAFETY: the buffer is the only index it holds.
#[expect(unsafe_code, reason = "implementors promise to list every index")]
unsafe impl<T> HeapIndices for BufferSlice<T> {
    fn heap_indices<V: FnMut(HeapIndex)>(&self, visit: &mut V) {
        self.buffer.heap_indices(visit);
    }
}

/// Implements `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash` and `Debug` for
/// types generic over the elements they index, comparing `$field`s. Deriving
/// them would require the elements to implement them too.
macro_rules! element_generic {
    ($($name:ident { $($field:ident),+ })+) => {$(
        impl<T> Clone for $name<T> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T> Copy for $name<T> {}

        impl<T> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool {
                $(self.$field == other.$field)&&+
            }
        }

        impl<T> Eq for $name<T> {}

        impl<T> core::hash::Hash for $name<T> {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                $(self.$field.hash(state);)+
            }
        }

        impl<T> fmt::Debug for $name<T> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    $(.field(stringify!($field), &self.$field))+
                    .finish()
            }
        }
    )+};
}

element_generic! {
    BufferIndex { slot }
    BufferSlice { buffer, first, len }
}

/// Shader-side access to the heap: the element types of its arrays, and
/// storage buffers.
#[cfg(feature = "gpu")]
#[expect(
    unsafe_code,
    reason = "array accesses are unchecked, but every index is masked to the array's capacity"
)]
pub mod gpu {
    use spirv_std::{
        RuntimeArray, Sampler, TypedBuffer,
        image::{Arrayed, Dimensionality, Image, ImageDepth, Multisampled, SampleType, Sampled},
    };

    use super::{BufferIndex, BufferSlice};

    mod sealed {
        /// Prevents element types outside this module.
        pub trait Sealed {}
    }

    /// Elements of the heap's arrays of sampled images, which
    /// [`TextureIndex`](super::TextureIndex) reads: single-sampled sampled
    /// 2D, 2D array, cube, cube array and 3D images not declared as depth
    /// images.
    pub trait Texture: sealed::Sealed {}

    /// Elements of the heap's arrays of storage images, which
    /// [`StorageImageIndex`](super::StorageImageIndex) reads: single-sampled
    /// storage 2D, 2D array and 3D images not declared as depth images.
    pub trait StorageImage: sealed::Sealed {}

    /// The element of the heap's sampler array, which
    /// [`SamplerIndex`](super::SamplerIndex) reads.
    pub trait HeapSampler: sealed::Sealed {}

    impl sealed::Sealed for Sampler {}
    impl HeapSampler for Sampler {}

    /// Makes the images of each dimensionality, arrayedness, depth and
    /// sampling, of any sampled type and format, elements of `$trait`.
    macro_rules! elements {
        ($($trait:ident: $(($dim:ident, $arrayed:ident, $depth:ident, $sampled:ident)),+;)+) => {$($(
            impl<S: SampleType<FORMAT, COMPONENTS>, const FORMAT: u32, const COMPONENTS: u32>
                sealed::Sealed
                for Image<
                    S,
                    { Dimensionality::$dim as u32 },
                    { ImageDepth::$depth as u32 },
                    { Arrayed::$arrayed as u32 },
                    { Multisampled::False as u32 },
                    { Sampled::$sampled as u32 },
                    FORMAT,
                    COMPONENTS,
                >
            {
            }

            impl<S: SampleType<FORMAT, COMPONENTS>, const FORMAT: u32, const COMPONENTS: u32> $trait
                for Image<
                    S,
                    { Dimensionality::$dim as u32 },
                    { ImageDepth::$depth as u32 },
                    { Arrayed::$arrayed as u32 },
                    { Multisampled::False as u32 },
                    { Sampled::$sampled as u32 },
                    FORMAT,
                    COMPONENTS,
                >
            {
            }
        )+)+};
    }

    elements! {
        Texture:
            (TwoD, False, Unknown, Yes),
            (TwoD, False, False, Yes),
            (TwoD, True, Unknown, Yes),
            (TwoD, True, False, Yes),
            (Cube, False, Unknown, Yes),
            (Cube, False, False, Yes),
            (Cube, True, Unknown, Yes),
            (Cube, True, False, Yes),
            (ThreeD, False, Unknown, Yes),
            (ThreeD, False, False, Yes);
        StorageImage:
            (TwoD, False, Unknown, No),
            (TwoD, False, False, No),
            (TwoD, True, Unknown, No),
            (TwoD, True, False, No),
            (ThreeD, False, Unknown, No),
            (ThreeD, False, False, No);
    }

    impl<T> BufferIndex<T> {
        /// This buffer's elements in `buffers`, the heap's storage buffer
        /// array declared with elements of `T`.
        #[must_use]
        pub fn get(self, buffers: &RuntimeArray<TypedBuffer<[T]>>) -> &[T] {
            // SAFETY: the slot is masked to the array's capacity.
            unsafe { buffers.index(self.slot() as usize) }
        }
    }

    impl<T> BufferSlice<T> {
        /// Element `index` of the slice in `buffers`, the heap's storage
        /// buffer array declared with elements of `T`.
        ///
        /// # Panics
        ///
        /// Panics, ending the invocation, if `index` is outside the slice or
        /// the slice outside its buffer.
        #[must_use]
        pub fn get(self, buffers: &RuntimeArray<TypedBuffer<[T]>>, index: u32) -> &T {
            assert!(index < self.len, "index outside the buffer slice");
            &self.buffer.get(buffers)[self.first.wrapping_add(index) as usize]
        }
    }
}
