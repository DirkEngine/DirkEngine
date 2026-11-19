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
//! Views of depth images live in their own arrays, which shaders declare as
//! depth images so they can compare them against a reference through a
//! comparison sampler, as shadow maps are filtered (see
//! [`DepthTextureIndex`]).
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

use core::{fmt, marker::PhantomData};

use bytemuck::NoUninit;

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
    /// Sampled 2D depth images, declared as depth images.
    pub const DEPTH_TEXTURES_2D: u32 = 10;
    /// Sampled 2D array depth images, declared as depth images.
    pub const DEPTH_TEXTURES_2D_ARRAY: u32 = 11;
    /// Sampled cube depth images, declared as depth images.
    pub const DEPTH_TEXTURES_CUBE: u32 = 12;
    /// Sampled cube array depth images, declared as depth images.
    pub const DEPTH_TEXTURES_CUBE_ARRAY: u32 = 13;

    /// Slots in each image array, sampled, depth and storage alike, so an
    /// image index masked to it fits every image array.
    pub const IMAGE_CAPACITY: u32 = 1 << 14;
    /// Slots in the sampler array.
    pub const SAMPLER_CAPACITY: u32 = 1 << 10;
    /// Slots in the storage buffer array.
    pub const BUFFER_CAPACITY: u32 = 1 << 16;

    /// Capacity of each array, indexed by binding.
    pub const CAPACITIES: [u32; 14] = [
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
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
        IMAGE_CAPACITY,
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
        #[expect(clippy::cast_possible_truncation, reason = "there are fourteen arrays")]
        pub const BUFFER_LENGTHS: u32 = super::first_element(super::CAPACITIES.len() as u32);
    }
}

/// Declares an index into the heap's arrays of one kind, and with the `gpu`
/// feature its shader-side access, to arrays of the `$element` elements.
macro_rules! indices {
    ($($(#[$doc:meta])* $name:ident: $capacity:ident, $element:ident,)+) => {$(
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, NoUninit)]
        pub struct $name(u32);

        impl $name {
            /// The index of `slot`. Created by the RHI.
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
    )+};
}

indices! {
    /// A sampled image view in the heap: in the texture array of its view
    /// type, such as [`heap::TEXTURES_2D`]. Views of depth images have a
    /// [`DepthTextureIndex`] instead.
    TextureIndex: IMAGE_CAPACITY, Texture,
    /// A sampled depth image view in the heap: in the depth texture array of
    /// its view type, such as [`heap::DEPTH_TEXTURES_2D`]. Shaders declare
    /// these arrays as depth images (`Image!(2D, type = f32, sampled,
    /// depth)`), which they can sample, load, or compare against a
    /// reference with a comparison sampler (`sample_depth_reference`).
    DepthTextureIndex: IMAGE_CAPACITY, DepthTexture,
    /// A storage image view in the heap: in the storage image array of its
    /// view type, such as [`heap::STORAGE_IMAGES_2D`].
    StorageImageIndex: IMAGE_CAPACITY, StorageImage,
    /// A sampler in the heap, at [`heap::SAMPLERS`]. Comparison samplers
    /// share the array; shaders use them only with depth comparisons.
    SamplerIndex: SAMPLER_CAPACITY, HeapSampler,
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
    /// The index of `slot`. Created by the RHI.
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

// SAFETY: three 4-byte fields (`BufferIndex` is a transparent `u32`) leave
// no padding, and `T` is only a marker. Deriving cannot see through `T`.
#[expect(
    unsafe_code,
    reason = "bytemuck cannot derive this for generic structs"
)]
unsafe impl<T: 'static> NoUninit for BufferSlice<T> {}

impl<T> BufferSlice<T> {
    /// The `len` elements of `buffer` from element `first`. Created by the
    /// RHI.
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

    /// Elements of the heap's arrays of sampled color images, which
    /// [`TextureIndex`](super::TextureIndex) reads: single-sampled sampled
    /// 2D, 2D array, cube, cube array and 3D images not declared as depth
    /// images.
    pub trait Texture: sealed::Sealed {}

    /// Elements of the heap's arrays of sampled depth images, which
    /// [`DepthTextureIndex`](super::DepthTextureIndex) reads: single-sampled
    /// sampled 2D, 2D array, cube and cube array images declared as depth
    /// images.
    pub trait DepthTexture: sealed::Sealed {}

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
        DepthTexture:
            (TwoD, False, True, Yes),
            (TwoD, True, True, Yes),
            (Cube, False, True, Yes),
            (Cube, True, True, Yes);
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
