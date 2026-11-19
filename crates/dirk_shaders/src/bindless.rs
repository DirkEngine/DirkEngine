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
//! Indices are memory-safe whatever their value: shaders mask them to their
//! array's capacity, every slot holds a valid descriptor (a null one once
//! freed), and each array holds one kind and dimension of resource. A stale
//! or forged index reads the wrong resource, or zeros, but nothing invalid.

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

    /// Slots in each texture array.
    pub const TEXTURE_CAPACITY: u32 = 1 << 14;
    /// Slots in each storage image array.
    pub const STORAGE_IMAGE_CAPACITY: u32 = 1 << 12;
    /// Slots in the sampler array.
    pub const SAMPLER_CAPACITY: u32 = 1 << 10;
    /// Slots in the storage buffer array.
    pub const BUFFER_CAPACITY: u32 = 1 << 16;
    /// Slots in each depth texture array.
    pub const DEPTH_TEXTURE_CAPACITY: u32 = 1 << 10;

    /// Capacity of each array, indexed by binding.
    pub const CAPACITIES: [u32; 14] = [
        TEXTURE_CAPACITY,
        TEXTURE_CAPACITY,
        TEXTURE_CAPACITY,
        TEXTURE_CAPACITY,
        TEXTURE_CAPACITY,
        STORAGE_IMAGE_CAPACITY,
        STORAGE_IMAGE_CAPACITY,
        STORAGE_IMAGE_CAPACITY,
        SAMPLER_CAPACITY,
        BUFFER_CAPACITY,
        DEPTH_TEXTURE_CAPACITY,
        DEPTH_TEXTURE_CAPACITY,
        DEPTH_TEXTURE_CAPACITY,
        DEPTH_TEXTURE_CAPACITY,
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

/// Declares an index into one of the heap's arrays, and with the `gpu`
/// feature its shader-side access.
macro_rules! indices {
    ($($(#[$doc:meta])* $name:ident: $capacity:ident,)+) => {$(
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

            /// This resource in `array`, the heap's array it indexes.
            #[cfg(feature = "gpu")]
            #[must_use]
            #[expect(
                unsafe_code,
                reason = "array accesses are unchecked, but the index is masked to the array's capacity"
            )]
            pub fn get<R>(self, array: &spirv_std::RuntimeArray<R>) -> &R {
                // SAFETY: the slot is masked to the array's capacity.
                unsafe { array.index(self.slot() as usize) }
            }
        }
    )+};
}

indices! {
    /// A sampled image view in the heap: in the texture array of its view
    /// type, such as [`heap::TEXTURES_2D`]. Views of depth images have a
    /// [`DepthTextureIndex`] instead.
    TextureIndex: TEXTURE_CAPACITY,
    /// A sampled depth image view in the heap: in the depth texture array of
    /// its view type, such as [`heap::DEPTH_TEXTURES_2D`]. Shaders declare
    /// these arrays as depth images (`Image!(2D, type = f32, sampled,
    /// depth)`), which they can sample, load, or compare against a
    /// reference with a comparison sampler (`sample_depth_reference`).
    DepthTextureIndex: DEPTH_TEXTURE_CAPACITY,
    /// A storage image view in the heap: in the storage image array of its
    /// view type, such as [`heap::STORAGE_IMAGES_2D`].
    StorageImageIndex: STORAGE_IMAGE_CAPACITY,
    /// A sampler in the heap, at [`heap::SAMPLERS`]. Comparison samplers
    /// share the array; shaders use them only with depth comparisons.
    SamplerIndex: SAMPLER_CAPACITY,
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

// Implemented by hand: deriving would require `T` to implement them too.
impl<T> Clone for BufferIndex<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for BufferIndex<T> {}

impl<T> PartialEq for BufferIndex<T> {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}

impl<T> Eq for BufferIndex<T> {}

impl<T> core::hash::Hash for BufferIndex<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
    }
}

impl<T> fmt::Debug for BufferIndex<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BufferIndex")
            .field(&self.slot)
            .finish()
    }
}

/// Shader-side access to storage buffers.
#[cfg(feature = "gpu")]
#[expect(
    unsafe_code,
    reason = "array accesses are unchecked, but every index is masked to the array's capacity"
)]
mod gpu {
    use spirv_std::{RuntimeArray, TypedBuffer};

    use super::BufferIndex;

    impl<T> BufferIndex<T> {
        /// This buffer's elements in `buffers`, the heap's storage buffer
        /// array declared with elements of `T`.
        #[must_use]
        pub fn get(self, buffers: &RuntimeArray<TypedBuffer<[T]>>) -> &[T] {
            // SAFETY: the slot is masked to the array's capacity.
            unsafe { buffers.index(self.slot() as usize) }
        }

        /// This buffer's elements in `buffers`, as [`BufferIndex::get`], to
        /// write them.
        #[must_use]
        pub fn get_mut(self, buffers: &mut RuntimeArray<TypedBuffer<[T]>>) -> &mut [T] {
            // SAFETY: the slot is masked to the array's capacity.
            unsafe { buffers.index_mut(self.slot() as usize) }
        }
    }
}
