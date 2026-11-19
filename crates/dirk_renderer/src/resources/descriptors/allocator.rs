//! Typed descriptor set allocator.

use std::{marker::PhantomData, mem::MaybeUninit};

use ash::vk;

use crate::{
    Result,
    resources::device::{Garbage, RenderDevice},
};

use super::{layouts::SetLayout, set::DescriptorSet};

/// Growable descriptor pool allocator for a single layout type.
pub struct DescriptorAllocator<L: SetLayout> {
    device: RenderDevice,
    layout: vk::DescriptorSetLayout,
    pools: Vec<vk::DescriptorPool>,
    cursor: usize,
    next_max_sets: u32,
    _layout: PhantomData<L>,
}

impl<L: SetLayout> DescriptorAllocator<L> {
    /// Creates an allocator with one initial pool page.
    pub fn new(device: &RenderDevice, initial_max_sets: u32) -> Result<Self> {
        let initial_max_sets = initial_max_sets.max(1);
        let mut allocator = Self {
            device: device.clone(),
            layout: L::create_layout(&device.device)?,
            pools: Vec::new(),
            cursor: 0,
            next_max_sets: initial_max_sets,
            _layout: PhantomData,
        };
        allocator.add_page(initial_max_sets)?;
        Ok(allocator)
    }

    /// Allocates one descriptor set.
    pub fn allocate(&mut self) -> Result<DescriptorSet<L>> {
        if !self.pools.is_empty() {
            match self.allocate_from(self.cursor) {
                Ok(set) => return Ok(set),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(error) => return Err(error.into()),
            }
        }

        let page_index = self.add_page(1)?;
        self.allocate_from(page_index).map_err(Into::into)
    }

    /// Allocates exactly `N` descriptor sets.
    pub fn allocate_array<const N: usize>(&mut self) -> Result<[DescriptorSet<L>; N]> {
        let mut out: [MaybeUninit<DescriptorSet<L>>; N] =
            std::array::from_fn(|_| MaybeUninit::uninit());

        for i in 0..N {
            match self.allocate() {
                Ok(set) => {
                    out[i].write(set);
                }
                Err(error) => {
                    for initialized in out.iter_mut().take(i) {
                        // SAFETY: slots before `i` were initialized above.
                        unsafe { initialized.assume_init_drop() };
                    }
                    return Err(error);
                }
            }
        }

        // SAFETY: every slot was initialized by the loop above.
        Ok(std::array::from_fn(|i| unsafe {
            out[i].assume_init_read()
        }))
    }

    fn allocate_from(&self, page_index: usize) -> ash::prelude::VkResult<DescriptorSet<L>> {
        let pool = self.pools[page_index];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&self.layout));
        // SAFETY: `pool` and `self.layout` are live objects from this device,
        // and `alloc_info` outlives the call. On success Vulkan returns one set
        // per layout, here exactly one.
        let raw_sets = unsafe { self.device.device.allocate_descriptor_sets(&alloc_info)? };
        Ok(DescriptorSet::new(self.device.clone(), pool, raw_sets[0]))
    }

    fn add_page(&mut self, required_sets: u32) -> Result<usize> {
        let max_sets = self.next_max_sets.max(required_sets).max(1);
        self.next_max_sets = max_sets.saturating_mul(2).max(1);

        let pool_sizes = L::pool_sizes(max_sets);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);

        // SAFETY: `pool_info` and `pool_sizes` outlive the call.
        let pool = unsafe {
            self.device
                .device
                .create_descriptor_pool(&pool_info, None)?
        };
        let index = self.pools.len();
        self.pools.push(pool);
        self.cursor = index;
        Ok(index)
    }
}

impl<L: SetLayout> Drop for DescriptorAllocator<L> {
    fn drop(&mut self) {
        for pool in self.pools.drain(..) {
            self.device.destroy(Garbage::DescriptorPool(pool));
        }
        self.device
            .destroy(Garbage::DescriptorSetLayout(self.layout));
    }
}
