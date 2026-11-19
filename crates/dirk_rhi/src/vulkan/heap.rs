//! The bindless resource heap: one descriptor set every pipeline binds.

use std::collections::VecDeque;

use ash::vk;
use dirk_shaders::bindless::heap;
use parking_lot::Mutex;

use super::{Check, Device, pipeline::PUSH_STAGES};
use crate::{MAX_PUSH_DATA_SIZE, Result, ViewType};

/// One array of the heap.
struct Array {
    binding: u32,
    kind: vk::DescriptorType,
    capacity: u32,
}

impl Array {
    const fn new(binding: u32, kind: vk::DescriptorType, capacity: u32) -> Self {
        Self {
            binding,
            kind,
            capacity,
        }
    }

    const fn texture(binding: u32) -> Self {
        Self::new(
            binding,
            vk::DescriptorType::SAMPLED_IMAGE,
            heap::TEXTURE_CAPACITY,
        )
    }

    const fn storage_image(binding: u32) -> Self {
        Self::new(
            binding,
            vk::DescriptorType::STORAGE_IMAGE,
            heap::STORAGE_IMAGE_CAPACITY,
        )
    }
}

/// Every array of the heap, indexed by binding.
const ARRAYS: [Array; 10] = [
    Array::texture(heap::TEXTURES_2D),
    Array::texture(heap::TEXTURES_2D_ARRAY),
    Array::texture(heap::TEXTURES_CUBE),
    Array::texture(heap::TEXTURES_CUBE_ARRAY),
    Array::texture(heap::TEXTURES_3D),
    Array::storage_image(heap::STORAGE_IMAGES_2D),
    Array::storage_image(heap::STORAGE_IMAGES_2D_ARRAY),
    Array::storage_image(heap::STORAGE_IMAGES_3D),
    Array::new(
        heap::SAMPLERS,
        vk::DescriptorType::SAMPLER,
        heap::SAMPLER_CAPACITY,
    ),
    Array::new(
        heap::BUFFERS,
        vk::DescriptorType::STORAGE_BUFFER,
        heap::BUFFER_CAPACITY,
    ),
];

// Arrays are looked up by binding.
const _: () = {
    let mut index = 0;
    while index < ARRAYS.len() {
        assert!(ARRAYS[index].binding as usize == index);
        index += 1;
    }
};

/// A slot in one of the heap's arrays.
#[derive(Clone, Copy, Debug)]
pub(super) struct Slot {
    pub(super) binding: u32,
    pub(super) index: u32,
}

/// The descriptor a slot holds.
#[derive(Clone, Copy)]
pub(super) enum Descriptor {
    Image(vk::ImageView),
    Sampler(vk::Sampler),
    Buffer(vk::Buffer),
}

/// Free slots of each array.
#[derive(Default)]
struct Slots {
    /// Slots never used yet start here.
    next: [u32; ARRAYS.len()],
    /// Slots freed since, reused first, oldest first, so a stale index
    /// reads the null descriptor for as long as possible.
    free: [VecDeque<u32>; ARRAYS.len()],
}

/// The heap's descriptor set and the pipeline layout every pipeline shares.
pub(super) struct Heap {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pub(super) pipeline_layout: vk::PipelineLayout,
    /// Fills sampler slots that hold no sampler, since samplers cannot be
    /// null.
    default_sampler: vk::Sampler,
    /// Locked to assign, write and free slots, which also synchronizes host
    /// access to the set.
    slots: Mutex<Slots>,
}

impl Heap {
    /// Whether a device with update-after-bind descriptor limits
    /// `properties` can hold the heap in one set read by one stage.
    pub(super) fn fits(properties: &vk::PhysicalDeviceVulkan12Properties<'_>) -> bool {
        let capacity = |kind| {
            ARRAYS
                .iter()
                .filter(|array| array.kind == kind)
                .map(|array| array.capacity)
                .sum::<u32>()
        };
        let p = properties;
        let total: u32 = ARRAYS.iter().map(|array| array.capacity).sum();
        [
            (
                vk::DescriptorType::SAMPLED_IMAGE,
                p.max_descriptor_set_update_after_bind_sampled_images,
                p.max_per_stage_descriptor_update_after_bind_sampled_images,
            ),
            (
                vk::DescriptorType::STORAGE_IMAGE,
                p.max_descriptor_set_update_after_bind_storage_images,
                p.max_per_stage_descriptor_update_after_bind_storage_images,
            ),
            (
                vk::DescriptorType::SAMPLER,
                p.max_descriptor_set_update_after_bind_samplers,
                p.max_per_stage_descriptor_update_after_bind_samplers,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER,
                p.max_descriptor_set_update_after_bind_storage_buffers,
                p.max_per_stage_descriptor_update_after_bind_storage_buffers,
            ),
        ]
        .into_iter()
        .all(|(kind, set, stage)| capacity(kind) <= set.min(stage))
            && total <= p.max_per_stage_update_after_bind_resources
            && total <= p.max_update_after_bind_descriptors_in_all_pools
    }

    /// Creates the heap with every slot empty: null descriptors, or the
    /// default sampler.
    pub(super) fn new(device: &ash::Device) -> Result<Self> {
        let flags = [vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
            ARRAYS.len()];
        let bindings = ARRAYS.map(|array| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(array.binding)
                .descriptor_type(array.kind)
                .descriptor_count(array.capacity)
                .stage_flags(vk::ShaderStageFlags::ALL)
        });
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut binding_flags);
        // The heap is destroyed with whatever was created on failure.
        let mut heap = Self {
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            default_sampler: vk::Sampler::null(),
            slots: Mutex::default(),
        };
        let result = heap.create(device, &info);
        if result.is_err() {
            // SAFETY: nothing uses the partly created heap.
            unsafe { heap.destroy(device) };
        }
        result.map(|()| heap)
    }

    /// Creates the heap's objects from the set layout `info`, then fills
    /// every slot.
    fn create(
        &mut self,
        device: &ash::Device,
        info: &vk::DescriptorSetLayoutCreateInfo<'_>,
    ) -> Result<()> {
        // SAFETY: `info` is valid for the device, which supports every
        // flag and count, as adapter selection checked.
        self.set_layout = unsafe { device.create_descriptor_set_layout(info, None) }
            .check("vkCreateDescriptorSetLayout")?;
        let sizes = ARRAYS.map(|array| vk::DescriptorPoolSize {
            ty: array.kind,
            descriptor_count: array.capacity,
        });
        let pool = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `pool` is valid for the device.
        self.pool = unsafe { device.create_descriptor_pool(&pool, None) }
            .check("vkCreateDescriptorPool")?;
        let layouts = [self.set_layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: the pool has room for exactly this set.
        self.set = unsafe { device.allocate_descriptor_sets(&allocate) }
            .check("vkAllocateDescriptorSets")?[0];
        let push = [vk::PushConstantRange {
            stage_flags: PUSH_STAGES,
            offset: 0,
            size: MAX_PUSH_DATA_SIZE,
        }];
        let layout = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push);
        // SAFETY: `layout` is valid for the device, whose push data limit is
        // at least 128 bytes.
        self.pipeline_layout = unsafe { device.create_pipeline_layout(&layout, None) }
            .check("vkCreatePipelineLayout")?;
        // SAFETY: the default sampler info is valid for every device.
        self.default_sampler =
            unsafe { device.create_sampler(&vk::SamplerCreateInfo::default(), None) }
                .check("vkCreateSampler")?;
        for array in &ARRAYS {
            let empty: Vec<_> = (0..array.capacity)
                .map(|index| Slot {
                    binding: array.binding,
                    index,
                })
                .collect();
            self.write(device, &empty, None);
        }
        Ok(())
    }

    /// Writes `descriptor`, or the empty one, to `slots` of one array.
    fn write(&self, device: &ash::Device, slots: &[Slot], descriptor: Option<Descriptor>) {
        let Some(first) = slots.first() else { return };
        let array = &ARRAYS[first.binding as usize];
        let image = |view, layout| vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: layout,
        };
        let (images, buffers): (Vec<_>, Vec<_>) = match (array.kind, descriptor) {
            (vk::DescriptorType::SAMPLER, sampler) => {
                let sampler = match sampler {
                    Some(Descriptor::Sampler(sampler)) => sampler,
                    _ => self.default_sampler,
                };
                let info = vk::DescriptorImageInfo::default().sampler(sampler);
                (vec![info; slots.len()], Vec::new())
            }
            (vk::DescriptorType::STORAGE_BUFFER, buffer) => {
                let buffer = match buffer {
                    Some(Descriptor::Buffer(buffer)) => buffer,
                    _ => vk::Buffer::null(),
                };
                let info = vk::DescriptorBufferInfo {
                    buffer,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                };
                (Vec::new(), vec![info; slots.len()])
            }
            (kind, view) => {
                let view = match view {
                    Some(Descriptor::Image(view)) => view,
                    _ => vk::ImageView::null(),
                };
                let layout = if kind == vk::DescriptorType::STORAGE_IMAGE {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::READ_ONLY_OPTIMAL
                };
                (vec![image(view, layout); slots.len()], Vec::new())
            }
        };
        // Slots are written in runs of consecutive indices, which is how
        // every caller passes them.
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(first.binding)
            .dst_array_element(first.index)
            .descriptor_type(array.kind);
        let write = if buffers.is_empty() {
            write.image_info(&images)
        } else {
            write.buffer_info(&buffers)
        };
        // SAFETY: the caller holds the slot lock or the set is not shared
        // yet; descriptors are valid or null, which nullDescriptor allows, and
        // no pending work uses these slots.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Destroys the heap's objects.
    ///
    /// # Safety
    ///
    /// No pending work uses the heap.
    pub(super) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller guarantees nothing uses the heap; null handles
        // are ignored.
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.default_sampler, None);
        }
    }
}

impl Device {
    /// Assigns `descriptor` a slot in the array at `binding`.
    pub(super) fn add_to_heap(&self, binding: u32, descriptor: Descriptor) -> Result<Slot> {
        let heap = &self.heap;
        let mut slots = heap.slots.lock();
        let array = &ARRAYS[binding as usize];
        let index = match slots.free[binding as usize].pop_front() {
            Some(index) => index,
            None if slots.next[binding as usize] < array.capacity => {
                let index = slots.next[binding as usize];
                slots.next[binding as usize] += 1;
                index
            }
            None => {
                return Err(crate::Error::invalid(
                    "bindless heap",
                    &format!("binding {binding}"),
                    format!("all {} slots are in use", array.capacity),
                ));
            }
        };
        let slot = Slot { binding, index };
        heap.write(&self.raw, &[slot], Some(descriptor));
        Ok(slot)
    }

    /// Empties `slot` and makes it available again. Only call once no work
    /// can use the descriptor it held.
    pub(super) fn remove_from_heap(&self, slot: Slot) {
        let mut slots = self.heap.slots.lock();
        self.heap.write(&self.raw, &[slot], None);
        slots.free[slot.binding as usize].push_back(slot.index);
    }

    /// Binds the heap for graphics work recorded in `commands`.
    pub(super) fn bind_heap(&self, commands: vk::CommandBuffer) {
        // SAFETY: `commands` is recording and the set outlives its work.
        unsafe {
            self.raw.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::GRAPHICS,
                self.heap.pipeline_layout,
                0,
                &[self.heap.set],
                &[],
            );
        }
    }
}

impl ViewType {
    /// The heap arrays images viewed this way are sampled and stored through,
    /// if any.
    pub(super) const fn heap_bindings(self) -> (u32, Option<u32>) {
        match self {
            Self::D2 => (heap::TEXTURES_2D, Some(heap::STORAGE_IMAGES_2D)),
            Self::D2Array => (heap::TEXTURES_2D_ARRAY, Some(heap::STORAGE_IMAGES_2D_ARRAY)),
            Self::Cube => (heap::TEXTURES_CUBE, None),
            Self::CubeArray => (heap::TEXTURES_CUBE_ARRAY, None),
            Self::D3 => (heap::TEXTURES_3D, Some(heap::STORAGE_IMAGES_3D)),
        }
    }
}
