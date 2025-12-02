use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Result, Shape};
use candle_vulkan_kernels::{Kernels, VulkanContext};
use std::sync::{atomic, Arc, Mutex};
use vk_mem::Alloc;

use super::{VulkanBuffer, VulkanStorage};

/// Unique identifier for Vulkan devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct VulkanDevice {
    /// Unique identifier
    id: DeviceId,
    /// The GPU ordinal (index)
    ordinal: usize,
    /// The Vulkan context from candle-vulkan-kernels
    context: Arc<VulkanContext>,
    /// The kernel manager
    kernels: Arc<Kernels>,
    /// Seed for random number generation
    seed: Arc<Mutex<u64>>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice({:?})", self.id)
    }
}

impl VulkanDevice {
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn kernels(&self) -> &Arc<Kernels> {
        &self.kernels
    }

    pub fn allocator(&self) -> &vk_mem::Allocator {
        &self.context.allocator
    }

    /// Create a new buffer with the given element count and dtype
    pub fn new_buffer(&self, element_count: usize, dtype: DType) -> Result<Arc<VulkanBuffer>> {
        let size = (element_count * dtype.size_in_bytes()) as u64;

        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::TRANSFER_SRC
                    | ash::vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            self.context
                .allocator
                .create_buffer(&buffer_info, &alloc_info)
                .map_err(|e| crate::Error::Msg(format!("Failed to create buffer: {:?}", e)))?
        };

        Ok(Arc::new(VulkanBuffer::new(buffer, allocation, self.context.clone())))
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = VulkanContext::new(ordinal)
            .map_err(|e| crate::Error::Msg(format!("Failed to create Vulkan context: {:?}", e)))?;
        let kernels = Arc::new(
            Kernels::new(context.clone())
                .map_err(|e| crate::Error::Msg(format!("Failed to create kernels: {:?}", e)))?,
        );

        Ok(Self {
            id: DeviceId::new(),
            ordinal,
            context,
            kernels,
            seed: Arc::new(Mutex::new(299792458)), // Speed of light as default seed
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan { gpu_id: self.ordinal }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let buffer = self.new_buffer(elem_count, dtype)?;

        // Zero the buffer
        let allocation_info = self.allocator().get_allocation_info(buffer.allocation());
        if !allocation_info.mapped_data.is_null() {
            unsafe {
                std::ptr::write_bytes(
                    allocation_info.mapped_data,
                    0,
                    elem_count * dtype.size_in_bytes(),
                );
            }
        }

        Ok(VulkanStorage::new(buffer, self.clone(), elem_count, dtype))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let buffer = self.new_buffer(elem_count, dtype)?;
        Ok(VulkanStorage::new(buffer, self.clone(), elem_count, dtype))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let elem_count = data.len();
        let dtype = T::DTYPE;
        let buffer = self.new_buffer(elem_count, dtype)?;

        // Copy data to mapped memory
        let allocation_info = self.allocator().get_allocation_info(buffer.allocation());
        if allocation_info.mapped_data.is_null() {
            return Err(crate::Error::Msg("Buffer not mapped".to_string()));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                allocation_info.mapped_data as *mut u8,
                elem_count * dtype.size_in_bytes(),
            );
        }

        Ok(VulkanStorage::new(buffer, self.clone(), elem_count, dtype))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        match storage {
            CpuStorage::U8(data) => self.storage_from_slice(data),
            CpuStorage::U32(data) => self.storage_from_slice(data),
            CpuStorage::I64(data) => self.storage_from_slice(data),
            CpuStorage::F16(data) => self.storage_from_slice(data),
            CpuStorage::BF16(data) => self.storage_from_slice(data),
            CpuStorage::F32(data) => self.storage_from_slice(data),
            CpuStorage::F64(data) => self.storage_from_slice(data),
            CpuStorage::F8E4M3(data) => self.storage_from_slice(data),
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        // For now, generate on CPU and upload
        let elem_count = shape.elem_count();
        let seed = {
            let mut s = self.seed.lock().unwrap();
            *s = s.wrapping_add(1);
            *s
        };

        use rand::SeedableRng;
        use rand_distr::{Distribution, Uniform};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        match dtype {
            DType::F32 => {
                let dist = Uniform::new(lo as f32, hi as f32)
                    .map_err(|e| crate::Error::Msg(format!("Invalid uniform distribution: {}", e)))?;
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| dist.sample(&mut rng))
                    .collect();
                self.storage_from_slice(&data)
            }
            DType::F64 => {
                let dist = Uniform::new(lo, hi)
                    .map_err(|e| crate::Error::Msg(format!("Invalid uniform distribution: {}", e)))?;
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| dist.sample(&mut rng))
                    .collect();
                self.storage_from_slice(&data)
            }
            _ => crate::bail!("rand_uniform not implemented for {:?}", dtype),
        }
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        // For now, generate on CPU and upload
        let elem_count = shape.elem_count();
        let seed = {
            let mut s = self.seed.lock().unwrap();
            *s = s.wrapping_add(1);
            *s
        };

        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        match dtype {
            DType::F32 => {
                let normal = Normal::new(mean as f32, std as f32)
                    .map_err(|e| crate::Error::Msg(format!("Invalid normal distribution: {}", e)))?;
                let data: Vec<f32> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                self.storage_from_slice(&data)
            }
            DType::F64 => {
                let normal = Normal::new(mean, std)
                    .map_err(|e| crate::Error::Msg(format!("Invalid normal distribution: {}", e)))?;
                let data: Vec<f64> = (0..elem_count).map(|_| normal.sample(&mut rng)).collect();
                self.storage_from_slice(&data)
            }
            _ => crate::bail!("rand_normal not implemented for {:?}", dtype),
        }
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut s = self.seed.lock().unwrap();
        *s = seed;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.context
                .device
                .device_wait_idle()
                .map_err(|e| crate::Error::Msg(format!("Failed to synchronize: {:?}", e)))?;
        }
        // Reset descriptor set index for reuse (llama.cpp style)
        self.kernels.reset_descriptor_sets();
        Ok(())
    }
}
