use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Layout, Result, Shape};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use half::{bf16, f16};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex, RwLock};

use super::{VulkanBuffer, VulkanError, VulkanStorage, VulkanStorageSlice, WrapErr};

/// Unique identifier for Vulkan devices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// Pipeline cache for compiled shaders
pub struct PipelineCache {
    pipelines: HashMap<String, vk::Pipeline>,
    pipeline_layouts: HashMap<String, vk::PipelineLayout>,
    descriptor_set_layouts: HashMap<String, vk::DescriptorSetLayout>,
    shader_modules: HashMap<String, vk::ShaderModule>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            pipeline_layouts: HashMap::new(),
            descriptor_set_layouts: HashMap::new(),
            shader_modules: HashMap::new(),
        }
    }
}

/// Main Vulkan device structure
#[derive(Clone)]
pub struct VulkanDevice {
    id: DeviceId,
    entry: Arc<ash::Entry>,
    instance: Arc<ash::Instance>,
    physical_device: vk::PhysicalDevice,
    device: Arc<ash::Device>,
    compute_queue: Arc<Mutex<vk::Queue>>,
    compute_queue_family: u32,
    command_pool: Arc<Mutex<vk::CommandPool>>,
    descriptor_pool: Arc<Mutex<vk::DescriptorPool>>,
    allocator: Arc<Mutex<Allocator>>,
    pipeline_cache: Arc<RwLock<PipelineCache>>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice({:?})", self.id)
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            // Only clean up if this is the last reference
            if Arc::strong_count(&self.device) == 1 {
                self.device.device_wait_idle().ok();

                // Clean up pipelines
                if let Ok(cache) = self.pipeline_cache.write() {
                    for pipeline in cache.pipelines.values() {
                        self.device.destroy_pipeline(*pipeline, None);
                    }
                    for layout in cache.pipeline_layouts.values() {
                        self.device.destroy_pipeline_layout(*layout, None);
                    }
                    for layout in cache.descriptor_set_layouts.values() {
                        self.device.destroy_descriptor_set_layout(*layout, None);
                    }
                    for module in cache.shader_modules.values() {
                        self.device.destroy_shader_module(*module, None);
                    }
                }

                // Clean up descriptor pool
                if let Ok(pool) = self.descriptor_pool.lock() {
                    self.device.destroy_descriptor_pool(*pool, None);
                }

                // Clean up command pool
                if let Ok(pool) = self.command_pool.lock() {
                    self.device.destroy_command_pool(*pool, None);
                }
            }
        }
    }
}

impl VulkanDevice {
    /// Create a new Vulkan device
    pub fn new(ordinal: usize) -> Result<Self> {
        unsafe {
            // Create Vulkan entry
            let entry = ash::Entry::linked();

            // Create instance
            let app_info = vk::ApplicationInfo::default()
                .application_name(CStr::from_bytes_with_nul_unchecked(b"Candle\0"))
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul_unchecked(b"Candle\0"))
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_2);

            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

            let instance = entry.create_instance(&create_info, None).w()?;

            // Select physical device
            let physical_devices = instance.enumerate_physical_devices().w()?;
            if physical_devices.is_empty() {
                return Err(VulkanError::Message("No Vulkan devices found".to_string()).into());
            }

            let physical_device = *physical_devices.get(ordinal).ok_or_else(|| {
                VulkanError::Message(format!("Vulkan device {} not found", ordinal))
            })?;

            // Find compute queue family
            let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
            let compute_queue_family = queue_families
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| VulkanError::Message("No compute queue family found".to_string()))?;

            // Create logical device
            let queue_priorities = [1.0];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_queue_family)
                .queue_priorities(&queue_priorities);

            let device_create_info =
                vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_create_info));

            let device = instance.create_device(physical_device, &device_create_info, None).w()?;

            // Get compute queue
            let compute_queue = device.get_device_queue(compute_queue_family, 0);

            // Create command pool
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(compute_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device.create_command_pool(&command_pool_info, None).w()?;

            // Create descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1000,
                },
            ];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1000)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

            let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None).w()?;

            // Create memory allocator
            let allocator = Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .w()?;

            Ok(Self {
                id: DeviceId::new(),
                entry: Arc::new(entry),
                instance: Arc::new(instance),
                physical_device,
                device: Arc::new(device),
                compute_queue: Arc::new(Mutex::new(compute_queue)),
                compute_queue_family,
                command_pool: Arc::new(Mutex::new(command_pool)),
                descriptor_pool: Arc::new(Mutex::new(descriptor_pool)),
                allocator: Arc::new(Mutex::new(allocator)),
                pipeline_cache: Arc::new(RwLock::new(PipelineCache::new())),
            })
        }
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Allocate a buffer with the given size and memory location
    pub fn alloc_buffer<T>(
        &self,
        len: usize,
        location: MemoryLocation,
    ) -> Result<Arc<VulkanBuffer<T>>> {
        let size = len * std::mem::size_of::<T>();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None).w()? };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "candle_buffer",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .w()?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .w()?;
        }

        Ok(Arc::new(VulkanBuffer {
            buffer,
            allocation,
            len,
            device: self.clone(),
            _phantom: std::marker::PhantomData,
        }))
    }

    /// Write data to a buffer
    pub fn write_buffer<T: Copy>(&self, buffer: &VulkanBuffer<T>, data: &[T]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            let ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    /// Read data from a buffer
    pub fn read_buffer<T: Copy>(&self, buffer: &VulkanBuffer<T>) -> Result<Vec<T>> {
        let mut result = vec![T::default(); buffer.len];
        unsafe {
            let ptr = buffer.allocation.mapped_ptr().unwrap().as_ptr() as *const T;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), buffer.len);
        }
        Ok(result)
    }

    /// Get or create a shader module
    pub fn get_or_create_shader_module(&self, name: &str) -> Result<vk::ShaderModule> {
        {
            let cache = self.pipeline_cache.read().unwrap();
            if let Some(module) = cache.shader_modules.get(name) {
                return Ok(*module);
            }
        }

        let spirv = candle_vulkan_kernels::get_shader(name).w()?;
        let create_info = vk::ShaderModuleCreateInfo::default()
            .code(unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) });

        let module = unsafe { self.device.create_shader_module(&create_info, None).w()? };

        let mut cache = self.pipeline_cache.write().unwrap();
        cache.shader_modules.insert(name.to_string(), module);

        Ok(module)
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        // TODO: Implement random number generation
        Ok(())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: 0, // TODO: Get actual GPU ID
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let buf = self.alloc_buffer::<u8>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::U8(buf)
            }
            DType::U32 => {
                let buf = self.alloc_buffer::<u32>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::U32(buf)
            }
            DType::I64 => {
                let buf = self.alloc_buffer::<i64>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::I64(buf)
            }
            DType::BF16 => {
                let buf = self.alloc_buffer::<bf16>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::BF16(buf)
            }
            DType::F16 => {
                let buf = self.alloc_buffer::<f16>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::F16(buf)
            }
            DType::F32 => {
                let buf = self.alloc_buffer::<f32>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::F32(buf)
            }
            DType::F64 => {
                let buf = self.alloc_buffer::<f64>(elem_count, MemoryLocation::GpuOnly)?;
                VulkanStorageSlice::F64(buf)
            }
            dtype => {
                return Err(VulkanError::UnsupportedDtype {
                    dtype,
                    op: "zeros_impl",
                }
                .into())
            }
        };

        Ok(VulkanStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, _shape: &Shape, _dtype: DType, _lo: f64, _up: f64) -> Result<VulkanStorage> {
        todo!("rand_uniform not yet implemented for Vulkan")
    }

    fn rand_normal(&self, _shape: &Shape, _dtype: DType, _mean: f64, _std: f64) -> Result<VulkanStorage> {
        todo!("rand_normal not yet implemented for Vulkan")
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage> {
        self.zeros_impl(shape, dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<VulkanStorage> {
        use crate::CpuStorageRef;

        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(data) => {
                let buf = self.alloc_buffer::<u8>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::U8(buf)
            }
            CpuStorageRef::U32(data) => {
                let buf = self.alloc_buffer::<u32>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::U32(buf)
            }
            CpuStorageRef::I64(data) => {
                let buf = self.alloc_buffer::<i64>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::I64(buf)
            }
            CpuStorageRef::BF16(data) => {
                let buf = self.alloc_buffer::<bf16>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::BF16(buf)
            }
            CpuStorageRef::F16(data) => {
                let buf = self.alloc_buffer::<f16>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F16(buf)
            }
            CpuStorageRef::F32(data) => {
                let buf = self.alloc_buffer::<f32>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F32(buf)
            }
            CpuStorageRef::F64(data) => {
                let buf = self.alloc_buffer::<f64>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F64(buf)
            }
            _ => {
                return Err(VulkanError::Message("Unsupported dtype".to_string()).into());
            }
        };

        Ok(VulkanStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<VulkanStorage> {
        let slice = match storage {
            CpuStorage::U8(data) => {
                let buf = self.alloc_buffer::<u8>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::U8(buf)
            }
            CpuStorage::U32(data) => {
                let buf = self.alloc_buffer::<u32>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::U32(buf)
            }
            CpuStorage::I64(data) => {
                let buf = self.alloc_buffer::<i64>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::I64(buf)
            }
            CpuStorage::BF16(data) => {
                let buf = self.alloc_buffer::<bf16>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::BF16(buf)
            }
            CpuStorage::F16(data) => {
                let buf = self.alloc_buffer::<f16>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F16(buf)
            }
            CpuStorage::F32(data) => {
                let buf = self.alloc_buffer::<f32>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F32(buf)
            }
            CpuStorage::F64(data) => {
                let buf = self.alloc_buffer::<f64>(data.len(), MemoryLocation::CpuToGpu)?;
                self.write_buffer(&buf, data)?;
                VulkanStorageSlice::F64(buf)
            }
            _ => {
                return Err(VulkanError::Message("Unsupported dtype".to_string()).into());
            }
        };

        Ok(VulkanStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<VulkanStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.device.device_wait_idle().w()?;
        }
        Ok(())
    }
}
