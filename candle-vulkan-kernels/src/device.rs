//! Vulkan device management and context creation
//!
//! This module provides Vulkan device initialization and management following
//! the patterns established in GGML's Vulkan implementation.

use crate::{error::DType, error::Result, error::VulkanError};
use ash::{vk, Device, Entry, Instance};
use gpu_allocator::vulkan::{Allocation, AllocationSizes, AllocatorCreateDesc};
use parking_lot::Mutex;
use std::ffi::{c_char, CString};
use std::sync::Arc;

pub struct VulkanContext {
    /// Vulkan objects (RAII via Drop)
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,

    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,

    pub compute_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,

    pub allocator: Arc<Mutex<Allocator>>,
    pub pipelines: PipelineCache,

    pub device_info: DeviceInfo,
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub subgroup_size: u32,
    pub max_workgroup_size: [u32; 3],
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub api_version: u32,
}

pub struct PipelineCache {
    inner: std::sync::RwLock<std::collections::HashMap<String, vk::Pipeline>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<vk::Pipeline> {
        self.inner.read().unwrap().get(key).copied()
    }

    pub fn insert(&self, key: String, pipeline: vk::Pipeline) {
        self.inner.write().unwrap().insert(key, pipeline);
    }
}

impl VulkanContext {
    pub fn new(device_index: usize) -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let physical_device = Self::select_device(&instance, device_index)?;

        let (device, compute_queue, compute_queue_family) =
            Self::create_device(&instance, physical_device)?;

        let compute_pool = Self::create_command_pool(&device, compute_queue_family)?;
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        let allocator = Self::create_allocator(&instance, &device, physical_device)?;

        let device_info = Self::query_device_info(&instance, physical_device)?;
        let pipelines = PipelineCache::new();

        Ok(Self {
            instance,
            device,
            physical_device,
            compute_queue,
            compute_queue_family,
            compute_pool,
            descriptor_pool,
            allocator: Arc::new(Mutex::new(allocator)),
            pipelines,
            device_info,
        })
    }

    fn create_instance(entry: &Entry) -> Result<Instance> {
        let app_name = CString::new("candle-vulkan-kernels")?;
        let engine_name = CString::new("Candle")?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        let mut extension_names =
            vec![ash::vk::KhrGetPhysicalDeviceProperties2Extension::name().as_ptr()];

        let layer_names = vec![CString::new("VK_LAYER_KHRONOS_validation")?.as_ptr()];

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| VulkanError::AshError(e))?
        };

        Ok(instance)
    }

    fn select_device(instance: &Instance, device_index: usize) -> Result<vk::PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| VulkanError::AshError(e))?;

        if device_index >= physical_devices.len() {
            return Err(VulkanError::InvalidDeviceIndex(device_index));
        }

        Ok(physical_devices[device_index])
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(Device, vk::Queue, u32)> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let compute_queue_family = queue_family_properties
            .iter()
            .position(|qfp| qfp.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or(VulkanError::Message("No compute queue family found".into()))?
            as u32;

        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities);

        let device_extensions = [
            ash::vk::KhrMaintenance1Extension::name().as_ptr(),
            ash::vk::KhrShaderFloat16Int8Extension::name().as_ptr(),
            ash::vk::Khr16bitStorageExtension::name().as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_float64(true)
            .shader_int64(true)
            .shader_float16(true);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_create_info])
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| VulkanError::AshError(e))?
        };

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        Ok((device, compute_queue, compute_queue_family))
    }

    fn create_command_pool(device: &Device, queue_family: u32) -> Result<vk::CommandPool> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(16)
                .build(),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(256)
            .pool_sizes(&pool_sizes);

        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    fn create_allocator(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Allocator> {
        let create_desc = AllocatorCreateDesc {
            instance,
            device,
            physical_device,
            device_name: "Vulkan Device",
            preferred_device_type: gpu_allocator::MemoryLocation::GpuOnly,
            allocation_sizes: gpu_allocator::AllocationSizes::new(),
        };

        Allocator::new(create_desc)
    }

    fn query_device_info(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<DeviceInfo> {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let compute_queue_family = queue_family_properties
            .iter()
            .position(|qfp| qfp.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or(VulkanError::Message("No compute queue family found".into()))?
            as u32;

        let subgroup_properties =
            unsafe { instance.get_physical_device_subgroup_properties(physical_device) };

        let subgroup_size = if !subgroup_properties.is_empty() {
            subgroup_properties[0].subgroup_size
        } else {
            32 // Default subgroup size
        };

        Ok(DeviceInfo {
            name: properties.device_name.to_string_lossy().into_owned(),
            subgroup_size,
            max_workgroup_size: properties.limits.max_workgroup_size,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            driver_version: properties.driver_version,
            api_version: properties.api_version,
        })
    }

    pub fn create_shader_module(&self, shader_code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(shader_code);

        unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    pub fn create_compute_pipeline(
        &self,
        shader_module: vk::ShaderModule,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name("main")
                    .build(),
            )
            .layout(layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .map_err(|(_, e)| VulkanError::AshError(e))?;

        Ok(pipelines[0])
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.compute_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl std::fmt::Debug for VulkanContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanContext {{ device: '{}', subgroup_size: {}, vendor_id: {} }}",
            self.device_info.name, self.device_info.subgroup_size, self.device_info.vendor_id
        )
    }
}
