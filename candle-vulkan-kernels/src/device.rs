//! Vulkan device management and context creation
//!
//! This module provides Vulkan device initialization and management following
//! the patterns established in GGML's Vulkan implementation.

use crate::{error::Result, error::VulkanError};
use ash::{vk, Entry, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use std::ffi::CString;
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

#[derive(Debug, Clone)]
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
            .api_version(vk::API_VERSION_1_2)
            .build();

        let extension_names = vec![
            ash::vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            ash::vk::KhrMaintenance1Fn::name().as_ptr(),
            ash::vk::KhrShaderFloat16Int8Fn::name().as_ptr(),
            ash::vk::Khr16bitStorageFn::name().as_ptr(),
            ash::vk::KhrBufferDeviceAddressFn::name().as_ptr(),
        ];

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .build();

        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }

    fn select_device(instance: &Instance, device_index: usize) -> Result<vk::PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err(VulkanError::NoVulkanDevices);
        }

        let physical_device = physical_devices
            .get(device_index)
            .copied()
            .ok_or(VulkanError::InvalidDeviceIndex)?;

        Ok(physical_device)
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(Device, vk::Queue, u32)> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let compute_queue_family = queue_family_properties
            .iter()
            .enumerate()
            .find_map(|(i, props)| {
                if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .ok_or(VulkanError::NoComputeQueueFamily)?;

        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priorities)
            .build();

        let device_extensions = [
            ash::vk::KhrMaintenance1Fn::name().as_ptr(),
            ash::vk::KhrShaderFloat16Int8Fn::name().as_ptr(),
            ash::vk::Khr16bitStorageFn::name().as_ptr(),
            ash::vk::KhrBufferDeviceAddressFn::name().as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_float64(true)
            .shader_int64(true)
            .shader_float16(true)
            .build();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_create_info])
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features)
            .build();

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        Ok((device, compute_queue, compute_queue_family))
    }

    fn create_command_pool(device: &Device, queue_family: u32) -> Result<vk::CommandPool> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family)
            .build();

        unsafe { Ok(device.create_command_pool(&command_pool_info, None)?) }
    }

    fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(32)
                .build(),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(256)
            .pool_sizes(&pool_sizes)
            .build();

        unsafe { Ok(device.create_descriptor_pool(&pool_info, None)?) }
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
            debug_settings: Default::default(),
            buffer_device_address: true,
        };

        Allocator::new(&create_desc)
    }

    fn query_device_info(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<DeviceInfo> {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        // Get subgroup size
        let mut subgroup_size_properties = vk::PhysicalDeviceSubgroupProperties {
            s_type: vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
            p_next: std::ptr::null_mut(),
            subgroup_size: 0,
            supported_operations: vk::SubgroupFeatureFlags::empty(),
            ..Default::default()
        };

        let mut properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut subgroup_size_properties)
            .build();

        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };

        Ok(DeviceInfo {
            name: std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned(),
            subgroup_size: subgroup_size_properties.subgroup_size,
            max_workgroup_size: properties.limits.max_workgroup_size,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            driver_version: properties.driver_version,
            api_version: properties.api_version,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.compute_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub type Device = ash::Device;
