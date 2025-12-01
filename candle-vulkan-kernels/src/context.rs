//! Vulkan context management
//!
//! Provides device initialization and command submission infrastructure.

use crate::error::{Result, VulkanError};
use ash::vk;
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use vk_mem::AllocatorCreateInfo;

/// Vulkan context holding device, queue, and allocator
pub struct VulkanContext {
    // Order matters for Drop! Allocator must be dropped before device.
    // We use ManuallyDrop to control destruction order.
    pub allocator: ManuallyDrop<vk_mem::Allocator>,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
}

impl VulkanContext {
    /// Create a new Vulkan context with the specified device index
    pub fn new(device_index: usize) -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load().map_err(|_| VulkanError::NoDevices)? };

        let app_name = CString::new("candle-vulkan").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 2, 0))
            .application_name(&app_name);

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        // Select physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        if physical_devices.is_empty() {
            return Err(VulkanError::NoDevices);
        }
        let physical_device = *physical_devices
            .get(device_index)
            .ok_or(VulkanError::InvalidDeviceIndex(device_index))?;

        // Find compute queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or(VulkanError::NoComputeQueue)? as u32;

        // Create logical device
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];

        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        // Create memory allocator
        let allocator_info = AllocatorCreateInfo::new(&instance, &device, physical_device);
        let allocator = unsafe {
            vk_mem::Allocator::new(allocator_info)
                .map_err(|e| VulkanError::Allocator(e.to_string()))?
        };

        Ok(Arc::new(Self {
            allocator: ManuallyDrop::new(allocator),
            command_pool,
            queue,
            queue_family_index,
            device,
            physical_device,
            instance,
            entry,
        }))
    }

    /// Allocate a command buffer
    pub fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    /// Submit a command buffer and wait for completion
    pub fn submit_and_wait(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            self.device.queue_submit(self.queue, &[submit_info], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
        }

        Ok(())
    }

    /// Get device properties
    pub fn device_name(&self) -> String {
        let props = unsafe { self.instance.get_physical_device_properties(self.physical_device) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
        name.to_string_lossy().into_owned()
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Drop allocator FIRST (before device is destroyed)
            ManuallyDrop::drop(&mut self.allocator);

            // Now destroy Vulkan objects
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
