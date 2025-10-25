//! Vulkan storage buffer management
//!
//! This module provides buffer storage management following GGML patterns
//! for efficient GPU memory allocation and data transfer.

use crate::{error::VulkanError, error::Result, error::DType};
use ash::{vk, Device};
use gpu_allocator::vulkan::{Allocation, AllocationScheme, MemoryLocation, VmaAllocationInfo};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct VulkanStorage {
    /// The actual buffer containing the data.
    buffer: vk::Buffer,
    /// A reference to the device owning this buffer
    device: VulkanDevice,
    /// The count of allocated elements in the buffer
    count: usize,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
    /// GPU memory allocation
    allocation: Arc<Mutex<Option<Allocation>>>,
    /// Host-visible buffer for staging
    host_buffer: Option<vk::Buffer>,
    /// Host memory allocation
    host_allocation: Option<Arc<Mutex<Option<Allocation>>>>,
}

pub struct VulkanDevice {
    pub device: Arc<vk::Device>,
    pub allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
    pub context: super::VulkanContext,
}

impl VulkanStorage {
    pub fn new(
        device: &VulkanDevice,
        count: usize,
        dtype: DType,
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
    ) -> Result<Self> {
        let buffer_size = count as u64 * dtype.size_in_bytes() as u64;

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device.device.create_buffer(&buffer_info, None)
                .map_err(|e| VulkanError::AshError(e))?
        };

        let requirements = unsafe {
            device.device.get_buffer_memory_requirements(buffer)
        };

        let allocation_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Vulkan Buffer",
            requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorOwned,
        };

        let allocation = device.allocator.lock().allocate(&allocation_desc)
            .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

        unsafe {
            device.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            ).map_err(|e| VulkanError::AshError(e))?;
        }

        let (host_buffer, host_allocation) = if memory_location == MemoryLocation::GpuOnly {
            let host_usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
            let host_buffer_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(host_usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let host_buffer = unsafe {
                device.device.create_buffer(&host_buffer_info, None)
                    .map_err(|e| VulkanError::AshError(e))?
            };

            let host_requirements = unsafe {
                device.device.get_buffer_memory_requirements(host_buffer)
            };

            let host_allocation_desc = gpu_allocator::vulkan::AllocationCreateDesc {
                name: "Vulkan Host Buffer",
                requirements: host_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorOwned,
            };

            let host_allocation = device.allocator.lock().allocate(&host_allocation_desc)
                .map_err(|e| VulkanError::MemoryAllocation(e.to_string()))?;

            unsafe {
                device.device.bind_buffer_memory(
                    host_buffer,
                    host_allocation.memory(),
                    host_allocation.offset(),
                ).map_err(|e| VulkanError::AshError(e))?;
            }

            (Some(host_buffer), Some(Arc::new(Mutex::new(Some(host_allocation)))))
        } else {
            (None, None)
        };

        Ok(Self {
            buffer,
            device: device.clone(),
            count,
            dtype,
            allocation: Arc::new(Mutex::new(Some(allocation))),
            host_buffer,
            host_allocation,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn map_memory(&self) -> Result<*mut u8> {
        let allocation = self.allocation.lock().as_ref().ok_or_else(|| {
            VulkanError::BufferMapping("Buffer allocation not found".into())
        })?;

        unsafe {
            allocation.map()
                .map_err(|e| VulkanError::BufferMapping(e.to_string()))
        }
    }

    pub fn unmap_memory(&self) -> Result<()> {
        let allocation = self.allocation.lock().as_ref().ok_or_else(|| {
            VulkanError::BufferMapping("Buffer allocation not found".into())
        })?;

        unsafe {
            allocation.unmap();
        }

        Ok(())
    }

    pub fn copy_from_cpu(&self, data: &[u8]) -> Result<()> {
        let size = data.len() as u64;

        // Map the buffer directly if possible
        let ptr = self.map_memory()?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.unmap_memory()?;
        }

        Ok(())
    }

    pub fn copy_to_cpu(&self) -> Result<Vec<u8>> {
        let size = self.count as u64 * self.dtype.size_in_bytes() as u64;

        let ptr = self.map_memory()?;
        let mut result = vec![0u8; size as usize];

        unsafe {
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size as usize);
            self.unmap_memory()?;
        }

        Ok(result)
    }

    pub async fn copy_from_gpu_async(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        if let (Some(host_buffer), Some(host_allocation)) = (&self.host_buffer, &self.host_allocation) {
            let src_buffer = self.buffer;
            let dst_buffer = *host_buffer;

            let copy_region = vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(self.count as u64 * self.dtype.size_in_bytes() as u64)
                .build();

            unsafe {
                self.device.device.cmd_copy_buffer(
                    command_buffer,
                    src_buffer,
                    dst_buffer,
                    &[copy_region],
                );
            }
        }

        Ok(())
    }

    pub async fn copy_to_gpu_async(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        if let (Some(host_buffer), Some(host_allocation)) = (&self.host_buffer, &self.host_allocation) {
            let src_buffer = *host_buffer;
            let dst_buffer = self.buffer;

            let copy_region = vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(self.count as u64 * self.dtype.size_in_bytes() as u64)
                .build();

            unsafe {
                self.device.device.cmd_copy_buffer(
                    command_buffer,
                    src_buffer,
                    dst_buffer,
                    &[copy_region],
                );
            }
        }

        Ok(())
    }
}

impl Drop for VulkanStorage {
    fn drop(&mut self) {
        unsafe {
            if let Some(allocation) = self.allocation.lock().take() {
                self.device.allocator.lock().free(allocation).unwrap_or_else(|e| {
                    eprintln!("Failed to free allocation: {}", e);
                });
            }

            if let Some(host_allocation) = &self.host_allocation {
                if let Some(allocation) = host_allocation.lock().take() {
                    self.device.allocator.lock().free(allocation).unwrap_or_else(|e| {
                        eprintln!("Failed to free host allocation: {}", e);
                    });
                }
            }

            self.device.device.destroy_buffer(self.buffer, None);

            if let Some(host_buffer) = self.host_buffer {
                self.device.device.destroy_buffer(host_buffer, None);
            }
        }
    }
}

impl Clone for VulkanStorage {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            device: self.device.clone(),
            count: self.count,
            dtype: self.dtype,
            allocation: self.allocation.clone(),
            host_buffer: self.host_buffer,
            host_allocation: self.host_allocation.clone(),
        }
    }
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanStorage {{ buffer: {:?}, count: {}, dtype: {:?} }}",
            self.buffer, self.count, self.dtype
        )
    }
}

impl VulkanDevice {
    pub fn new(context: super::VulkanContext) -> Self {
        Self {
            device: Arc::new(context.device),
            allocator: context.allocator.clone(),
            context,
        }
    }

    pub fn device(&self) -> &vk::Device {
        &self.device
    }

    pub fn context(&self) -> &super::VulkanContext {
        &self.context
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice")
    }
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            context: self.context.clone(),
        }
    }
}
