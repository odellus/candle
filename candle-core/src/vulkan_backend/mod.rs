mod device;
mod error;
mod utils;

pub use device::{DeviceId, VulkanDevice};
pub use error::{VulkanError, WrapErr};

use crate::{CpuStorage, DType, Layout, Result, Shape};
use half::{bf16, f16};

pub use candle_vulkan_kernels as kernels;
use std::sync::Arc;

/// Vulkan buffer wrapper for different data types
pub struct VulkanBuffer<T> {
    pub buffer: ash::vk::Buffer,
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub len: usize,
    pub device: VulkanDevice,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for VulkanBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("len", &self.len)
            .field("buffer", &self.buffer)
            .finish()
    }
}

/// Vulkan storage slice enum for different data types
#[derive(Debug, Clone)]
pub enum VulkanStorageSlice {
    U8(Arc<VulkanBuffer<u8>>),
    U32(Arc<VulkanBuffer<u32>>),
    I64(Arc<VulkanBuffer<i64>>),
    BF16(Arc<VulkanBuffer<bf16>>),
    F16(Arc<VulkanBuffer<f16>>),
    F32(Arc<VulkanBuffer<f32>>),
    F64(Arc<VulkanBuffer<f64>>),
}

impl VulkanStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn device(&self) -> &VulkanDevice {
        match self {
            Self::U8(buf) => &buf.device,
            Self::U32(buf) => &buf.device,
            Self::I64(buf) => &buf.device,
            Self::BF16(buf) => &buf.device,
            Self::F16(buf) => &buf.device,
            Self::F32(buf) => &buf.device,
            Self::F64(buf) => &buf.device,
        }
    }
}

/// Main Vulkan storage structure
#[derive(Debug, Clone)]
pub struct VulkanStorage {
    pub slice: VulkanStorageSlice,
    pub device: VulkanDevice,
}

impl VulkanStorage {
    pub fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Convert Vulkan storage to CPU storage
    pub fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            VulkanStorageSlice::U8(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::U8(data))
            }
            VulkanStorageSlice::U32(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::U32(data))
            }
            VulkanStorageSlice::I64(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::I64(data))
            }
            VulkanStorageSlice::BF16(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::BF16(data))
            }
            VulkanStorageSlice::F16(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::F16(data))
            }
            VulkanStorageSlice::F32(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::F32(data))
            }
            VulkanStorageSlice::F64(buf) => {
                let data = self.device.read_buffer(buf)?;
                Ok(CpuStorage::F64(data))
            }
        }
    }
}

// Macro to implement binary operations
macro_rules! impl_binary_op {
    ($trait_name:ident, $method:ident, $op:expr) => {
        impl utils::$trait_name for VulkanStorageSlice {
            fn $method(
                &self,
                rhs: &Self,
                lhs_layout: &Layout,
                rhs_layout: &Layout,
            ) -> Result<Self> {
                // Implementation will be added in the next step
                todo!("Binary operation implementation")
            }
        }
    };
}

// Implement BackendStorage trait (to be completed)
// This will be implemented in the next iteration
