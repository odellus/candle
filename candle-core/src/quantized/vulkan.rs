//! Vulkan backend for quantized tensors
use super::{GgmlDType, QStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{DType, Result, VulkanDevice, VulkanStorage};
use crate::vulkan_backend::VulkanBuffer;
use std::sync::Arc;

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
    buffer: Arc<VulkanBuffer>,
}

impl QVulkanStorage {
    pub fn new(dtype: GgmlDType, device: VulkanDevice, buffer: Arc<VulkanBuffer>) -> Self {
        Self { dtype, device, buffer }
    }

    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = elem_count * dtype.type_size() / dtype.block_size();

        // Create a buffer large enough for the quantized data
        // We use F32 dtype for size calculation since we're working with raw bytes
        let num_f32_elements = (size_in_bytes + 3) / 4; // Round up to f32 boundary
        let buffer = device.new_buffer(num_f32_elements, DType::F32)?;

        // Zero the buffer
        let allocation_info = device.allocator().get_allocation_info(buffer.allocation());
        if !allocation_info.mapped_data.is_null() {
            unsafe {
                std::ptr::write_bytes(allocation_info.mapped_data, 0, size_in_bytes);
            }
        }

        Ok(Self {
            dtype,
            device: device.clone(),
            buffer,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Arc<VulkanBuffer> {
        &self.buffer
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        let allocation_info = self.device.allocator().get_allocation_info(self.buffer.allocation());
        allocation_info.size as usize
    }

    /// Dequantize to f32 storage
    ///
    /// For now, we copy to CPU, dequantize there, and copy back.
    /// TODO: Use GPU dequantization shaders for Q4_0 and Q8_0
    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        use crate::quantized::k_quants::GgmlType;

        // Sync and read buffer to CPU
        self.device.synchronize()?;

        let allocation_info = self.device.allocator().get_allocation_info(self.buffer.allocation());
        if allocation_info.mapped_data.is_null() {
            crate::bail!("QVulkanStorage buffer not mapped");
        }

        let block_len = elem_count / self.dtype.block_size();
        let mut out = vec![0.0f32; elem_count];

        // Read quantized data and dequantize on CPU
        match self.dtype {
            GgmlDType::F32 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const f32,
                        elem_count,
                    )
                };
                out.copy_from_slice(data);
            }
            GgmlDType::F16 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const half::f16,
                        elem_count,
                    )
                };
                half::f16::to_float(data, &mut out);
            }
            GgmlDType::BF16 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const half::bf16,
                        elem_count,
                    )
                };
                half::bf16::to_float(data, &mut out);
            }
            GgmlDType::Q4_0 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ4_0,
                        block_len,
                    )
                };
                crate::quantized::BlockQ4_0::to_float(data, &mut out);
            }
            GgmlDType::Q4_1 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ4_1,
                        block_len,
                    )
                };
                crate::quantized::BlockQ4_1::to_float(data, &mut out);
            }
            GgmlDType::Q5_0 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ5_0,
                        block_len,
                    )
                };
                crate::quantized::BlockQ5_0::to_float(data, &mut out);
            }
            GgmlDType::Q5_1 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ5_1,
                        block_len,
                    )
                };
                crate::quantized::BlockQ5_1::to_float(data, &mut out);
            }
            GgmlDType::Q8_0 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ8_0,
                        block_len,
                    )
                };
                crate::quantized::BlockQ8_0::to_float(data, &mut out);
            }
            GgmlDType::Q8_1 => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ8_1,
                        block_len,
                    )
                };
                crate::quantized::BlockQ8_1::to_float(data, &mut out);
            }
            GgmlDType::Q2K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ2K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ2K::to_float(data, &mut out);
            }
            GgmlDType::Q3K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ3K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ3K::to_float(data, &mut out);
            }
            GgmlDType::Q4K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ4K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ4K::to_float(data, &mut out);
            }
            GgmlDType::Q5K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ5K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ5K::to_float(data, &mut out);
            }
            GgmlDType::Q6K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ6K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ6K::to_float(data, &mut out);
            }
            GgmlDType::Q8K => {
                let data = unsafe {
                    std::slice::from_raw_parts(
                        allocation_info.mapped_data as *const crate::quantized::BlockQ8K,
                        block_len,
                    )
                };
                crate::quantized::BlockQ8K::to_float(data, &mut out);
            }
        }

        // Upload dequantized f32 data back to GPU
        let output_buffer = self.device.new_buffer(elem_count, DType::F32)?;
        let out_alloc_info = self.device.allocator().get_allocation_info(output_buffer.allocation());
        if out_alloc_info.mapped_data.is_null() {
            crate::bail!("Output buffer not mapped");
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                out.as_ptr() as *const u8,
                out_alloc_info.mapped_data as *mut u8,
                elem_count * std::mem::size_of::<f32>(),
            );
        }

        Ok(VulkanStorage::new(
            output_buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    /// Quantize from f32 VulkanStorage
    pub fn quantize(&mut self, src: &VulkanStorage) -> Result<()> {
        // Quantization happens on CPU for now (same as Metal)
        let src_data = src.to_cpu_storage()?;
        let src_f32 = match src_data {
            crate::CpuStorage::F32(data) => data,
            _ => crate::bail!("Expected F32 storage for quantization"),
        };

        let elem_count = src_f32.len();
        let src_storage = crate::Storage::Cpu(crate::CpuStorage::F32(src_f32));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src_storage)?;

        // Get the quantized data and upload to GPU
        let quantized_data = qcpu_storage.data()?;
        let size_in_bytes = quantized_data.len();

        // Create new buffer if needed
        let num_f32_elements = (size_in_bytes + 3) / 4;
        let new_buffer = self.device.new_buffer(num_f32_elements, DType::F32)?;

        let alloc_info = self.device.allocator().get_allocation_info(new_buffer.allocation());
        if alloc_info.mapped_data.is_null() {
            crate::bail!("Buffer not mapped");
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                quantized_data.as_ptr(),
                alloc_info.mapped_data as *mut u8,
                size_in_bytes,
            );
        }

        self.buffer = new_buffer;
        Ok(())
    }

    /// Get raw data as bytes (for saving/serialization)
    pub fn data(&self) -> Result<Vec<u8>> {
        self.device.synchronize()?;

        let allocation_info = self.device.allocator().get_allocation_info(self.buffer.allocation());
        if allocation_info.mapped_data.is_null() {
            crate::bail!("Buffer not mapped");
        }

        let size = self.storage_size_in_bytes();
        let data = unsafe {
            std::slice::from_raw_parts(allocation_info.mapped_data as *const u8, size)
        };
        Ok(data.to_vec())
    }
}

/// Load quantized data to Vulkan device
pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<QStorage> {
    let size_in_bytes = data.len() * std::mem::size_of::<T>();
    let num_f32_elements = (size_in_bytes + 3) / 4;

    let buffer = device.new_buffer(num_f32_elements, DType::F32)?;

    let allocation_info = device.allocator().get_allocation_info(buffer.allocation());
    if allocation_info.mapped_data.is_null() {
        crate::bail!("Buffer not mapped");
    }

    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            allocation_info.mapped_data as *mut u8,
            size_in_bytes,
        );
    }

    Ok(QStorage::Vulkan(QVulkanStorage {
        dtype: T::DTYPE,
        device: device.clone(),
        buffer,
    }))
}
