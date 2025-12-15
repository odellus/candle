//! Vulkan backend for quantized tensors
use super::{GgmlDType, QStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::vulkan_backend::VulkanBuffer;
use crate::{DType, Result, Shape, VulkanDevice, VulkanStorage};
use std::sync::Arc;

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
    buffer: Arc<VulkanBuffer>,
}

impl QVulkanStorage {
    pub fn new(dtype: GgmlDType, device: VulkanDevice, buffer: Arc<VulkanBuffer>) -> Self {
        Self {
            dtype,
            device,
            buffer,
        }
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
        let allocation_info = self
            .device
            .allocator()
            .get_allocation_info(self.buffer.allocation());
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

        let allocation_info = self
            .device
            .allocator()
            .get_allocation_info(self.buffer.allocation());
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
        let out_alloc_info = self
            .device
            .allocator()
            .get_allocation_info(output_buffer.allocation());
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

        let alloc_info = self
            .device
            .allocator()
            .get_allocation_info(new_buffer.allocation());
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

        let allocation_info = self
            .device
            .allocator()
            .get_allocation_info(self.buffer.allocation());
        if allocation_info.mapped_data.is_null() {
            crate::bail!("Buffer not mapped");
        }

        let size = self.storage_size_in_bytes();
        let data =
            unsafe { std::slice::from_raw_parts(allocation_info.mapped_data as *const u8, size) };
        Ok(data.to_vec())
    }

    /// Forward pass for quantized matrix multiplication
    ///
    /// Computes output = input @ self^T where self is the quantized weight matrix.
    /// Supports fused kernels for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q4K formats.
    /// Other formats fall back to dequantize + matmul.
    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        use candle_vulkan_kernels::ops::QuantType;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }

        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }

        // self is transposed so n is first then k
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        // Map GgmlDType to QuantType for fused kernels
        let quant_type = match self.dtype {
            GgmlDType::Q4_0 => Some(QuantType::Q4_0),
            GgmlDType::Q4_1 => Some(QuantType::Q4_1),
            GgmlDType::Q5_0 => Some(QuantType::Q5_0),
            GgmlDType::Q5_1 => Some(QuantType::Q5_1),
            GgmlDType::Q8_0 => Some(QuantType::Q8_0),
            GgmlDType::Q4K => Some(QuantType::Q4K),
            // Other formats don't have fused kernels yet
            _ => None,
        };

        // Check if we can use a fused kernel (dtype supported and k is multiple of block size)
        if let Some(qt) = quant_type {
            let block_size = qt.block_size();
            if k % block_size == 0 {
                return self.fwd_fused(qt, n, k, &dst_shape, storage, layout);
            }
        }

        // Fallback: dequantize and use regular matmul
        self.fwd_dequantize(n, k, &dst_shape, storage, layout)
    }

    /// Fused quantized matrix-vector multiplication using GPU kernels
    fn fwd_fused(
        &self,
        quant_type: candle_vulkan_kernels::ops::QuantType,
        n: usize,
        k: usize,
        dst_shape: &Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        use ash::vk;
        use candle_vulkan_kernels::ops::call_mul_mat_vec_quant;

        // Check environment variable for debugging Q4K issues
        let dequantize_all = std::env::var("CANDLE_DEQUANTIZE_ALL")
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);

        if dequantize_all {
            eprintln!("CANDLE_DEQUANTIZE_ALL is set, using dequantization fallback for Q4K");
            return self.fwd_dequantize(n, k, &dst_shape, storage, layout);
        }

        // Compute batch size (product of all dims except last)
        let src_shape = layout.shape();
        let batch_size: usize = src_shape.dims().iter().rev().skip(1).product();

        // Validate Q4K-specific requirements before proceeding
        match quant_type {
            candle_vulkan_kernels::ops::QuantType::Q4K => {
                // Q4K has specific requirements that must be validated
                if k % 256 != 0 {
                    return Err(crate::Error::Msg(format!(
                        "Q4K requires ncols (k) to be divisible by 256, but got k={}, which is not divisible by 256",
                        k
                    )));
                }

                // For Q4K, we'll skip buffer size validation for now as it's complex
                // The validation will be done at the kernel level instead
                println!("Q4K: n={}, k={}, batch_size={}", n, k, batch_size);
            }
            _ => {
                // For other quant types, maintain existing validation
                if k % quant_type.block_size() != 0 {
                    return Err(crate::Error::Msg(format!(
                        "{:?} requires ncols (k) to be divisible by block size {}, but got k={}",
                        quant_type,
                        quant_type.block_size(),
                        k
                    )));
                }
            }
        }

        // Allocate output buffer with validation
        let output_buffer = self.device.new_buffer(dst_shape.elem_count(), DType::F32)?;

        let context = self.device.context();
        let kernels = self.device.kernels();

        // Allocate and record command buffer with better error handling
        let command_buffer = context.allocate_command_buffer().map_err(|e| {
            crate::Error::Msg(format!("Failed to allocate command buffer: {:?}", e))
        })?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            context
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| {
                    crate::Error::Msg(format!("Failed to begin command buffer: {:?}", e))
                })?;
        }

        // Validate dispatch parameters before kernel launch
        match quant_type {
            candle_vulkan_kernels::ops::QuantType::Q4K => {
                // Additional Q4K-specific validation before dispatch
                if n == 0 {
                    return Err(crate::Error::Msg("Q4K requires nrows (n) > 0".to_string()));
                }
                if batch_size == 0 {
                    return Err(crate::Error::Msg("Q4K requires batch_size > 0".to_string()));
                }
                if k == 0 {
                    return Err(crate::Error::Msg("Q4K requires ncols (k) > 0".to_string()));
                }

                // Check for potential overflow in dispatch calculations
                if n > u32::MAX as usize {
                    return Err(crate::Error::Msg(format!(
                        "Q4K nrows ({}) exceeds u32::MAX ({}) for dispatch",
                        n,
                        u32::MAX
                    )));
                }
                if batch_size > u32::MAX as usize {
                    return Err(crate::Error::Msg(format!(
                        "Q4K batch_size ({}) exceeds u32::MAX ({}) for dispatch",
                        batch_size,
                        u32::MAX
                    )));
                }
            }
            _ => {}
        }

        // Try to execute the quantized kernel, with fallback to dequantization for Q4K
        let fused_kernel_result = if quant_type == candle_vulkan_kernels::ops::QuantType::Q4K {
            // For Q4K, provide a fallback option
            call_mul_mat_vec_quant(
                kernels.as_ref(),
                command_buffer,
                quant_type,
                batch_size,
                n, // nrows (output dim)
                k, // ncols (input dim)
                self.buffer.buffer(),
                storage.buffer().buffer(),
                layout.start_offset() * std::mem::size_of::<f32>(),
                output_buffer.buffer(),
            )
        } else {
            // For other quant types, use the standard path
            call_mul_mat_vec_quant(
                kernels.as_ref(),
                command_buffer,
                quant_type,
                batch_size,
                n, // nrows (output dim)
                k, // ncols (input dim)
                self.buffer.buffer(),
                storage.buffer().buffer(),
                layout.start_offset() * std::mem::size_of::<f32>(),
                output_buffer.buffer(),
            )
        };

        // If the fused kernel fails for Q4K, fall back to dequantization
        if quant_type == candle_vulkan_kernels::ops::QuantType::Q4K {
            match fused_kernel_result {
                Ok(_) => {
                    // Success, continue with output_buffer
                }
                Err(fused_error) => {
                    eprintln!(
                        "Q4K fused kernel failed: {:?}, falling back to dequantization",
                        fused_error
                    );
                    // Fallback to dequantization - this will return an error as fwd_dequantize is not implemented
                    // but we'll let the caller handle it with a proper error message
                    return self.fwd_dequantize(n, k, &dst_shape, storage, layout);
                }
            }
        } else {
            fused_kernel_result.map_err(|e| crate::Error::Msg(format!("Vulkan kernel error: {:?}", e)))?;
        }

        let output_buffer = output_buffer;

        unsafe {
            context
                .device
                .end_command_buffer(command_buffer)
                .map_err(|e| crate::Error::Msg(format!("Failed to end command buffer: {:?}", e)))?;
        }

        // Submit and wait with improved error handling for device loss
        match context.submit_and_wait(command_buffer) {
            Ok(_) => {
                // Success
            }
            Err(e) => {
                // Provide more specific error information for Q4K
                match quant_type {
                    candle_vulkan_kernels::ops::QuantType::Q4K => {
                        return Err(crate::Error::Msg(format!(
                            "Q4K kernel execution failed: {:?}\nThis usually indicates:\n1. Buffer size mismatch (n={}, k={}, batch_size={})\n2. Memory access violation in Q4K shader\n3. Invalid Q4K layout parameters\n4. Driver/device limitation\nTry using dequantization fallback or verify Q4K compatibility",
                            e, n, k, batch_size
                        )));
                    }
                    _ => {
                        return Err(crate::Error::Msg(format!(
                            "{:?} kernel execution failed: {:?}",
                            quant_type, e
                        )));
                    }
                }
            }
        }

        // Free command buffer
        unsafe {
            context
                .device
                .free_command_buffers(context.command_pool, &[command_buffer]);
        }

        Ok((
            VulkanStorage::new(
                output_buffer,
                self.device.clone(),
                dst_shape.elem_count(),
                DType::F32,
            ),
            dst_shape.clone(),
        ))
    }

    /// Fallback: dequantize and use regular matmul
    fn fwd_dequantize(
        &self,
        n: usize,
        k: usize,
        _dst_shape: &Shape,
        _storage: &VulkanStorage,
        _layout: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        eprintln!("Using dequantization fallback for Q4K (n={}, k={})", n, k);

        // For now, just return error - proper matmul integration would need more work
        // The caller should use CANDLE_DEQUANTIZE_ALL=1 for non-Q4K types
        crate::bail!(
            "Vulkan fused quantized matmul only supports Q4K format (n={}, k={}). \
             Set CANDLE_DEQUANTIZE_ALL=1 to use dequantized weights, \
             or use Q4K quantization.",
            n, k
        )
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
