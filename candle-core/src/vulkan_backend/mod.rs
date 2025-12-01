//! Implementation of Backend traits for Vulkan
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
use ash::vk;
use candle_vulkan_kernels::VulkanContext;
use std::sync::Arc;

mod device;
pub use device::{DeviceId, VulkanDevice};

/// Vulkan related errors
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_vulkan_kernels::VulkanError),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

/// A Vulkan buffer with its allocation, properly cleaned up on drop.
/// This wrapper ensures that the buffer is destroyed before the allocator.
pub struct VulkanBuffer {
    buffer: ash::vk::Buffer,
    allocation: vk_mem::Allocation,
    context: Arc<VulkanContext>,
}

impl VulkanBuffer {
    pub fn new(buffer: ash::vk::Buffer, allocation: vk_mem::Allocation, context: Arc<VulkanContext>) -> Self {
        Self { buffer, allocation, context }
    }

    pub fn buffer(&self) -> ash::vk::Buffer {
        self.buffer
    }

    pub fn allocation(&self) -> &vk_mem::Allocation {
        &self.allocation
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("buffer", &self.buffer)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct VulkanStorage {
    /// The buffer with its allocation (Arc for cloning)
    buffer: Arc<VulkanBuffer>,
    /// A reference to the device owning this buffer
    device: VulkanDevice,
    /// The count of allocated elements in the buffer
    count: usize,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
}

impl VulkanStorage {
    pub fn new(
        buffer: Arc<VulkanBuffer>,
        device: VulkanDevice,
        count: usize,
        dtype: DType,
    ) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> &VulkanBuffer {
        &self.buffer
    }

    pub fn vk_buffer(&self) -> ash::vk::Buffer {
        self.buffer.buffer()
    }

    fn to_cpu<T: Clone>(&self) -> Result<Vec<T>> {
        self.device.synchronize()?;

        let allocation_info = self.device.allocator().get_allocation_info(self.buffer.allocation());
        let mapped_ptr = allocation_info.mapped_data;

        if mapped_ptr.is_null() {
            return Err(crate::Error::Msg("Buffer not mapped".to_string()));
        }

        let slice = unsafe {
            std::slice::from_raw_parts(mapped_ptr as *const T, self.count)
        };
        Ok(slice.to_vec())
    }

    /// Execute a unary kernel operation
    fn run_unary_kernel<F>(&self, layout: &Layout, kernel_fn: F) -> Result<Self>
    where
        F: FnOnce(
            &candle_vulkan_kernels::Kernels,
            vk::CommandBuffer,
            usize,
            vk::Buffer,
            vk::Buffer,
        ) -> candle_vulkan_kernels::Result<()>,
    {
        // Only support contiguous tensors for now
        if !layout.is_contiguous() {
            crate::bail!("Vulkan unary ops only support contiguous tensors");
        }

        // Only support F32 for now
        if self.dtype != DType::F32 {
            crate::bail!("Vulkan unary ops only support F32 for now");
        }

        let num_elements = layout.shape().elem_count();
        let output_buffer = self.device.new_buffer(num_elements, self.dtype)?;

        let context = self.device.context();
        let kernels = self.device.kernels();

        // Allocate and record command buffer
        let command_buffer = context.allocate_command_buffer()
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate command buffer: {:?}", e)))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            context.device.begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| crate::Error::Msg(format!("Failed to begin command buffer: {:?}", e)))?;
        }

        // Dispatch the kernel
        kernel_fn(
            kernels.as_ref(),
            command_buffer,
            num_elements,
            self.vk_buffer(),
            output_buffer.buffer(),
        ).map_err(|e| crate::Error::Msg(format!("Kernel dispatch failed: {:?}", e)))?;

        unsafe {
            context.device.end_command_buffer(command_buffer)
                .map_err(|e| crate::Error::Msg(format!("Failed to end command buffer: {:?}", e)))?;
        }

        // Submit and wait
        context.submit_and_wait(command_buffer)
            .map_err(|e| crate::Error::Msg(format!("Failed to submit command buffer: {:?}", e)))?;

        // Free command buffer
        unsafe {
            context.device.free_command_buffers(context.command_pool, &[command_buffer]);
        }

        Ok(VulkanStorage::new(output_buffer, self.device.clone(), num_elements, self.dtype))
    }

    /// Execute a binary kernel operation
    fn run_binary_kernel<F>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout, kernel_fn: F) -> Result<Self>
    where
        F: FnOnce(
            &candle_vulkan_kernels::Kernels,
            vk::CommandBuffer,
            usize,
            vk::Buffer,
            vk::Buffer,
            vk::Buffer,
        ) -> candle_vulkan_kernels::Result<()>,
    {
        // Only support contiguous tensors for now
        if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
            crate::bail!("Vulkan binary ops only support contiguous tensors");
        }

        // Only support F32 for now
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            crate::bail!("Vulkan binary ops only support F32 for now");
        }

        // Shapes must match for element-wise ops
        if lhs_l.shape() != rhs_l.shape() {
            crate::bail!("Vulkan binary ops require matching shapes (broadcasting not yet supported)");
        }

        let num_elements = lhs_l.shape().elem_count();
        let output_buffer = self.device.new_buffer(num_elements, self.dtype)?;

        let context = self.device.context();
        let kernels = self.device.kernels();

        // Allocate and record command buffer
        let command_buffer = context.allocate_command_buffer()
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate command buffer: {:?}", e)))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            context.device.begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| crate::Error::Msg(format!("Failed to begin command buffer: {:?}", e)))?;
        }

        // Dispatch the kernel
        kernel_fn(
            kernels.as_ref(),
            command_buffer,
            num_elements,
            self.vk_buffer(),
            rhs.vk_buffer(),
            output_buffer.buffer(),
        ).map_err(|e| crate::Error::Msg(format!("Kernel dispatch failed: {:?}", e)))?;

        unsafe {
            context.device.end_command_buffer(command_buffer)
                .map_err(|e| crate::Error::Msg(format!("Failed to end command buffer: {:?}", e)))?;
        }

        // Submit and wait
        context.submit_and_wait(command_buffer)
            .map_err(|e| crate::Error::Msg(format!("Failed to submit command buffer: {:?}", e)))?;

        // Free command buffer
        unsafe {
            context.device.free_command_buffers(context.command_pool, &[command_buffer]);
        }

        Ok(VulkanStorage::new(output_buffer, self.device.clone(), num_elements, self.dtype))
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
        }
    }

    fn affine(&self, _layout: &Layout, _mul: f64, _add: f64) -> Result<Self> {
        crate::bail!("Vulkan affine not yet implemented")
    }

    fn powf(&self, _layout: &Layout, _pow: f64) -> Result<Self> {
        crate::bail!("Vulkan powf not yet implemented")
    }

    fn elu(&self, _layout: &Layout, _alpha: f64) -> Result<Self> {
        crate::bail!("Vulkan elu not yet implemented")
    }

    fn reduce_op(&self, _op: ReduceOp, _layout: &Layout, _sum_dims: &[usize]) -> Result<Self> {
        crate::bail!("Vulkan reduce_op not yet implemented")
    }

    fn cmp(&self, _op: CmpOp, _rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        crate::bail!("Vulkan cmp not yet implemented")
    }

    fn const_set(&mut self, _s: crate::scalar::Scalar, _l: &Layout) -> Result<()> {
        crate::bail!("Vulkan const_set not yet implemented")
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        crate::bail!("Vulkan to_dtype not yet implemented")
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        use candle_vulkan_kernels::ops::{call_exp, call_gelu, call_relu, call_silu};

        match B::NAME {
            "exp" => self.run_unary_kernel(layout, |k, cb, n, i, o| call_exp(k, cb, n, i, o)),
            "silu" => self.run_unary_kernel(layout, |k, cb, n, i, o| call_silu(k, cb, n, i, o)),
            "gelu" => self.run_unary_kernel(layout, |k, cb, n, i, o| call_gelu(k, cb, n, i, o)),
            "relu" => self.run_unary_kernel(layout, |k, cb, n, i, o| call_relu(k, cb, n, i, o)),
            op => crate::bail!("Vulkan unary op '{}' not yet implemented", op),
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use candle_vulkan_kernels::ops::{call_add, call_div, call_mul};

        match B::NAME {
            "add" => self.run_binary_kernel(rhs, lhs_l, rhs_l, |k, cb, n, a, b, o| call_add(k, cb, n, a, b, o)),
            "mul" => self.run_binary_kernel(rhs, lhs_l, rhs_l, |k, cb, n, a, b, o| call_mul(k, cb, n, a, b, o)),
            "div" => self.run_binary_kernel(rhs, lhs_l, rhs_l, |k, cb, n, a, b, o| call_div(k, cb, n, a, b, o)),
            op => crate::bail!("Vulkan binary op '{}' not yet implemented", op),
        }
    }

    fn where_cond(
        &self,
        _layout: &Layout,
        _t: &Self,
        _t_l: &Layout,
        _f: &Self,
        _f_l: &Layout,
    ) -> Result<Self> {
        crate::bail!("Vulkan where_cond not yet implemented")
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv1d not yet implemented")
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv_transpose1d not yet implemented")
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv2d not yet implemented")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("Vulkan conv_transpose2d not yet implemented")
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        crate::bail!("Vulkan avg_pool2d not yet implemented")
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        crate::bail!("Vulkan max_pool2d not yet implemented")
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> Result<Self> {
        crate::bail!("Vulkan upsample_nearest1d not yet implemented")
    }

    fn upsample_nearest2d(&self, _layout: &Layout, _h: usize, _w: usize) -> Result<Self> {
        crate::bail!("Vulkan upsample_nearest2d not yet implemented")
    }

    fn gather(&self, _l: &Layout, _indexes: &Self, _indexes_l: &Layout, _d: usize) -> Result<Self> {
        crate::bail!("Vulkan gather not yet implemented")
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _d: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan scatter_set not yet implemented")
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _d: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan scatter_add_set not yet implemented")
    }

    fn index_select(
        &self,
        _rhs: &Self,
        _lhs_l: &Layout,
        _rhs_l: &Layout,
        _d: usize,
    ) -> Result<Self> {
        crate::bail!("Vulkan index_select not yet implemented")
    }

    fn index_add(
        &self,
        _l: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _d: usize,
    ) -> Result<Self> {
        crate::bail!("Vulkan index_add not yet implemented")
    }

    fn matmul(
        &self,
        _rhs: &Self,
        _bmnk: (usize, usize, usize, usize),
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<Self> {
        // TODO: Implement using our matvec kernel as a starting point
        crate::bail!("Vulkan matmul not yet implemented")
    }

    fn copy_strided_src(&self, _dst: &mut Self, _dst_offset: usize, _src_l: &Layout) -> Result<()> {
        crate::bail!("Vulkan copy_strided_src not yet implemented")
    }

    fn copy2d(
        &self,
        _dst: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_s: usize,
        _dst_s: usize,
        _src_o: usize,
        _dst_o: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan copy2d not yet implemented")
    }
}
