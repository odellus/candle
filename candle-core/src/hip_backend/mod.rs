//! Implementation of Backend traits for HIP (ROCm) device
//!

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
pub use candle_hip_kernels as kernels;
use half::{bf16, f16};
use std::ffi::c_void;

pub mod ffi;
pub mod memory;
mod device;
mod error;

pub use device::{DeviceId, Function, HipDevice, Module, Stream};
pub use error::{HipError, WrapErr};
pub use memory::DeviceMemory;

use ffi::{rocblas_datatype, rocblas_gemm_algo, rocblas_operation, ROCBLAS_GEMM_FLAGS_NONE};

#[derive(Debug)]
pub enum HipStorageSlice {
    U8(DeviceMemory<u8>),
    U32(DeviceMemory<u32>),
    I16(DeviceMemory<i16>),
    I32(DeviceMemory<i32>),
    I64(DeviceMemory<i64>),
    BF16(DeviceMemory<bf16>),
    F16(DeviceMemory<f16>),
    F32(DeviceMemory<f32>),
    F64(DeviceMemory<f64>),
}

impl HipStorageSlice {
    pub fn from_host_slice<T: WithDType>(device: &HipDevice, data: &[T]) -> Result<Self> {
        let elem_count = data.len();
        match T::DTYPE {
            DType::U8 => {
                let data: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, elem_count)
                };
                let mut mem = device.alloc::<u8>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::U8(mem))
            }
            DType::U32 => {
                let data: &[u32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u32, elem_count)
                };
                let mut mem = device.alloc::<u32>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::U32(mem))
            }
            DType::I64 => {
                let data: &[i64] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i64, elem_count)
                };
                let mut mem = device.alloc::<i64>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::I64(mem))
            }
            DType::BF16 => {
                let data: &[bf16] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const bf16, elem_count)
                };
                let mut mem = device.alloc::<bf16>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::BF16(mem))
            }
            DType::F16 => {
                let data: &[f16] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f16, elem_count)
                };
                let mut mem = device.alloc::<f16>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F16(mem))
            }
            DType::F32 => {
                let data: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count)
                };
                let mut mem = device.alloc::<f32>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F32(mem))
            }
            DType::F64 => {
                let data: &[f64] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f64, elem_count)
                };
                let mut mem = device.alloc::<f64>(elem_count)?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F64(mem))
            }
            dtype => crate::bail!("dtype {dtype:?} is not supported on HIP"),
        }
    }

    pub fn from_cpu_storage(device: &HipDevice, storage: &CpuStorage) -> Result<Self> {
        match storage {
            CpuStorage::U8(data) => {
                let mut mem = device.alloc::<u8>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::U8(mem))
            }
            CpuStorage::U32(data) => {
                let mut mem = device.alloc::<u32>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::U32(mem))
            }
            CpuStorage::I16(data) => {
                let mut mem = device.alloc::<i16>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::I16(mem))
            }
            CpuStorage::I32(data) => {
                let mut mem = device.alloc::<i32>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::I32(mem))
            }
            CpuStorage::I64(data) => {
                let mut mem = device.alloc::<i64>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::I64(mem))
            }
            CpuStorage::BF16(data) => {
                let mut mem = device.alloc::<bf16>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::BF16(mem))
            }
            CpuStorage::F16(data) => {
                let mut mem = device.alloc::<f16>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F16(mem))
            }
            CpuStorage::F32(data) => {
                let mut mem = device.alloc::<f32>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F32(mem))
            }
            CpuStorage::F64(data) => {
                let mut mem = device.alloc::<f64>(data.len())?;
                device.memcpy_htod(data, &mut mem)?;
                Ok(Self::F64(mem))
            }
            _ => crate::bail!("Unsupported CPU storage type for HIP"),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

#[derive(Debug)]
pub struct HipStorage {
    pub(crate) slice: HipStorageSlice,
    pub(crate) device: HipDevice,
}

impl HipStorage {
    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    /// Get the raw device pointer for this storage
    pub fn as_hip_ptr(&self) -> Result<*const c_void> {
        let ptr = match &self.slice {
            HipStorageSlice::U8(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::U32(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::I16(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::I32(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::I64(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::BF16(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::F16(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::F32(s) => s.as_ptr() as *const c_void,
            HipStorageSlice::F64(s) => s.as_ptr() as *const c_void,
        };
        Ok(ptr)
    }

    /// Get the raw mutable device pointer for this storage
    pub fn as_hip_ptr_mut(&mut self) -> Result<*mut c_void> {
        let ptr = match &mut self.slice {
            HipStorageSlice::U8(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::U32(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::I16(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::I32(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::I64(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::BF16(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::F16(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::F32(s) => s.as_mut_ptr() as *mut c_void,
            HipStorageSlice::F64(s) => s.as_mut_ptr() as *mut c_void,
        };
        Ok(ptr)
    }

    /// Launch a kernel with the given parameters
    fn launch_kernel(
        &self,
        module_name: &str,
        func_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> Result<()> {
        // Ensure we're on the correct device
        unsafe {
            ffi::check_hip_error(ffi::hipSetDevice(self.device.ordinal() as i32))
                .map_err(|e| HipError::Hip(e))
                .w()?;
        }

        // Load the module
        let hsaco = match module_name {
            "fill" => kernels::FILL.hsaco(),
            "unary" => kernels::UNARY.hsaco(),
            "binary" => kernels::BINARY.hsaco(),
            "affine" => kernels::AFFINE.hsaco(),
            "cast" => kernels::CAST.hsaco(),
            "reduce" => kernels::REDUCE.hsaco(),
            "indexing" => kernels::INDEXING.hsaco(),
            "ternary" => kernels::TERNARY.hsaco(),
            "conv" => kernels::CONV.hsaco(),
            _ => crate::bail!("Unknown kernel module: {}", module_name),
        };

        let module = self.device.load_module(module_name, hsaco)?;
        let func = module.get_function(func_name)?;

        unsafe {
            ffi::check_hip_error(ffi::hipModuleLaunchKernel(
                func.as_ptr(),
                grid_dim.0,
                grid_dim.1,
                grid_dim.2,
                block_dim.0,
                block_dim.1,
                block_dim.2,
                0, // shared memory
                self.device.stream().as_ptr(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ))
            .map_err(|e| HipError::KernelLaunch(e))
            .w()?;

            // Synchronize to catch any kernel execution errors
            ffi::check_hip_error(ffi::hipStreamSynchronize(self.device.stream().as_ptr()))
                .map_err(|e| HipError::Hip(format!("Kernel sync error: {}", e)))
                .w()?;
        }

        Ok(())
    }

    /// Calculate grid dimensions for a given number of elements
    fn grid_dims(numel: usize, block_size: u32) -> (u32, u32, u32) {
        let num_blocks = ((numel as u32) + block_size - 1) / block_size;
        (num_blocks.min(65535), 1, 1)
    }

    pub fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            HipStorageSlice::U8(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::U8(vec))
            }
            HipStorageSlice::U32(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::U32(vec))
            }
            HipStorageSlice::I16(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::I16(vec))
            }
            HipStorageSlice::I32(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::I32(vec))
            }
            HipStorageSlice::I64(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::I64(vec))
            }
            HipStorageSlice::BF16(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::BF16(vec))
            }
            HipStorageSlice::F16(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::F16(vec))
            }
            HipStorageSlice::F32(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::F32(vec))
            }
            HipStorageSlice::F64(data) => {
                let vec = self.device.memcpy_dtov(data)?;
                Ok(CpuStorage::F64(vec))
            }
        }
    }

    /// Im2col1d transformation for conv1d
    fn im2col1d_kernel(
        &self,
        l: &Layout,
        l_out: usize,
        l_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<HipStorageSlice> {
        let shape = l.shape();
        let dims = shape.dims();
        let dst_numel = dims[0] * l_out * dims[1] * l_k;

        // Prepare info buffer with dims and strides
        let info_vec: Vec<usize> = dims.iter().copied().chain(l.stride().iter().copied()).collect();
        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_numel, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_im2col1d {
            ($src_slice:ident, $ty:ty, $slice_variant:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_numel)?;

                let mut dst_numel_val = dst_numel;
                let mut l_out_val = l_out;
                let mut l_k_val = l_k;
                let mut stride_val = stride;
                let mut padding_val = padding;
                let mut dilation_val = dilation;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut dst_numel_val as *mut usize as *mut c_void,
                    &mut l_out_val as *mut usize as *mut c_void,
                    &mut l_k_val as *mut usize as *mut c_void,
                    &mut stride_val as *mut usize as *mut c_void,
                    &mut padding_val as *mut usize as *mut c_void,
                    &mut dilation_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::F32(src) => impl_im2col1d!(src, f32, F32, "im2col1d_f32"),
            HipStorageSlice::F64(src) => impl_im2col1d!(src, f64, F64, "im2col1d_f64"),
            HipStorageSlice::F16(src) => impl_im2col1d!(src, f16, F16, "im2col1d_f16"),
            HipStorageSlice::BF16(src) => impl_im2col1d!(src, bf16, BF16, "im2col1d_bf16"),
            HipStorageSlice::U8(src) => impl_im2col1d!(src, u8, U8, "im2col1d_u8"),
            HipStorageSlice::U32(src) => impl_im2col1d!(src, u32, U32, "im2col1d_u32"),
            _ => crate::bail!("im2col1d: unsupported dtype"),
        };

        Ok(slice)
    }

    /// Im2col transformation for conv2d
    fn im2col_kernel(
        &self,
        l: &Layout,
        h_out: usize,
        w_out: usize,
        h_k: usize,
        w_k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<HipStorageSlice> {
        let shape = l.shape();
        let dims = shape.dims();
        let dst_numel = dims[0] * h_out * w_out * dims[1] * h_k * w_k;

        // Prepare info buffer with dims and strides
        let info_vec: Vec<usize> = dims.iter().copied().chain(l.stride().iter().copied()).collect();
        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_numel, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_im2col {
            ($src_slice:ident, $ty:ty, $slice_variant:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_numel)?;

                let mut dst_numel_val = dst_numel;
                let mut h_out_val = h_out;
                let mut w_out_val = w_out;
                let mut h_k_val = h_k;
                let mut w_k_val = w_k;
                let mut stride_val = stride;
                let mut padding_val = padding;
                let mut dilation_val = dilation;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut dst_numel_val as *mut usize as *mut c_void,
                    &mut h_out_val as *mut usize as *mut c_void,
                    &mut w_out_val as *mut usize as *mut c_void,
                    &mut h_k_val as *mut usize as *mut c_void,
                    &mut w_k_val as *mut usize as *mut c_void,
                    &mut stride_val as *mut usize as *mut c_void,
                    &mut padding_val as *mut usize as *mut c_void,
                    &mut dilation_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::F32(src) => impl_im2col!(src, f32, F32, "im2col_f32"),
            HipStorageSlice::F64(src) => impl_im2col!(src, f64, F64, "im2col_f64"),
            HipStorageSlice::F16(src) => impl_im2col!(src, f16, F16, "im2col_f16"),
            HipStorageSlice::BF16(src) => impl_im2col!(src, bf16, BF16, "im2col_bf16"),
            HipStorageSlice::U8(src) => impl_im2col!(src, u8, U8, "im2col_u8"),
            HipStorageSlice::U32(src) => impl_im2col!(src, u32, U32, "im2col_u32"),
            _ => crate::bail!("im2col: unsupported dtype"),
        };

        Ok(slice)
    }
}

// ============================================================================
// GEMM configuration for rocBLAS
// ============================================================================

/// GEMM configuration for strided batched operations
struct GemmConfig {
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    batch_count: i32,
}

fn gemm_config(
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<GemmConfig> {
    // rocBLAS uses column-major order, so we swap A and B and compute C = B * A
    // This gives us row-major C = A * B
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, rocblas_operation::rocblas_operation_none)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, rocblas_operation::rocblas_operation_transpose)
    } else {
        crate::bail!(
            "matmul: rhs has non-contiguous layout, rhs_stride={:?} mnk=({}, {}, {})",
            rhs_stride, m, n, k
        )
    };

    // The b tensor has dims batching, m, k (lhs)
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, rocblas_operation::rocblas_operation_none)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, rocblas_operation::rocblas_operation_transpose)
    } else {
        crate::bail!(
            "matmul: lhs has non-contiguous layout, lhs_stride={:?} mnk=({}, {}, {})",
            lhs_stride, m, n, k
        )
    };

    let stride_b: i64 = match &lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if *s1 == stride * lhs_l.dims()[1] => *stride as i64,
        [_, stride] if lhs_l.dims()[0] == 1 => *stride as i64,
        [stride, _] if lhs_l.dims()[1] == 1 => *stride as i64,
        [stride] => *stride as i64,
        [] => (m * k) as i64,
        _ => crate::bail!(
            "matmul: lhs stride not supported, lhs_stride={:?} mnk=({}, {}, {})",
            lhs_stride, m, n, k
        ),
    };

    let stride_a: i64 = match &rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if *s1 == stride * rhs_l.dims()[1] => *stride as i64,
        [_, stride] if rhs_l.dims()[0] == 1 => *stride as i64,
        [stride, _] if rhs_l.dims()[1] == 1 => *stride as i64,
        [stride] => *stride as i64,
        [] => (n * k) as i64,
        _ => crate::bail!(
            "matmul: rhs stride not supported, rhs_stride={:?} mnk=({}, {}, {})",
            rhs_stride, m, n, k
        ),
    };

    Ok(GemmConfig {
        transa,
        transb,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        stride_a,
        stride_b,
        stride_c: (m * n) as i64,
        batch_count: b as i32,
    })
}

impl BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I16(_) => HipStorageSlice::I16(self.device.alloc::<i16>(0)?),
                HipStorageSlice::I32(_) => HipStorageSlice::I32(self.device.alloc::<i32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // For contiguous layouts, use simple memcpy
        if layout.is_contiguous() {
            let slice = match &self.slice {
                HipStorageSlice::U8(s) => {
                    let mut dst = self.device.alloc::<u8>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::U8(dst)
                }
                HipStorageSlice::U32(s) => {
                    let mut dst = self.device.alloc::<u32>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::U32(dst)
                }
                HipStorageSlice::I16(s) => {
                    let mut dst = self.device.alloc::<i16>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::I16(dst)
                }
                HipStorageSlice::I32(s) => {
                    let mut dst = self.device.alloc::<i32>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::I32(dst)
                }
                HipStorageSlice::I64(s) => {
                    let mut dst = self.device.alloc::<i64>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::I64(dst)
                }
                HipStorageSlice::BF16(s) => {
                    let mut dst = self.device.alloc::<bf16>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::BF16(dst)
                }
                HipStorageSlice::F16(s) => {
                    let mut dst = self.device.alloc::<f16>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::F16(dst)
                }
                HipStorageSlice::F32(s) => {
                    let mut dst = self.device.alloc::<f32>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::F32(dst)
                }
                HipStorageSlice::F64(s) => {
                    let mut dst = self.device.alloc::<f64>(numel)?;
                    self.device.memcpy_dtod(s, &mut dst, numel)?;
                    HipStorageSlice::F64(dst)
                }
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // For non-contiguous, use the copy kernel
        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_copy {
            ($slice:ident, $ty:ty, $slice_variant:ident, $kernel_name:expr) => {{
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 5] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(out_mem)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::U8(s) => impl_copy!(s, u8, U8, "ucopy_u8"),
            HipStorageSlice::U32(s) => impl_copy!(s, u32, U32, "ucopy_u32"),
            HipStorageSlice::I64(s) => impl_copy!(s, i64, I64, "ucopy_i64"),
            HipStorageSlice::BF16(s) => impl_copy!(s, bf16, BF16, "ucopy_bf16"),
            HipStorageSlice::F16(s) => impl_copy!(s, f16, F16, "ucopy_f16"),
            HipStorageSlice::F32(s) => impl_copy!(s, f32, F32, "ucopy_f32"),
            HipStorageSlice::F64(s) => impl_copy!(s, f64, F64, "ucopy_f64"),
            _ => crate::bail!("try_clone not supported for this dtype on HIP"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        HipStorage::to_cpu_storage(self)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I16(_) => HipStorageSlice::I16(self.device.alloc::<i16>(0)?),
                HipStorageSlice::I32(_) => HipStorageSlice::I32(self.device.alloc::<i32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // For now, require contiguous layout
        if !layout.is_contiguous() {
            crate::bail!("affine on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;  // 0 dims = contiguous
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_affine {
            ($slice:ident, $ty:ty, $kernel_name:expr) => {{
                let mul_val: $ty = mul as $ty;
                let add_val: $ty = add as $ty;
                let mut mul_mut = mul_val;
                let mut add_mut = add_val;
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 7] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                    &mut mul_mut as *mut $ty as *mut c_void,
                    &mut add_mut as *mut $ty as *mut c_void,
                ];
                self.launch_kernel("affine", $kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::U8(s) => HipStorageSlice::U8(impl_affine!(s, u8, "affine_u8")),
            HipStorageSlice::U32(s) => HipStorageSlice::U32(impl_affine!(s, u32, "affine_u32")),
            HipStorageSlice::I16(s) => HipStorageSlice::I16(impl_affine!(s, i16, "affine_i16")),
            HipStorageSlice::I32(s) => HipStorageSlice::I32(impl_affine!(s, i32, "affine_i32")),
            HipStorageSlice::I64(s) => HipStorageSlice::I64(impl_affine!(s, i64, "affine_i64")),
            HipStorageSlice::BF16(s) => {
                let mul_val: bf16 = bf16::from_f64(mul);
                let add_val: bf16 = bf16::from_f64(add);
                let mut mul_mut = mul_val;
                let mut add_mut = add_val;
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<bf16> = self.device.alloc::<bf16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 7] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                    &mut mul_mut as *mut bf16 as *mut c_void,
                    &mut add_mut as *mut bf16 as *mut c_void,
                ];
                self.launch_kernel("affine", "affine_bf16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::BF16(out_mem)
            }
            HipStorageSlice::F16(s) => {
                let mul_val: f16 = f16::from_f64(mul);
                let add_val: f16 = f16::from_f64(add);
                let mut mul_mut = mul_val;
                let mut add_mut = add_val;
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<f16> = self.device.alloc::<f16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 7] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                    &mut mul_mut as *mut f16 as *mut c_void,
                    &mut add_mut as *mut f16 as *mut c_void,
                ];
                self.launch_kernel("affine", "affine_f16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::F16(out_mem)
            }
            HipStorageSlice::F32(s) => HipStorageSlice::F32(impl_affine!(s, f32, "affine_f32")),
            HipStorageSlice::F64(s) => HipStorageSlice::F64(impl_affine!(s, f64, "affine_f64")),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("powf not supported for integer types on HIP"),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        if !layout.is_contiguous() {
            crate::bail!("powf on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_powf {
            ($slice:ident, $ty:ty, $kernel_name:expr) => {{
                let mut param: $ty = e as $ty;
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut $ty as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", $kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::BF16(s) => {
                let mut param: bf16 = bf16::from_f64(e);
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<bf16> = self.device.alloc::<bf16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut bf16 as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", "upowf_bf16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::BF16(out_mem)
            }
            HipStorageSlice::F16(s) => {
                let mut param: f16 = f16::from_f64(e);
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<f16> = self.device.alloc::<f16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut f16 as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", "upowf_f16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::F16(out_mem)
            }
            HipStorageSlice::F32(s) => HipStorageSlice::F32(impl_powf!(s, f32, "upowf_f32")),
            HipStorageSlice::F64(s) => HipStorageSlice::F64(impl_powf!(s, f64, "upowf_f64")),
            _ => crate::bail!("powf not supported for integer types on HIP"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("elu not supported for integer types on HIP"),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        if !layout.is_contiguous() {
            crate::bail!("elu on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_elu {
            ($slice:ident, $ty:ty, $kernel_name:expr) => {{
                let mut param: $ty = alpha as $ty;
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut $ty as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", $kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::BF16(s) => {
                let mut param: bf16 = bf16::from_f64(alpha);
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<bf16> = self.device.alloc::<bf16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut bf16 as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", "uelu_bf16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::BF16(out_mem)
            }
            HipStorageSlice::F16(s) => {
                let mut param: f16 = f16::from_f64(alpha);
                let mut inp_ptr = s.as_ptr();
                let mut out_mem: DeviceMemory<f16> = self.device.alloc::<f16>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut param as *mut f16 as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", "uelu_f16", grid_dim, block_dim, &mut params)?;
                HipStorageSlice::F16(out_mem)
            }
            HipStorageSlice::F32(s) => HipStorageSlice::F32(impl_elu!(s, f32, "uelu_f32")),
            HipStorageSlice::F64(s) => HipStorageSlice::F64(impl_elu!(s, f64, "uelu_f64")),
            _ => crate::bail!("elu not supported for integer types on HIP"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let src_numel = layout.shape().elem_count();
        if src_numel == 0 {
            // For empty tensor reduction, return appropriate empty/default result
            crate::bail!("cannot reduce empty tensor");
        }

        if !layout.is_contiguous() {
            crate::bail!("reduce_op on non-contiguous HIP tensors not yet implemented");
        }

        // Calculate output shape by removing the reduced dimensions
        let dims = layout.dims();
        let mut out_dims: Vec<usize> = Vec::new();
        let mut el_to_sum = 1usize;

        for (i, &d) in dims.iter().enumerate() {
            if sum_dims.contains(&i) {
                el_to_sum *= d;
            } else {
                out_dims.push(d);
            }
        }

        let dst_numel = out_dims.iter().product::<usize>().max(1);

        // If reducing all dims, we get a scalar (out_dims is empty)
        if out_dims.is_empty() {
            out_dims.push(1);
        }

        let mut src_numel_val = src_numel;
        let mut el_to_sum_per_block = el_to_sum;

        // The fast_sum kernel requires valid dims and strides for strided indexing
        // Build the info array: [dims..., strides...]
        let dims_vec: Vec<usize> = dims.to_vec();
        let strides_vec: Vec<usize> = layout.stride().to_vec();
        let num_dims_val = dims_vec.len();

        // Create info array on device: [dims..., strides...]
        let mut info_vec: Vec<usize> = Vec::with_capacity(num_dims_val * 2);
        info_vec.extend_from_slice(&dims_vec);
        info_vec.extend_from_slice(&strides_vec);

        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let mut num_dims: usize = num_dims_val;
        let mut info_ptr: *const usize = info_mem.as_ptr() as *const usize;

        // The fast_* reduce kernels expect exactly 1024 threads per block
        // for the parallel reduction to work properly
        let block_size = 1024u32;
        let grid_dim = (dst_numel as u32, 1, 1);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_reduce {
            ($slice:ident, $ty:ty, $slice_variant:ident, $kernel_name:expr) => {{
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut src_numel_val as *mut usize as *mut c_void,
                    &mut el_to_sum_per_block as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("reduce", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(out_mem)
            }};
        }

        macro_rules! impl_reduce_arg {
            ($slice:ident, $kernel_name:expr) => {{
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<u32> = self.device.alloc::<u32>(dst_numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut src_numel_val as *mut usize as *mut c_void,
                    &mut el_to_sum_per_block as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("reduce", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::U32(out_mem)
            }};
        }

        let slice = match op {
            ReduceOp::Sum => match &self.slice {
                HipStorageSlice::BF16(s) => impl_reduce!(s, bf16, BF16, "fast_sum_bf16"),
                HipStorageSlice::F16(s) => impl_reduce!(s, f16, F16, "fast_sum_f16"),
                HipStorageSlice::F32(s) => impl_reduce!(s, f32, F32, "fast_sum_f32"),
                HipStorageSlice::F64(s) => impl_reduce!(s, f64, F64, "fast_sum_f64"),
                HipStorageSlice::U32(s) => impl_reduce!(s, u32, U32, "fast_sum_u32"),
                HipStorageSlice::I64(s) => impl_reduce!(s, i64, I64, "fast_sum_i64"),
                HipStorageSlice::U8(s) => impl_reduce!(s, u8, U8, "fast_sum_u8"),
                _ => crate::bail!("reduce_op sum not supported for this dtype on HIP"),
            },
            ReduceOp::Min => match &self.slice {
                HipStorageSlice::BF16(s) => impl_reduce!(s, bf16, BF16, "fast_min_bf16"),
                HipStorageSlice::F16(s) => impl_reduce!(s, f16, F16, "fast_min_f16"),
                HipStorageSlice::F32(s) => impl_reduce!(s, f32, F32, "fast_min_f32"),
                HipStorageSlice::F64(s) => impl_reduce!(s, f64, F64, "fast_min_f64"),
                HipStorageSlice::U32(s) => impl_reduce!(s, u32, U32, "fast_min_u32"),
                HipStorageSlice::I64(s) => impl_reduce!(s, i64, I64, "fast_min_i64"),
                HipStorageSlice::U8(s) => impl_reduce!(s, u8, U8, "fast_min_u8"),
                _ => crate::bail!("reduce_op min not supported for this dtype on HIP"),
            },
            ReduceOp::Max => match &self.slice {
                HipStorageSlice::BF16(s) => impl_reduce!(s, bf16, BF16, "fast_max_bf16"),
                HipStorageSlice::F16(s) => impl_reduce!(s, f16, F16, "fast_max_f16"),
                HipStorageSlice::F32(s) => impl_reduce!(s, f32, F32, "fast_max_f32"),
                HipStorageSlice::F64(s) => impl_reduce!(s, f64, F64, "fast_max_f64"),
                HipStorageSlice::U32(s) => impl_reduce!(s, u32, U32, "fast_max_u32"),
                HipStorageSlice::I64(s) => impl_reduce!(s, i64, I64, "fast_max_i64"),
                HipStorageSlice::U8(s) => impl_reduce!(s, u8, U8, "fast_max_u8"),
                _ => crate::bail!("reduce_op max not supported for this dtype on HIP"),
            },
            ReduceOp::ArgMin => match &self.slice {
                HipStorageSlice::BF16(s) => impl_reduce_arg!(s, "fast_argmin_bf16"),
                HipStorageSlice::F16(s) => impl_reduce_arg!(s, "fast_argmin_f16"),
                HipStorageSlice::F32(s) => impl_reduce_arg!(s, "fast_argmin_f32"),
                HipStorageSlice::F64(s) => impl_reduce_arg!(s, "fast_argmin_f64"),
                HipStorageSlice::U32(s) => impl_reduce_arg!(s, "fast_argmin_u32"),
                HipStorageSlice::I64(s) => impl_reduce_arg!(s, "fast_argmin_i64"),
                HipStorageSlice::U8(s) => impl_reduce_arg!(s, "fast_argmin_u8"),
                _ => crate::bail!("reduce_op argmin not supported for this dtype on HIP"),
            },
            ReduceOp::ArgMax => match &self.slice {
                HipStorageSlice::BF16(s) => impl_reduce_arg!(s, "fast_argmax_bf16"),
                HipStorageSlice::F16(s) => impl_reduce_arg!(s, "fast_argmax_f16"),
                HipStorageSlice::F32(s) => impl_reduce_arg!(s, "fast_argmax_f32"),
                HipStorageSlice::F64(s) => impl_reduce_arg!(s, "fast_argmax_f64"),
                HipStorageSlice::U32(s) => impl_reduce_arg!(s, "fast_argmax_u32"),
                HipStorageSlice::I64(s) => impl_reduce_arg!(s, "fast_argmax_i64"),
                HipStorageSlice::U8(s) => impl_reduce_arg!(s, "fast_argmax_u8"),
                _ => crate::bail!("reduce_op argmax not supported for this dtype on HIP"),
            },
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let numel = lhs_l.shape().elem_count();
        if numel == 0 {
            return Ok(Self {
                slice: HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                device: self.device.clone(),
            });
        }

        if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
            crate::bail!("cmp on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut dims_ptr: *const usize = std::ptr::null();

        let op_name = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
        };

        macro_rules! impl_cmp {
            ($lhs_slice:ident, $rhs_slice:ident, $suffix:expr) => {{
                let kernel_name = format!("{}_{}", op_name, $suffix);
                let mut lhs_ptr = $lhs_slice.as_ptr();
                let mut rhs_ptr = $rhs_slice.as_ptr();
                let mut out_mem: DeviceMemory<u8> = self.device.alloc::<u8>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut dims_ptr as *mut *const usize as *mut c_void,
                    &mut lhs_ptr as *mut *mut c_void as *mut c_void,
                    &mut rhs_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("binary", &kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let out_slice = match (&self.slice, &rhs.slice) {
            (HipStorageSlice::U8(l), HipStorageSlice::U8(r)) => impl_cmp!(l, r, "u8"),
            (HipStorageSlice::U32(l), HipStorageSlice::U32(r)) => impl_cmp!(l, r, "u32"),
            (HipStorageSlice::I64(l), HipStorageSlice::I64(r)) => impl_cmp!(l, r, "i64"),
            (HipStorageSlice::BF16(l), HipStorageSlice::BF16(r)) => impl_cmp!(l, r, "bf16"),
            (HipStorageSlice::F16(l), HipStorageSlice::F16(r)) => impl_cmp!(l, r, "f16"),
            (HipStorageSlice::F32(l), HipStorageSlice::F32(r)) => impl_cmp!(l, r, "f32"),
            (HipStorageSlice::F64(l), HipStorageSlice::F64(r)) => impl_cmp!(l, r, "f64"),
            _ => crate::bail!("dtype mismatch in cmp op"),
        };

        Ok(Self {
            slice: HipStorageSlice::U8(out_slice),
            device: self.device.clone(),
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match dtype {
                DType::U8 => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                DType::U32 => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                DType::I64 => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                DType::BF16 => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                DType::F16 => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                DType::F32 => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                DType::F64 => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("to_dtype: unsupported target dtype {:?}", dtype),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        if !layout.is_contiguous() {
            crate::bail!("to_dtype on non-contiguous HIP tensors not yet implemented");
        }

        // Same dtype is just a copy
        if self.dtype() == dtype {
            return self.try_clone(layout);
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut info_ptr: *const usize = std::ptr::null();

        // Helper to get kernel name: cast_{src}_{dst}
        let src_suffix = match self.dtype() {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            other => crate::bail!("to_dtype: unsupported source dtype {:?}", other),
        };

        let dst_suffix = match dtype {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            other => crate::bail!("to_dtype: unsupported target dtype {:?}", other),
        };

        let kernel_name = format!("cast_{}_{}", src_suffix, dst_suffix);

        macro_rules! impl_cast {
            ($src_slice:ident, $dst_ty:ty, $dst_slice:ident) => {{
                let mut inp_ptr = $src_slice.as_ptr();
                let mut out_mem: DeviceMemory<$dst_ty> = self.device.alloc::<$dst_ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 5] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("cast", &kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$dst_slice(out_mem)
            }};
        }

        let slice = match (&self.slice, dtype) {
            // From U8
            (HipStorageSlice::U8(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::U8(s), DType::I64) => impl_cast!(s, i64, I64),
            (HipStorageSlice::U8(s), DType::BF16) => impl_cast!(s, bf16, BF16),
            (HipStorageSlice::U8(s), DType::F16) => impl_cast!(s, f16, F16),
            (HipStorageSlice::U8(s), DType::F32) => impl_cast!(s, f32, F32),
            (HipStorageSlice::U8(s), DType::F64) => impl_cast!(s, f64, F64),
            // From U32
            (HipStorageSlice::U32(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::U32(s), DType::I64) => impl_cast!(s, i64, I64),
            (HipStorageSlice::U32(s), DType::BF16) => impl_cast!(s, bf16, BF16),
            (HipStorageSlice::U32(s), DType::F16) => impl_cast!(s, f16, F16),
            (HipStorageSlice::U32(s), DType::F32) => impl_cast!(s, f32, F32),
            (HipStorageSlice::U32(s), DType::F64) => impl_cast!(s, f64, F64),
            // From I64
            (HipStorageSlice::I64(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::I64(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::I64(s), DType::F32) => impl_cast!(s, f32, F32),
            (HipStorageSlice::I64(s), DType::F64) => impl_cast!(s, f64, F64),
            // From BF16
            (HipStorageSlice::BF16(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::BF16(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::BF16(s), DType::F16) => impl_cast!(s, f16, F16),
            (HipStorageSlice::BF16(s), DType::F32) => impl_cast!(s, f32, F32),
            (HipStorageSlice::BF16(s), DType::F64) => impl_cast!(s, f64, F64),
            // From F16
            (HipStorageSlice::F16(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::F16(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::F16(s), DType::BF16) => impl_cast!(s, bf16, BF16),
            (HipStorageSlice::F16(s), DType::F32) => impl_cast!(s, f32, F32),
            (HipStorageSlice::F16(s), DType::F64) => impl_cast!(s, f64, F64),
            // From F32
            (HipStorageSlice::F32(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::F32(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::F32(s), DType::I64) => impl_cast!(s, i64, I64),
            (HipStorageSlice::F32(s), DType::BF16) => impl_cast!(s, bf16, BF16),
            (HipStorageSlice::F32(s), DType::F16) => impl_cast!(s, f16, F16),
            (HipStorageSlice::F32(s), DType::F64) => impl_cast!(s, f64, F64),
            // From F64
            (HipStorageSlice::F64(s), DType::U8) => impl_cast!(s, u8, U8),
            (HipStorageSlice::F64(s), DType::U32) => impl_cast!(s, u32, U32),
            (HipStorageSlice::F64(s), DType::I64) => impl_cast!(s, i64, I64),
            (HipStorageSlice::F64(s), DType::BF16) => impl_cast!(s, bf16, BF16),
            (HipStorageSlice::F64(s), DType::F16) => impl_cast!(s, f16, F16),
            (HipStorageSlice::F64(s), DType::F32) => impl_cast!(s, f32, F32),
            _ => crate::bail!("to_dtype: unsupported cast from {:?} to {:?}", self.dtype(), dtype),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I16(_) => HipStorageSlice::I16(self.device.alloc::<i16>(0)?),
                HipStorageSlice::I32(_) => HipStorageSlice::I32(self.device.alloc::<i32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // For now, require contiguous layout
        if !layout.is_contiguous() {
            crate::bail!("unary_impl on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;  // 0 dims = contiguous
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_unary {
            ($slice:ident, $ty:ty, $suffix:expr) => {{
                let kernel_name = format!("{}_{}", U::KERNEL, $suffix);
                let mut inp_ptr = $slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 5] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut inp_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("unary", &kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::BF16(s) => HipStorageSlice::BF16(impl_unary!(s, bf16, "bf16")),
            HipStorageSlice::F16(s) => HipStorageSlice::F16(impl_unary!(s, f16, "f16")),
            HipStorageSlice::F32(s) => HipStorageSlice::F32(impl_unary!(s, f32, "f32")),
            HipStorageSlice::F64(s) => HipStorageSlice::F64(impl_unary!(s, f64, "f64")),
            _ => crate::bail!("unary ops not supported for integer types on HIP"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let numel = lhs_l.shape().elem_count();
        if numel == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I16(_) => HipStorageSlice::I16(self.device.alloc::<i16>(0)?),
                HipStorageSlice::I32(_) => HipStorageSlice::I32(self.device.alloc::<i32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // For now, require both to be contiguous
        if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
            crate::bail!("binary_impl on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;  // 0 dims = contiguous
        let mut dims_ptr: *const usize = std::ptr::null();

        macro_rules! impl_binary {
            ($lhs_slice:ident, $rhs_slice:ident, $ty:ty, $out_ty:ty, $suffix:expr) => {{
                let kernel_name = format!("{}_{}", B::KERNEL, $suffix);
                let mut lhs_ptr = $lhs_slice.as_ptr();
                let mut rhs_ptr = $rhs_slice.as_ptr();
                let mut out_mem: DeviceMemory<$out_ty> = self.device.alloc::<$out_ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 6] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut dims_ptr as *mut *const usize as *mut c_void,
                    &mut lhs_ptr as *mut *mut c_void as *mut c_void,
                    &mut rhs_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("binary", &kernel_name, grid_dim, block_dim, &mut params)?;
                out_mem
            }};
        }

        let slice = match (&self.slice, &rhs.slice) {
            (HipStorageSlice::U8(l), HipStorageSlice::U8(r)) => {
                HipStorageSlice::U8(impl_binary!(l, r, u8, u8, "u8"))
            }
            (HipStorageSlice::U32(l), HipStorageSlice::U32(r)) => {
                HipStorageSlice::U32(impl_binary!(l, r, u32, u32, "u32"))
            }
            (HipStorageSlice::I64(l), HipStorageSlice::I64(r)) => {
                HipStorageSlice::I64(impl_binary!(l, r, i64, i64, "i64"))
            }
            (HipStorageSlice::BF16(l), HipStorageSlice::BF16(r)) => {
                HipStorageSlice::BF16(impl_binary!(l, r, bf16, bf16, "bf16"))
            }
            (HipStorageSlice::F16(l), HipStorageSlice::F16(r)) => {
                HipStorageSlice::F16(impl_binary!(l, r, f16, f16, "f16"))
            }
            (HipStorageSlice::F32(l), HipStorageSlice::F32(r)) => {
                HipStorageSlice::F32(impl_binary!(l, r, f32, f32, "f32"))
            }
            (HipStorageSlice::F64(l), HipStorageSlice::F64(r)) => {
                HipStorageSlice::F64(impl_binary!(l, r, f64, f64, "f64"))
            }
            _ => crate::bail!("dtype mismatch in binary op"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn where_cond(&self, layout: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            let slice = match &t.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("where_cond: unsupported dtype"),
            };
            return Ok(Self {
                slice,
                device: self.device.clone(),
            });
        }

        // Require all tensors to be contiguous for now
        if !layout.is_contiguous() || !t_l.is_contiguous() || !f_l.is_contiguous() {
            crate::bail!("where_cond on non-contiguous HIP tensors not yet implemented");
        }

        // Condition must be U8
        let cond_slice = match &self.slice {
            HipStorageSlice::U8(s) => s,
            _ => crate::bail!("where_cond: condition must be U8"),
        };

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;
        let mut info_ptr: *const usize = std::ptr::null();

        macro_rules! impl_where {
            ($t_slice:ident, $f_slice:ident, $ty:ty, $slice_variant:ident, $kernel_name:expr) => {{
                let mut cond_ptr = cond_slice.as_ptr();
                let mut t_ptr = $t_slice.as_ptr();
                let mut f_ptr = $f_slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(numel)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 7] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut cond_ptr as *mut *mut c_void as *mut c_void,
                    &mut t_ptr as *mut *mut c_void as *mut c_void,
                    &mut f_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("ternary", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(out_mem)
            }};
        }

        let slice = match (&t.slice, &f.slice) {
            (HipStorageSlice::U8(ts), HipStorageSlice::U8(fs)) => impl_where!(ts, fs, u8, U8, "where_u8_u8"),
            (HipStorageSlice::U32(ts), HipStorageSlice::U32(fs)) => impl_where!(ts, fs, u32, U32, "where_u8_u32"),
            (HipStorageSlice::I64(ts), HipStorageSlice::I64(fs)) => impl_where!(ts, fs, i64, I64, "where_u8_i64"),
            (HipStorageSlice::BF16(ts), HipStorageSlice::BF16(fs)) => impl_where!(ts, fs, bf16, BF16, "where_u8_bf16"),
            (HipStorageSlice::F16(ts), HipStorageSlice::F16(fs)) => impl_where!(ts, fs, f16, F16, "where_u8_f16"),
            (HipStorageSlice::F32(ts), HipStorageSlice::F32(fs)) => impl_where!(ts, fs, f32, F32, "where_u8_f32"),
            (HipStorageSlice::F64(ts), HipStorageSlice::F64(fs)) => impl_where!(ts, fs, f64, F64, "where_u8_f64"),
            _ => crate::bail!("where_cond: dtype mismatch between true and false tensors"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        // Im2col-based conv1d implementation
        let device = self.device.clone();
        let shape = l.shape();
        let dims = shape.dims();
        let l_k = params.k_size;
        let stride = params.stride;
        let padding = params.padding;
        let dilation = params.dilation;
        let l_out = params.l_out();

        // Im2col1d: transform input to matrix for matmul
        let threads = dims[0] * l_out * dims[1];
        let col_slice = self.im2col1d_kernel(l, l_out, l_k, stride, padding, dilation)?;
        let col = Self {
            slice: col_slice,
            device: device.clone(),
        };

        // Matmul: col @ kernel.T
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));

        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case
            let mut kernel_c = unsafe { device.alloc_uninit(kernel_l.shape(), kernel.dtype())? };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(&kernel_c, (1, b * m, n, k), &col_l, &kernel_l)?
        };

        // Reshape result: (b, l_out, n) -> (b, n, l_out)
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let p = params;
        let l_out = p.l_out();
        let dst_el = p.c_out * l_out * p.b_size;

        let shape = l.shape();
        let dims = shape.dims();
        if dims.len() != 3 {
            crate::bail!("unexpected input shape for conv_transpose1d {:?}", dims);
        }
        let el = shape.elem_count();

        // Build info array: [inp_dims..., inp_strides..., k_dims..., k_strides...]
        let mut info_vec: Vec<usize> = Vec::with_capacity(12);
        info_vec.extend_from_slice(dims);
        info_vec.extend_from_slice(l.stride());
        info_vec.extend_from_slice(kernel_l.dims());
        info_vec.extend_from_slice(kernel_l.stride());

        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_conv_transpose1d {
            ($src_slice:ident, $k_slice:ident, $ty:ty, $slice_variant:ident, $kernel_name:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let k_ptr = unsafe { $k_slice.as_ptr().add(kernel_l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;

                let mut el_val = el;
                let mut l_out_val = l_out;
                let mut stride_val = p.stride;
                let mut padding_val = p.padding;
                let mut output_padding_val = p.output_padding;
                let mut dilation_val = p.dilation;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut k_ptr_mut = k_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut el_val as *mut usize as *mut c_void,
                    &mut l_out_val as *mut usize as *mut c_void,
                    &mut stride_val as *mut usize as *mut c_void,
                    &mut padding_val as *mut usize as *mut c_void,
                    &mut output_padding_val as *mut usize as *mut c_void,
                    &mut dilation_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut k_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match (&self.slice, &kernel.slice) {
            (HipStorageSlice::F32(src), HipStorageSlice::F32(k)) => {
                impl_conv_transpose1d!(src, k, f32, F32, "conv_transpose1d_f32")
            }
            (HipStorageSlice::F64(src), HipStorageSlice::F64(k)) => {
                impl_conv_transpose1d!(src, k, f64, F64, "conv_transpose1d_f64")
            }
            (HipStorageSlice::F16(src), HipStorageSlice::F16(k)) => {
                impl_conv_transpose1d!(src, k, f16, F16, "conv_transpose1d_f16")
            }
            (HipStorageSlice::BF16(src), HipStorageSlice::BF16(k)) => {
                impl_conv_transpose1d!(src, k, bf16, BF16, "conv_transpose1d_bf16")
            }
            _ => crate::bail!("conv_transpose1d: dtype mismatch or unsupported"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        // Im2col-based conv2d implementation
        let device = self.device.clone();
        let shape = l.shape();
        let dims = shape.dims();
        let h_k = params.k_h;
        let w_k = params.k_w;
        let stride = params.stride;
        let padding = params.padding;
        let dilation = params.dilation;
        let h_out = params.out_h();
        let w_out = params.out_w();

        // Im2col: transform input to matrix for matmul
        let col_slice = self.im2col_kernel(l, h_out, w_out, h_k, w_k, stride, padding, dilation)?;
        let col = Self {
            slice: col_slice,
            device: device.clone(),
        };

        // Matmul: col @ kernel.T
        let b = params.b_size;
        let n = params.c_out;
        let k = h_k * w_k * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));

        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case
            let mut kernel_c = unsafe { device.alloc_uninit(kernel_l.shape(), kernel.dtype())? };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(&kernel_c, (1, b * m, n, k), &col_l, &kernel_l)?
        };

        // Reshape result: (b, h_out, w_out, n) -> (b, n, h_out, w_out)
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = unsafe { device.alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let p = params;
        let (out_w, out_h) = (p.out_w(), p.out_h());
        let dst_el = p.c_out * out_w * out_h * p.b_size;

        let shape = l.shape();
        let dims = shape.dims();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for conv_transpose2d {:?}", dims);
        }
        let el = shape.elem_count();

        // Build info array: [inp_dims..., inp_strides..., k_dims..., k_strides...]
        let mut info_vec: Vec<usize> = Vec::with_capacity(16);
        info_vec.extend_from_slice(dims);
        info_vec.extend_from_slice(l.stride());
        info_vec.extend_from_slice(kernel_l.dims());
        info_vec.extend_from_slice(kernel_l.stride());

        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_conv_transpose2d {
            ($src_slice:ident, $k_slice:ident, $ty:ty, $slice_variant:ident, $kernel_name:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let k_ptr = unsafe { $k_slice.as_ptr().add(kernel_l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;

                let mut el_val = el;
                let mut out_w_val = out_w;
                let mut out_h_val = out_h;
                let mut stride_val = p.stride;
                let mut padding_val = p.padding;
                let mut output_padding_val = p.output_padding;
                let mut dilation_val = p.dilation;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut k_ptr_mut = k_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut el_val as *mut usize as *mut c_void,
                    &mut out_w_val as *mut usize as *mut c_void,
                    &mut out_h_val as *mut usize as *mut c_void,
                    &mut stride_val as *mut usize as *mut c_void,
                    &mut padding_val as *mut usize as *mut c_void,
                    &mut output_padding_val as *mut usize as *mut c_void,
                    &mut dilation_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut k_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match (&self.slice, &kernel.slice) {
            (HipStorageSlice::F32(src), HipStorageSlice::F32(k)) => {
                impl_conv_transpose2d!(src, k, f32, F32, "conv_transpose2d_f32")
            }
            (HipStorageSlice::F64(src), HipStorageSlice::F64(k)) => {
                impl_conv_transpose2d!(src, k, f64, F64, "conv_transpose2d_f64")
            }
            (HipStorageSlice::F16(src), HipStorageSlice::F16(k)) => {
                impl_conv_transpose2d!(src, k, f16, F16, "conv_transpose2d_f16")
            }
            (HipStorageSlice::BF16(src), HipStorageSlice::BF16(k)) => {
                impl_conv_transpose2d!(src, k, bf16, BF16, "conv_transpose2d_bf16")
            }
            _ => crate::bail!("conv_transpose2d: dtype mismatch or unsupported"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let shape = l.shape();
        let dims = shape.dims();
        if dims.len() != 4 {
            crate::bail!("avg_pool2d: expected 4D input, got {}D", dims.len());
        }

        let (w_k, h_k) = k;
        let (w_stride, h_stride) = stride;
        let out_w = (dims[2] - w_k) / w_stride + 1;
        let out_h = (dims[3] - h_k) / h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let src_el = shape.elem_count();

        // Prepare info buffer with dims and strides
        let info_vec: Vec<usize> = dims.iter().copied().chain(l.stride().iter().copied()).collect();
        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_avg_pool2d {
            ($src_slice:ident, $ty:ty, $slice_variant:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;

                let mut src_el_val = src_el;
                let mut w_k_val = w_k;
                let mut h_k_val = h_k;
                let mut w_stride_val = w_stride;
                let mut h_stride_val = h_stride;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut src_el_val as *mut usize as *mut c_void,
                    &mut w_k_val as *mut usize as *mut c_void,
                    &mut h_k_val as *mut usize as *mut c_void,
                    &mut w_stride_val as *mut usize as *mut c_void,
                    &mut h_stride_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::F32(src) => impl_avg_pool2d!(src, f32, F32, "avg_pool2d_f32"),
            HipStorageSlice::F64(src) => impl_avg_pool2d!(src, f64, F64, "avg_pool2d_f64"),
            HipStorageSlice::F16(src) => impl_avg_pool2d!(src, f16, F16, "avg_pool2d_f16"),
            HipStorageSlice::BF16(src) => impl_avg_pool2d!(src, bf16, BF16, "avg_pool2d_bf16"),
            HipStorageSlice::U8(src) => impl_avg_pool2d!(src, u8, U8, "avg_pool2d_u8"),
            HipStorageSlice::U32(src) => impl_avg_pool2d!(src, u32, U32, "avg_pool2d_u32"),
            _ => crate::bail!("avg_pool2d: unsupported dtype"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let shape = l.shape();
        let dims = shape.dims();
        if dims.len() != 4 {
            crate::bail!("max_pool2d: expected 4D input, got {}D", dims.len());
        }

        let (w_k, h_k) = k;
        let (w_stride, h_stride) = stride;
        let out_w = (dims[2] - w_k) / w_stride + 1;
        let out_h = (dims[3] - h_k) / h_stride + 1;
        let dst_el = out_w * out_h * dims[0] * dims[1];
        let src_el = shape.elem_count();

        // Prepare info buffer with dims and strides
        let info_vec: Vec<usize> = dims.iter().copied().chain(l.stride().iter().copied()).collect();
        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_max_pool2d {
            ($src_slice:ident, $ty:ty, $slice_variant:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;

                let mut src_el_val = src_el;
                let mut w_k_val = w_k;
                let mut h_k_val = h_k;
                let mut w_stride_val = w_stride;
                let mut h_stride_val = h_stride;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut src_el_val as *mut usize as *mut c_void,
                    &mut w_k_val as *mut usize as *mut c_void,
                    &mut h_k_val as *mut usize as *mut c_void,
                    &mut w_stride_val as *mut usize as *mut c_void,
                    &mut h_stride_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::F32(src) => impl_max_pool2d!(src, f32, F32, "max_pool2d_f32"),
            HipStorageSlice::F64(src) => impl_max_pool2d!(src, f64, F64, "max_pool2d_f64"),
            HipStorageSlice::F16(src) => impl_max_pool2d!(src, f16, F16, "max_pool2d_f16"),
            HipStorageSlice::BF16(src) => impl_max_pool2d!(src, bf16, BF16, "max_pool2d_bf16"),
            HipStorageSlice::U8(src) => impl_max_pool2d!(src, u8, U8, "max_pool2d_u8"),
            HipStorageSlice::U32(src) => impl_max_pool2d!(src, u32, U32, "max_pool2d_u32"),
            _ => crate::bail!("max_pool2d: unsupported dtype"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> Result<Self> {
        // upsample_nearest1d is not supported on CUDA either
        crate::bail!("upsample_nearest1d is not supported on HIP")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let shape = l.shape();
        let dims = shape.dims();
        if dims.len() != 4 {
            crate::bail!("upsample_nearest2d: expected 4D input, got {}D", dims.len());
        }

        let dst_el = out_w * out_h * dims[0] * dims[1];
        let scale_w = dims[2] as f64 / out_w as f64;
        let scale_h = dims[3] as f64 / out_h as f64;

        // Prepare info buffer with dims and strides
        let info_vec: Vec<usize> = dims.iter().copied().chain(l.stride().iter().copied()).collect();
        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_upsample_nearest2d {
            ($src_slice:ident, $ty:ty, $slice_variant:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(l.start_offset()) };
                let mut dst: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;

                let mut out_w_val = out_w;
                let mut out_h_val = out_h;
                let mut scale_w_val = scale_w;
                let mut scale_h_val = scale_h;
                let mut info_ptr = info_mem.as_ptr() as *const usize;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr = dst.as_mut_ptr();

                let mut params: Vec<*mut c_void> = vec![
                    &mut out_w_val as *mut usize as *mut c_void,
                    &mut out_h_val as *mut usize as *mut c_void,
                    &mut scale_w_val as *mut f64 as *mut c_void,
                    &mut scale_h_val as *mut f64 as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr as *mut _ as *mut c_void,
                ];

                self.launch_kernel("conv", $kernel, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(dst)
            }};
        }

        let slice = match &self.slice {
            HipStorageSlice::F32(src) => impl_upsample_nearest2d!(src, f32, F32, "upsample_nearest2d_f32"),
            HipStorageSlice::F64(src) => impl_upsample_nearest2d!(src, f64, F64, "upsample_nearest2d_f64"),
            HipStorageSlice::F16(src) => impl_upsample_nearest2d!(src, f16, F16, "upsample_nearest2d_f16"),
            HipStorageSlice::BF16(src) => impl_upsample_nearest2d!(src, bf16, BF16, "upsample_nearest2d_bf16"),
            HipStorageSlice::U8(src) => impl_upsample_nearest2d!(src, u8, U8, "upsample_nearest2d_u8"),
            HipStorageSlice::U32(src) => impl_upsample_nearest2d!(src, u32, U32, "upsample_nearest2d_u32"),
            _ => crate::bail!("upsample_nearest2d: unsupported dtype"),
        };

        Ok(Self {
            slice,
            device: self.device.clone(),
        })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        // Require contiguous layouts
        if !l.is_contiguous() {
            crate::bail!("index_select requires contiguous source tensor on HIP");
        }
        if !ids_l.is_contiguous() {
            crate::bail!("index_select requires contiguous ids tensor on HIP");
        }

        let src_dims = l.dims();
        let ids_dims = ids_l.dims();

        let left_size: usize = src_dims[..dim].iter().product();
        let right_size: usize = src_dims[dim + 1..].iter().product();
        let src_dim_size = src_dims[dim];
        let ids_dim_size = ids_l.shape().elem_count();

        let dst_el = ids_dim_size * left_size * right_size;
        if dst_el == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("index_select: unsupported dtype"),
            };
            return Ok(Self { slice, device: self.device.clone() });
        }

        // Build info array for ids: [ids_dims..., ids_strides...]
        let ids_strides: Vec<usize> = ids_l.stride().to_vec();
        let mut info_vec: Vec<usize> = Vec::with_capacity(ids_dims.len() * 2);
        info_vec.extend_from_slice(ids_dims);
        info_vec.extend_from_slice(&ids_strides);

        let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
        info_mem.copy_from_host(&info_vec)?;

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(dst_el, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = dst_el;
        let mut num_dims_val = ids_dims.len();
        let mut info_ptr: *const usize = info_mem.as_ptr() as *const usize;
        let mut left_size_val = left_size;
        let mut src_dim_size_val = src_dim_size;
        let mut ids_dim_size_val = ids_dim_size;
        let mut right_size_val = right_size;

        // Get ids type suffix and pointer
        let ids_suffix = match ids.dtype() {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
            _ => crate::bail!("index_select: ids must be u8, u32, or i64"),
        };

        macro_rules! impl_index_select {
            ($src_slice:ident, $ids_slice:ident, $ty:ty, $slice_variant:ident, $suffix:expr) => {{
                let kernel_name = format!("is_{}_{}", ids_suffix, $suffix);
                let mut ids_ptr = $ids_slice.as_ptr();
                let mut src_ptr = $src_slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(dst_el)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 10] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims_val as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut ids_ptr as *mut *mut c_void as *mut c_void,
                    &mut src_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                    &mut left_size_val as *mut usize as *mut c_void,
                    &mut src_dim_size_val as *mut usize as *mut c_void,
                    &mut ids_dim_size_val as *mut usize as *mut c_void,
                    &mut right_size_val as *mut usize as *mut c_void,
                ];
                self.launch_kernel("indexing", &kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(out_mem)
            }};
        }

        let slice = match (&self.slice, &ids.slice) {
            (HipStorageSlice::F32(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, f32, F32, "f32"),
            (HipStorageSlice::F32(s), HipStorageSlice::U32(i)) => impl_index_select!(s, i, f32, F32, "f32"),
            (HipStorageSlice::F64(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, f64, F64, "f64"),
            (HipStorageSlice::F64(s), HipStorageSlice::U32(i)) => impl_index_select!(s, i, f64, F64, "f64"),
            (HipStorageSlice::BF16(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, bf16, BF16, "bf16"),
            (HipStorageSlice::BF16(s), HipStorageSlice::U32(i)) => impl_index_select!(s, i, bf16, BF16, "bf16"),
            (HipStorageSlice::F16(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, f16, F16, "f16"),
            (HipStorageSlice::F16(s), HipStorageSlice::U32(i)) => impl_index_select!(s, i, f16, F16, "f16"),
            (HipStorageSlice::U32(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, u32, U32, "u32"),
            (HipStorageSlice::I64(s), HipStorageSlice::I64(i)) => impl_index_select!(s, i, i64, I64, "i64"),
            _ => crate::bail!("index_select: unsupported dtype combination"),
        };

        Ok(Self { slice, device: self.device.clone() })
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        // Require contiguous layouts
        if !l.is_contiguous() {
            crate::bail!("gather requires contiguous source tensor on HIP");
        }
        if !ids_l.is_contiguous() {
            crate::bail!("gather requires contiguous ids tensor on HIP");
        }

        let src_dims = l.dims();
        let ids_dims = ids_l.dims();

        let el = ids_l.shape().elem_count();
        if el == 0 {
            let slice = match &self.slice {
                HipStorageSlice::U8(_) => HipStorageSlice::U8(self.device.alloc::<u8>(0)?),
                HipStorageSlice::U32(_) => HipStorageSlice::U32(self.device.alloc::<u32>(0)?),
                HipStorageSlice::I64(_) => HipStorageSlice::I64(self.device.alloc::<i64>(0)?),
                HipStorageSlice::BF16(_) => HipStorageSlice::BF16(self.device.alloc::<bf16>(0)?),
                HipStorageSlice::F16(_) => HipStorageSlice::F16(self.device.alloc::<f16>(0)?),
                HipStorageSlice::F32(_) => HipStorageSlice::F32(self.device.alloc::<f32>(0)?),
                HipStorageSlice::F64(_) => HipStorageSlice::F64(self.device.alloc::<f64>(0)?),
                _ => crate::bail!("gather: unsupported dtype"),
            };
            return Ok(Self { slice, device: self.device.clone() });
        }

        let left_sz: usize = src_dims[..dim].iter().product();
        let right_sz: usize = src_dims[dim + 1..].iter().product();
        let src_dim_sz = src_dims[dim];
        let ids_dim_sz = ids_dims[dim];

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(el, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = el;
        let mut left_sz_val = left_sz;
        let mut src_dim_sz_val = src_dim_sz;
        let mut ids_dim_sz_val = ids_dim_sz;
        let mut right_sz_val = right_sz;

        // Get ids type suffix
        let ids_suffix = match ids.dtype() {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
            _ => crate::bail!("gather: ids must be u8, u32, or i64"),
        };

        macro_rules! impl_gather {
            ($src_slice:ident, $ids_slice:ident, $ty:ty, $slice_variant:ident, $suffix:expr) => {{
                let kernel_name = format!("gather_{}_{}", ids_suffix, $suffix);
                let mut ids_ptr = $ids_slice.as_ptr();
                let mut src_ptr = $src_slice.as_ptr();
                let mut out_mem: DeviceMemory<$ty> = self.device.alloc::<$ty>(el)?;
                let mut out_ptr = out_mem.as_mut_ptr();

                let mut params: [*mut c_void; 8] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut ids_ptr as *mut *mut c_void as *mut c_void,
                    &mut src_ptr as *mut *mut c_void as *mut c_void,
                    &mut out_ptr as *mut *mut c_void as *mut c_void,
                    &mut left_sz_val as *mut usize as *mut c_void,
                    &mut src_dim_sz_val as *mut usize as *mut c_void,
                    &mut ids_dim_sz_val as *mut usize as *mut c_void,
                    &mut right_sz_val as *mut usize as *mut c_void,
                ];
                self.launch_kernel("indexing", &kernel_name, grid_dim, block_dim, &mut params)?;
                HipStorageSlice::$slice_variant(out_mem)
            }};
        }

        let slice = match (&self.slice, &ids.slice) {
            (HipStorageSlice::F32(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, f32, F32, "f32"),
            (HipStorageSlice::F32(s), HipStorageSlice::U32(i)) => impl_gather!(s, i, f32, F32, "f32"),
            (HipStorageSlice::F64(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, f64, F64, "f64"),
            (HipStorageSlice::F64(s), HipStorageSlice::U32(i)) => impl_gather!(s, i, f64, F64, "f64"),
            (HipStorageSlice::BF16(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, bf16, BF16, "bf16"),
            (HipStorageSlice::BF16(s), HipStorageSlice::U32(i)) => impl_gather!(s, i, bf16, BF16, "bf16"),
            (HipStorageSlice::F16(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, f16, F16, "f16"),
            (HipStorageSlice::F16(s), HipStorageSlice::U32(i)) => impl_gather!(s, i, f16, F16, "f16"),
            (HipStorageSlice::U32(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, u32, U32, "u32"),
            (HipStorageSlice::I64(s), HipStorageSlice::I64(i)) => impl_gather!(s, i, i64, I64, "i64"),
            _ => crate::bail!("gather: unsupported dtype combination"),
        };

        Ok(Self { slice, device: self.device.clone() })
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if !l.is_contiguous() {
            crate::bail!("scatter_set requires contiguous dst tensor on HIP");
        }
        if !ids_l.is_contiguous() {
            crate::bail!("scatter_set requires contiguous ids tensor on HIP");
        }
        if !src_l.is_contiguous() {
            crate::bail!("scatter_set requires contiguous src tensor on HIP");
        }

        let src_dims = src_l.dims();
        let left_sz: usize = src_dims[..dim].iter().product();
        let right_sz: usize = src_dims[dim + 1..].iter().product();
        let src_dim_sz = src_dims[dim];
        let dst_dim_sz = l.dims()[dim];

        let block_size = 256u32;
        let numel = left_sz * right_sz;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_scatter {
            ($ids_slice:ident, $src_slice:ident, $dst_slice:ident, $ty:ty, $kernel:expr) => {{
                let ids_ptr = unsafe { $ids_slice.as_ptr().add(ids_l.start_offset()) };
                let src_ptr = unsafe { $src_slice.as_ptr().add(src_l.start_offset()) };
                let dst_ptr = unsafe { $dst_slice.as_mut_ptr().add(l.start_offset()) };

                let mut ids_ptr_val = ids_ptr;
                let mut src_ptr_val = src_ptr;
                let mut dst_ptr_val = dst_ptr;
                let mut left_sz_val = left_sz;
                let mut src_dim_sz_val = src_dim_sz;
                let mut dst_dim_sz_val = dst_dim_sz;
                let mut right_sz_val = right_sz;

                let mut params: Vec<*mut c_void> = vec![
                    &mut ids_ptr_val as *mut _ as *mut c_void,
                    &mut src_ptr_val as *mut _ as *mut c_void,
                    &mut dst_ptr_val as *mut _ as *mut c_void,
                    &mut left_sz_val as *mut usize as *mut c_void,
                    &mut src_dim_sz_val as *mut usize as *mut c_void,
                    &mut dst_dim_sz_val as *mut usize as *mut c_void,
                    &mut right_sz_val as *mut usize as *mut c_void,
                ];

                self.launch_kernel("indexing", $kernel, grid_dim, block_dim, &mut params)?;
            }};
        }

        match (&ids.slice, &src.slice, &mut self.slice) {
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f32, "s_u32_f32");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f64, "s_u32_f64");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, bf16, "s_u32_bf16");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f16, "s_u32_f16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f32, "s_i64_f32");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f64, "s_i64_f64");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, bf16, "s_i64_bf16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f16, "s_i64_f16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f32, "s_u8_f32");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f64, "s_u8_f64");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, bf16, "s_u8_bf16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter!(ids_s, src_s, dst_s, f16, "s_u8_f16");
            }
            _ => crate::bail!("scatter_set: unsupported dtype combination"),
        }

        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if !l.is_contiguous() {
            crate::bail!("scatter_add requires contiguous dst tensor on HIP");
        }
        if !ids_l.is_contiguous() {
            crate::bail!("scatter_add requires contiguous ids tensor on HIP");
        }
        if !src_l.is_contiguous() {
            crate::bail!("scatter_add requires contiguous src tensor on HIP");
        }

        let src_dims = src_l.dims();
        let left_sz: usize = src_dims[..dim].iter().product();
        let right_sz: usize = src_dims[dim + 1..].iter().product();
        let src_dim_sz = src_dims[dim];
        let dst_dim_sz = l.dims()[dim];

        let block_size = 256u32;
        let numel = left_sz * right_sz;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_scatter_add {
            ($ids_slice:ident, $src_slice:ident, $dst_slice:ident, $ty:ty, $kernel:expr) => {{
                let ids_ptr = unsafe { $ids_slice.as_ptr().add(ids_l.start_offset()) };
                let src_ptr = unsafe { $src_slice.as_ptr().add(src_l.start_offset()) };
                let dst_ptr = unsafe { $dst_slice.as_mut_ptr().add(l.start_offset()) };

                let mut ids_ptr_val = ids_ptr;
                let mut src_ptr_val = src_ptr;
                let mut dst_ptr_val = dst_ptr;
                let mut left_sz_val = left_sz;
                let mut src_dim_sz_val = src_dim_sz;
                let mut dst_dim_sz_val = dst_dim_sz;
                let mut right_sz_val = right_sz;

                let mut params: Vec<*mut c_void> = vec![
                    &mut ids_ptr_val as *mut _ as *mut c_void,
                    &mut src_ptr_val as *mut _ as *mut c_void,
                    &mut dst_ptr_val as *mut _ as *mut c_void,
                    &mut left_sz_val as *mut usize as *mut c_void,
                    &mut src_dim_sz_val as *mut usize as *mut c_void,
                    &mut dst_dim_sz_val as *mut usize as *mut c_void,
                    &mut right_sz_val as *mut usize as *mut c_void,
                ];

                self.launch_kernel("indexing", $kernel, grid_dim, block_dim, &mut params)?;
            }};
        }

        match (&ids.slice, &src.slice, &mut self.slice) {
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f32, "sa_u32_f32");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f64, "sa_u32_f64");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, bf16, "sa_u32_bf16");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f16, "sa_u32_f16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f32, "sa_i64_f32");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f64, "sa_i64_f64");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, bf16, "sa_i64_bf16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f16, "sa_i64_f16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f32, "sa_u8_f32");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f64, "sa_u8_f64");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, bf16, "sa_u8_bf16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_scatter_add!(ids_s, src_s, dst_s, f16, "sa_u8_f16");
            }
            _ => crate::bail!("scatter_add: unsupported dtype combination"),
        }

        Ok(())
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !l.is_contiguous() {
            crate::bail!("index_add requires contiguous dst tensor on HIP");
        }
        if !ids_l.is_contiguous() {
            crate::bail!("index_add requires contiguous ids tensor on HIP");
        }
        if !src_l.is_contiguous() {
            crate::bail!("index_add requires contiguous src tensor on HIP");
        }

        // Copy self into output first
        let mut acc = unsafe { self.device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;

        let src_dims = src_l.dims();
        let left_sz: usize = src_dims[..dim].iter().product();
        let right_sz: usize = src_dims[dim + 1..].iter().product();
        let src_dim_sz = src_dims[dim];
        let dst_dim_sz = l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];

        let block_size = 256u32;
        let numel = left_sz * right_sz;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_index_add {
            ($ids_slice:ident, $src_slice:ident, $dst_slice:ident, $ty:ty, $kernel:expr) => {{
                let ids_ptr = unsafe { $ids_slice.as_ptr().add(ids_l.start_offset()) };
                let src_ptr = unsafe { $src_slice.as_ptr().add(src_l.start_offset()) };
                let dst_ptr = $dst_slice.as_mut_ptr();

                let mut ids_ptr_val = ids_ptr;
                let mut ids_dim_sz_val = ids_dim_sz;
                let mut src_ptr_val = src_ptr;
                let mut dst_ptr_val = dst_ptr;
                let mut left_sz_val = left_sz;
                let mut src_dim_sz_val = src_dim_sz;
                let mut dst_dim_sz_val = dst_dim_sz;
                let mut right_sz_val = right_sz;

                let mut params: Vec<*mut c_void> = vec![
                    &mut ids_ptr_val as *mut _ as *mut c_void,
                    &mut ids_dim_sz_val as *mut usize as *mut c_void,
                    &mut src_ptr_val as *mut _ as *mut c_void,
                    &mut dst_ptr_val as *mut _ as *mut c_void,
                    &mut left_sz_val as *mut usize as *mut c_void,
                    &mut src_dim_sz_val as *mut usize as *mut c_void,
                    &mut dst_dim_sz_val as *mut usize as *mut c_void,
                    &mut right_sz_val as *mut usize as *mut c_void,
                ];

                self.launch_kernel("indexing", $kernel, grid_dim, block_dim, &mut params)?;
            }};
        }

        match (&ids.slice, &src.slice, &mut acc.slice) {
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f32, "ia_u32_f32");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f64, "ia_u32_f64");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, bf16, "ia_u32_bf16");
            }
            (HipStorageSlice::U32(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f16, "ia_u32_f16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f32, "ia_i64_f32");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f64, "ia_i64_f64");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, bf16, "ia_i64_bf16");
            }
            (HipStorageSlice::I64(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f16, "ia_i64_f16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F32(src_s), HipStorageSlice::F32(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f32, "ia_u8_f32");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F64(src_s), HipStorageSlice::F64(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f64, "ia_u8_f64");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::BF16(src_s), HipStorageSlice::BF16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, bf16, "ia_u8_bf16");
            }
            (HipStorageSlice::U8(ids_s), HipStorageSlice::F16(src_s), HipStorageSlice::F16(dst_s)) => {
                impl_index_add!(ids_s, src_s, dst_s, f16, "ia_u8_f16");
            }
            _ => crate::bail!("index_add: unsupported dtype combination"),
        }

        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let cfg = gemm_config((b, m, n, k), lhs_l, rhs_l)?;

        let slice = match (&self.slice, &rhs.slice) {
            (HipStorageSlice::F32(lhs), HipStorageSlice::F32(rhs)) => {
                let lhs_ptr = unsafe { lhs.as_ptr().add(lhs_l.start_offset()) };
                let rhs_ptr = unsafe { rhs.as_ptr().add(rhs_l.start_offset()) };
                let mut out: DeviceMemory<f32> = dev.alloc::<f32>(elem_count)?;

                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;

                unsafe {
                    ffi::check_rocblas_status(ffi::rocblas_gemm_strided_batched_ex(
                        dev.blas_handle(),
                        cfg.transa,
                        cfg.transb,
                        cfg.m,
                        cfg.n,
                        cfg.k,
                        &alpha as *const f32 as *const c_void,
                        rhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        cfg.lda,
                        cfg.stride_a,
                        lhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        cfg.ldb,
                        cfg.stride_b,
                        &beta as *const f32 as *const c_void,
                        out.as_mut_ptr() as *const c_void,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        cfg.ldc,
                        cfg.stride_c,
                        out.as_mut_ptr() as *mut c_void,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        cfg.ldc,
                        cfg.stride_c,
                        cfg.batch_count,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        ROCBLAS_GEMM_FLAGS_NONE,
                    ))
                    .map_err(|e| HipError::Hip(format!("rocBLAS gemm failed: {}", e)))
                    .w()?;
                }
                HipStorageSlice::F32(out)
            }
            (HipStorageSlice::F64(lhs), HipStorageSlice::F64(rhs)) => {
                let lhs_ptr = unsafe { lhs.as_ptr().add(lhs_l.start_offset()) };
                let rhs_ptr = unsafe { rhs.as_ptr().add(rhs_l.start_offset()) };
                let mut out: DeviceMemory<f64> = dev.alloc::<f64>(elem_count)?;

                let alpha: f64 = 1.0;
                let beta: f64 = 0.0;

                unsafe {
                    ffi::check_rocblas_status(ffi::rocblas_gemm_strided_batched_ex(
                        dev.blas_handle(),
                        cfg.transa,
                        cfg.transb,
                        cfg.m,
                        cfg.n,
                        cfg.k,
                        &alpha as *const f64 as *const c_void,
                        rhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f64_r,
                        cfg.lda,
                        cfg.stride_a,
                        lhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f64_r,
                        cfg.ldb,
                        cfg.stride_b,
                        &beta as *const f64 as *const c_void,
                        out.as_mut_ptr() as *const c_void,
                        rocblas_datatype::rocblas_datatype_f64_r,
                        cfg.ldc,
                        cfg.stride_c,
                        out.as_mut_ptr() as *mut c_void,
                        rocblas_datatype::rocblas_datatype_f64_r,
                        cfg.ldc,
                        cfg.stride_c,
                        cfg.batch_count,
                        rocblas_datatype::rocblas_datatype_f64_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        ROCBLAS_GEMM_FLAGS_NONE,
                    ))
                    .map_err(|e| HipError::Hip(format!("rocBLAS gemm failed: {}", e)))
                    .w()?;
                }
                HipStorageSlice::F64(out)
            }
            (HipStorageSlice::F16(lhs), HipStorageSlice::F16(rhs)) => {
                let lhs_ptr = unsafe { lhs.as_ptr().add(lhs_l.start_offset()) };
                let rhs_ptr = unsafe { rhs.as_ptr().add(rhs_l.start_offset()) };
                let mut out: DeviceMemory<f16> = dev.alloc::<f16>(elem_count)?;

                // Use f32 for alpha/beta with f16 compute
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;

                unsafe {
                    ffi::check_rocblas_status(ffi::rocblas_gemm_strided_batched_ex(
                        dev.blas_handle(),
                        cfg.transa,
                        cfg.transb,
                        cfg.m,
                        cfg.n,
                        cfg.k,
                        &alpha as *const f32 as *const c_void,
                        rhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        cfg.lda,
                        cfg.stride_a,
                        lhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        cfg.ldb,
                        cfg.stride_b,
                        &beta as *const f32 as *const c_void,
                        out.as_mut_ptr() as *const c_void,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        cfg.ldc,
                        cfg.stride_c,
                        out.as_mut_ptr() as *mut c_void,
                        rocblas_datatype::rocblas_datatype_f16_r,
                        cfg.ldc,
                        cfg.stride_c,
                        cfg.batch_count,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        ROCBLAS_GEMM_FLAGS_NONE,
                    ))
                    .map_err(|e| HipError::Hip(format!("rocBLAS gemm failed: {}", e)))
                    .w()?;
                }
                HipStorageSlice::F16(out)
            }
            (HipStorageSlice::BF16(lhs), HipStorageSlice::BF16(rhs)) => {
                let lhs_ptr = unsafe { lhs.as_ptr().add(lhs_l.start_offset()) };
                let rhs_ptr = unsafe { rhs.as_ptr().add(rhs_l.start_offset()) };
                let mut out: DeviceMemory<bf16> = dev.alloc::<bf16>(elem_count)?;

                // Use f32 for alpha/beta with bf16 compute
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;

                unsafe {
                    ffi::check_rocblas_status(ffi::rocblas_gemm_strided_batched_ex(
                        dev.blas_handle(),
                        cfg.transa,
                        cfg.transb,
                        cfg.m,
                        cfg.n,
                        cfg.k,
                        &alpha as *const f32 as *const c_void,
                        rhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_bf16_r,
                        cfg.lda,
                        cfg.stride_a,
                        lhs_ptr as *const c_void,
                        rocblas_datatype::rocblas_datatype_bf16_r,
                        cfg.ldb,
                        cfg.stride_b,
                        &beta as *const f32 as *const c_void,
                        out.as_mut_ptr() as *const c_void,
                        rocblas_datatype::rocblas_datatype_bf16_r,
                        cfg.ldc,
                        cfg.stride_c,
                        out.as_mut_ptr() as *mut c_void,
                        rocblas_datatype::rocblas_datatype_bf16_r,
                        cfg.ldc,
                        cfg.stride_c,
                        cfg.batch_count,
                        rocblas_datatype::rocblas_datatype_f32_r,
                        rocblas_gemm_algo::rocblas_gemm_algo_standard,
                        0,
                        ROCBLAS_GEMM_FLAGS_NONE,
                    ))
                    .map_err(|e| HipError::Hip(format!("rocBLAS gemm failed: {}", e)))
                    .w()?;
                }
                HipStorageSlice::BF16(out)
            }
            _ => crate::bail!("matmul: dtype mismatch or unsupported dtype"),
        };

        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(el_count, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_copy_strided {
            ($src_slice:ident, $dst_slice:ident, $ty:ty, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(src_l.start_offset()) };
                let dst_ptr = unsafe { $dst_slice.as_mut_ptr().add(dst_offset) };

                if src_l.is_contiguous() {
                    // Direct memcpy for contiguous
                    let size = el_count * std::mem::size_of::<$ty>();
                    unsafe {
                        ffi::check_hip_error(ffi::hipMemcpy(
                            dst_ptr as *mut c_void,
                            src_ptr as *const c_void,
                            size,
                            ffi::hipMemcpyKind::hipMemcpyDeviceToDevice,
                        ))
                        .map_err(|e| HipError::Hip(format!("copy_strided_src memcpy failed: {}", e)))
                        .w()?;
                    }
                } else {
                    // Use ucopy kernel for non-contiguous
                    let num_dims = dims.len();
                    let dims_vec: Vec<usize> = dims.to_vec();
                    let strides_vec: Vec<usize> = src_l.stride().to_vec();
                    let mut info_vec: Vec<usize> = Vec::with_capacity(num_dims * 2);
                    info_vec.extend_from_slice(&dims_vec);
                    info_vec.extend_from_slice(&strides_vec);
                    let mut info_mem: DeviceMemory<usize> = self.device.alloc::<usize>(info_vec.len())?;
                    info_mem.copy_from_host(&info_vec)?;

                    let mut el_count_val = el_count;
                    let mut num_dims_val = num_dims;
                    let mut info_ptr = info_mem.as_ptr() as *const usize;
                    let mut src_ptr_mut = src_ptr;
                    let mut dst_ptr_mut = dst_ptr;

                    let mut params: Vec<*mut c_void> = vec![
                        &mut el_count_val as *mut usize as *mut c_void,
                        &mut num_dims_val as *mut usize as *mut c_void,
                        &mut info_ptr as *mut *const usize as *mut c_void,
                        &mut src_ptr_mut as *mut _ as *mut c_void,
                        &mut dst_ptr_mut as *mut _ as *mut c_void,
                    ];

                    self.launch_kernel("unary", $kernel, grid_dim, block_dim, &mut params)?;
                }
            }};
        }

        match (&self.slice, &mut dst.slice) {
            (HipStorageSlice::F32(src), HipStorageSlice::F32(dst)) => {
                impl_copy_strided!(src, dst, f32, "ucopy_f32")
            }
            (HipStorageSlice::F64(src), HipStorageSlice::F64(dst)) => {
                impl_copy_strided!(src, dst, f64, "ucopy_f64")
            }
            (HipStorageSlice::F16(src), HipStorageSlice::F16(dst)) => {
                impl_copy_strided!(src, dst, f16, "ucopy_f16")
            }
            (HipStorageSlice::BF16(src), HipStorageSlice::BF16(dst)) => {
                impl_copy_strided!(src, dst, bf16, "ucopy_bf16")
            }
            (HipStorageSlice::U8(src), HipStorageSlice::U8(dst)) => {
                impl_copy_strided!(src, dst, u8, "ucopy_u8")
            }
            (HipStorageSlice::U32(src), HipStorageSlice::U32(dst)) => {
                impl_copy_strided!(src, dst, u32, "ucopy_u32")
            }
            (HipStorageSlice::I64(src), HipStorageSlice::I64(dst)) => {
                impl_copy_strided!(src, dst, i64, "ucopy_i64")
            }
            _ => crate::bail!("copy_strided_src: dtype mismatch"),
        }

        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(d1 * d2, block_size);
        let block_dim = (block_size, 1, 1);

        macro_rules! impl_copy2d {
            ($src_slice:ident, $dst_slice:ident, $kernel:expr) => {{
                let src_ptr = unsafe { $src_slice.as_ptr().add(src_o) };
                let dst_ptr = unsafe { $dst_slice.as_mut_ptr().add(dst_o) };

                let mut d1_val = d1 as u32;
                let mut d2_val = d2 as u32;
                let mut src_s_val = src_s as u32;
                let mut dst_s_val = dst_s as u32;
                let mut src_ptr_mut = src_ptr;
                let mut dst_ptr_mut = dst_ptr;

                let mut params: Vec<*mut c_void> = vec![
                    &mut src_ptr_mut as *mut _ as *mut c_void,
                    &mut dst_ptr_mut as *mut _ as *mut c_void,
                    &mut d1_val as *mut u32 as *mut c_void,
                    &mut d2_val as *mut u32 as *mut c_void,
                    &mut src_s_val as *mut u32 as *mut c_void,
                    &mut dst_s_val as *mut u32 as *mut c_void,
                ];

                self.launch_kernel("fill", $kernel, grid_dim, block_dim, &mut params)?;
            }};
        }

        match (&self.slice, &mut dst.slice) {
            (HipStorageSlice::F32(src), HipStorageSlice::F32(dst)) => {
                impl_copy2d!(src, dst, "copy2d_f32")
            }
            (HipStorageSlice::F64(src), HipStorageSlice::F64(dst)) => {
                impl_copy2d!(src, dst, "copy2d_f64")
            }
            (HipStorageSlice::F16(src), HipStorageSlice::F16(dst)) => {
                impl_copy2d!(src, dst, "copy2d_f16")
            }
            (HipStorageSlice::BF16(src), HipStorageSlice::BF16(dst)) => {
                impl_copy2d!(src, dst, "copy2d_bf16")
            }
            (HipStorageSlice::U8(src), HipStorageSlice::U8(dst)) => {
                impl_copy2d!(src, dst, "copy2d_u8")
            }
            (HipStorageSlice::U32(src), HipStorageSlice::U32(dst)) => {
                impl_copy2d!(src, dst, "copy2d_u32")
            }
            (HipStorageSlice::I64(src), HipStorageSlice::I64(dst)) => {
                impl_copy2d!(src, dst, "copy2d_i64")
            }
            _ => crate::bail!("copy2d: dtype mismatch"),
        }

        Ok(())
    }

    fn const_set(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let numel = layout.shape().elem_count();
        if numel == 0 {
            return Ok(());
        }

        // For contiguous layouts, we can pass null for info and the kernel will
        // use the fast path. For non-contiguous layouts, we need to copy info to device.
        if !layout.is_contiguous() {
            crate::bail!("const_set on non-contiguous HIP tensors not yet implemented");
        }

        let block_size = 256u32;
        let grid_dim = Self::grid_dims(numel, block_size);
        let block_dim = (block_size, 1, 1);

        let mut numel_val = numel;
        let mut num_dims: usize = 0;  // 0 dims = contiguous

        macro_rules! impl_const_set {
            ($slice:ident, $ty:ty, $kernel_name:expr, $convert:expr) => {{
                let mut val: $ty = $convert;
                let mut ptr = $slice.as_mut_ptr();  // *mut c_void
                let mut info_ptr: *const usize = std::ptr::null();  // null for contiguous
                let mut params: [*mut c_void; 5] = [
                    &mut numel_val as *mut usize as *mut c_void,
                    &mut num_dims as *mut usize as *mut c_void,
                    &mut info_ptr as *mut *const usize as *mut c_void,
                    &mut val as *mut $ty as *mut c_void,
                    &mut ptr as *mut *mut c_void as *mut c_void,
                ];
                self.launch_kernel("fill", $kernel_name, grid_dim, block_dim, &mut params)?;
            }};
        }

        let f64_val = value.to_f64();
        match &mut self.slice {
            HipStorageSlice::U8(s) => impl_const_set!(s, u8, "const_set_u8", f64_val as u8),
            HipStorageSlice::U32(s) => impl_const_set!(s, u32, "const_set_u32", f64_val as u32),
            HipStorageSlice::I64(s) => impl_const_set!(s, i64, "const_set_i64", f64_val as i64),
            HipStorageSlice::BF16(s) => impl_const_set!(s, bf16, "const_set_bf16", bf16::from_f64(f64_val)),
            HipStorageSlice::F16(s) => impl_const_set!(s, f16, "const_set_f16", f16::from_f64(f64_val)),
            HipStorageSlice::F32(s) => impl_const_set!(s, f32, "const_set_f32", f64_val as f32),
            HipStorageSlice::F64(s) => impl_const_set!(s, f64, "const_set_f64", f64_val),
            HipStorageSlice::I16(_) | HipStorageSlice::I32(_) => {
                crate::bail!("const_set not supported for I16/I32 on HIP")
            }
        }

        Ok(())
    }
}
