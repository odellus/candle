//! Minimal HIP FFI bindings for candle
//!
//! These bindings provide only what candle needs, avoiding version compatibility
//! issues with full ROCm bindings.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

/// HIP error type
pub type hipError_t = c_int;

/// HIP device pointer
pub type hipDeviceptr_t = *mut c_void;

/// HIP stream
pub type hipStream_t = *mut c_void;

/// HIP module (loaded kernel code)
pub type hipModule_t = *mut c_void;

/// HIP function (kernel)
pub type hipFunction_t = *mut c_void;

/// HIP memory copy kind
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
}

/// HIP device properties (simplified)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct hipDeviceProp_t {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multi_processor_count: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub compute_mode: c_int,
    pub clock_instruction_rate: c_int,
    // ... more fields exist but we don't need them
    _padding: [u8; 512],
}

impl Default for hipDeviceProp_t {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

/// hipSuccess constant
pub const HIP_SUCCESS: hipError_t = 0;

// Link to HIP runtime library
#[link(name = "amdhip64")]
extern "C" {
    // Device management
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDevice(device_id: *mut c_int) -> hipError_t;
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device_id: c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;

    // Memory management
    pub fn hipMalloc(ptr: *mut hipDeviceptr_t, size: usize) -> hipError_t;
    pub fn hipFree(ptr: hipDeviceptr_t) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemset(dst: hipDeviceptr_t, value: c_int, size: usize) -> hipError_t;
    pub fn hipMemsetAsync(
        dst: hipDeviceptr_t,
        value: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    // Stream management
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    // Module management
    pub fn hipModuleLoad(module: *mut hipModule_t, fname: *const c_char) -> hipError_t;
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;

    // Kernel launching
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    // Error handling
    pub fn hipGetErrorName(error: hipError_t) -> *const c_char;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}

/// Safe wrapper to check HIP errors
#[inline]
pub fn check_hip_error(error: hipError_t) -> Result<(), String> {
    if error == HIP_SUCCESS {
        Ok(())
    } else {
        let name = unsafe {
            let ptr = hipGetErrorName(error);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        let desc = unsafe {
            let ptr = hipGetErrorString(error);
            if ptr.is_null() {
                "No description".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(format!("HIP error {}: {} - {}", error, name, desc))
    }
}

// ============================================================================
// rocBLAS FFI bindings
// ============================================================================

/// rocBLAS status type
pub type rocblas_status = c_int;

/// rocBLAS handle
pub type rocblas_handle = *mut c_void;

/// rocBLAS int type (32-bit)
pub type rocblas_int = i32;

/// rocBLAS stride type (64-bit)
pub type rocblas_stride = i64;

/// rocBLAS operation type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

/// rocBLAS datatype
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rocblas_datatype {
    rocblas_datatype_f16_r = 150,
    rocblas_datatype_f32_r = 151,
    rocblas_datatype_f64_r = 152,
    rocblas_datatype_f16_c = 153,
    rocblas_datatype_f32_c = 154,
    rocblas_datatype_f64_c = 155,
    rocblas_datatype_i8_r = 160,
    rocblas_datatype_u8_r = 161,
    rocblas_datatype_i32_r = 162,
    rocblas_datatype_u32_r = 163,
    rocblas_datatype_i8_c = 164,
    rocblas_datatype_u8_c = 165,
    rocblas_datatype_i32_c = 166,
    rocblas_datatype_u32_c = 167,
    rocblas_datatype_bf16_r = 168,
    rocblas_datatype_bf16_c = 169,
}

/// rocBLAS GEMM algorithm
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rocblas_gemm_algo {
    rocblas_gemm_algo_standard = 0,
    rocblas_gemm_algo_solution_index = 1,
}

/// rocBLAS status codes
pub const ROCBLAS_STATUS_SUCCESS: rocblas_status = 0;
pub const ROCBLAS_STATUS_INVALID_HANDLE: rocblas_status = 1;
pub const ROCBLAS_STATUS_NOT_IMPLEMENTED: rocblas_status = 2;
pub const ROCBLAS_STATUS_INVALID_POINTER: rocblas_status = 3;
pub const ROCBLAS_STATUS_INVALID_SIZE: rocblas_status = 4;
pub const ROCBLAS_STATUS_MEMORY_ERROR: rocblas_status = 5;
pub const ROCBLAS_STATUS_INTERNAL_ERROR: rocblas_status = 6;

/// rocBLAS GEMM flags
pub const ROCBLAS_GEMM_FLAGS_NONE: u32 = 0;

// Link to rocBLAS library
#[link(name = "rocblas")]
extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
    pub fn rocblas_set_stream(handle: rocblas_handle, stream: hipStream_t) -> rocblas_status;

    pub fn rocblas_gemm_strided_batched_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: rocblas_int,
        n: rocblas_int,
        k: rocblas_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: rocblas_int,
        stride_a: rocblas_stride,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: rocblas_int,
        stride_b: rocblas_stride,
        beta: *const c_void,
        c: *const c_void,
        c_type: rocblas_datatype,
        ldc: rocblas_int,
        stride_c: rocblas_stride,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: rocblas_int,
        stride_d: rocblas_stride,
        batch_count: rocblas_int,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: i32,
        flags: u32,
    ) -> rocblas_status;
}

/// Safe wrapper to check rocBLAS errors
#[inline]
pub fn check_rocblas_status(status: rocblas_status) -> Result<(), String> {
    match status {
        ROCBLAS_STATUS_SUCCESS => Ok(()),
        ROCBLAS_STATUS_INVALID_HANDLE => Err("rocBLAS: invalid handle".to_string()),
        ROCBLAS_STATUS_NOT_IMPLEMENTED => Err("rocBLAS: not implemented".to_string()),
        ROCBLAS_STATUS_INVALID_POINTER => Err("rocBLAS: invalid pointer".to_string()),
        ROCBLAS_STATUS_INVALID_SIZE => Err("rocBLAS: invalid size".to_string()),
        ROCBLAS_STATUS_MEMORY_ERROR => Err("rocBLAS: memory error".to_string()),
        ROCBLAS_STATUS_INTERNAL_ERROR => Err("rocBLAS: internal error".to_string()),
        code => Err(format!("rocBLAS: unknown error code {}", code)),
    }
}
