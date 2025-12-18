//! Safe wrappers for HIP device memory

use super::ffi::{self, hipDeviceptr_t, hipMemcpyKind, hipStream_t};
use super::{HipError, WrapErr};
use crate::Result;
use std::ffi::c_void;
use std::marker::PhantomData;

// Helper to convert HipError Result to crate::Result
fn wrap_err<T>(r: std::result::Result<T, HipError>) -> Result<T> {
    r.w()
}

/// A safe wrapper around HIP device memory allocation.
///
/// This type owns a region of memory on the HIP device and will
/// free it when dropped.
#[derive(Debug)]
pub struct DeviceMemory<T> {
    ptr: hipDeviceptr_t,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceMemory<T> {
    /// Allocate device memory for `len` elements of type T.
    pub fn alloc(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }

        let size = len * std::mem::size_of::<T>();
        let mut ptr: hipDeviceptr_t = std::ptr::null_mut();

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMalloc(&mut ptr, size))
                    .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Returns the number of elements this memory can hold.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the memory region is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns the raw device pointer.
    pub fn as_ptr(&self) -> hipDeviceptr_t {
        self.ptr
    }

    /// Returns the raw device pointer as a mutable pointer.
    pub fn as_mut_ptr(&mut self) -> hipDeviceptr_t {
        self.ptr
    }

    /// Copy data from host to this device memory.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            crate::bail!("size mismatch: host has {} elements, device has {}", src.len(), self.len);
        }

        if self.len == 0 {
            return Ok(());
        }

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMemcpy(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    self.size_bytes(),
                    hipMemcpyKind::hipMemcpyHostToDevice,
                ))
                .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(())
    }

    /// Copy data from host to this device memory asynchronously.
    pub fn copy_from_host_async(&mut self, src: &[T], stream: hipStream_t) -> Result<()> {
        if src.len() != self.len {
            crate::bail!("size mismatch: host has {} elements, device has {}", src.len(), self.len);
        }

        if self.len == 0 {
            return Ok(());
        }

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMemcpyAsync(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    self.size_bytes(),
                    hipMemcpyKind::hipMemcpyHostToDevice,
                    stream,
                ))
                .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(())
    }

    /// Copy data from this device memory to host.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            crate::bail!("size mismatch: host has {} elements, device has {}", dst.len(), self.len);
        }

        if self.len == 0 {
            return Ok(());
        }

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMemcpy(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    self.size_bytes(),
                    hipMemcpyKind::hipMemcpyDeviceToHost,
                ))
                .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(())
    }

    /// Copy data from this device memory to a new Vec.
    pub fn copy_to_vec(&self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut vec = vec![T::default(); self.len];
        self.copy_to_host(&mut vec)?;
        Ok(vec)
    }

    /// Fill memory with zeros.
    pub fn memset_zero(&mut self) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMemset(self.ptr, 0, self.size_bytes()))
                    .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(())
    }

    /// Fill memory with a byte value.
    pub fn memset(&mut self, value: i32) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        unsafe {
            wrap_err(
                ffi::check_hip_error(ffi::hipMemset(self.ptr, value, self.size_bytes()))
                    .map_err(|e| HipError::Memory(e))
            )?;
        }

        Ok(())
    }
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                // Ignore errors during drop
                let _ = ffi::hipFree(self.ptr);
            }
        }
    }
}

// DeviceMemory is Send because the pointer can be safely transferred between threads
// (the GPU memory is global and not thread-local)
unsafe impl<T: Send> Send for DeviceMemory<T> {}

// DeviceMemory is Sync because read-only access from multiple threads is safe
// (actual GPU operations are serialized by the HIP runtime)
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}
