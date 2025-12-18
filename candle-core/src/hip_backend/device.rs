//! HIP Device implementation

use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Result, Shape};
use half::{bf16, f16};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Arc, RwLock};

use super::ffi::{self, hipFunction_t, hipModule_t, hipStream_t, rocblas_handle};
use super::memory::DeviceMemory;
use super::{HipError, HipStorage, HipStorageSlice, WrapErr};

/// Unique identifier for HIP devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// Safe wrapper around HIP stream
#[derive(Debug)]
pub struct Stream {
    stream: hipStream_t,
}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut stream: hipStream_t = std::ptr::null_mut();
        unsafe {
            ffi::check_hip_error(ffi::hipStreamCreate(&mut stream))
                .map_err(|e| HipError::Hip(e))
                .w()?;
        }
        Ok(Self { stream })
    }

    pub fn as_ptr(&self) -> hipStream_t {
        self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            ffi::check_hip_error(ffi::hipStreamSynchronize(self.stream))
                .map_err(|e| HipError::Hip(e))
                .w()?;
        }
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = ffi::hipStreamDestroy(self.stream);
            }
        }
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// Safe wrapper around HIP module
#[derive(Debug)]
pub struct Module {
    module: hipModule_t,
}

impl Module {
    pub fn load_data(data: &[u8]) -> Result<Self> {
        let mut module: hipModule_t = std::ptr::null_mut();
        unsafe {
            ffi::check_hip_error(ffi::hipModuleLoadData(
                &mut module,
                data.as_ptr() as *const c_void,
            ))
            .map_err(|e| HipError::ModuleLoad(e))
            .w()?;
        }
        Ok(Self { module })
    }

    pub fn get_function(&self, name: &str) -> Result<Function> {
        let c_name = CString::new(name)
            .map_err(|_| HipError::Internal(format!("Invalid function name: {}", name)));
        let c_name = c_name.w()?;
        let mut func: hipFunction_t = std::ptr::null_mut();
        unsafe {
            ffi::check_hip_error(ffi::hipModuleGetFunction(
                &mut func,
                self.module,
                c_name.as_ptr(),
            ))
            .map_err(|e| HipError::KernelLaunch(format!("Failed to get function {}: {}", name, e)))
            .w()?;
        }
        Ok(Function { func })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = ffi::hipModuleUnload(self.module);
            }
        }
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// Safe wrapper around HIP function
#[derive(Debug, Clone, Copy)]
pub struct Function {
    func: hipFunction_t,
}

impl Function {
    pub fn as_ptr(&self) -> hipFunction_t {
        self.func
    }
}

pub struct ModuleStore {
    modules: HashMap<String, Arc<Module>>,
}

impl ModuleStore {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }
}

/// Safe wrapper around rocBLAS handle
pub struct RocBlasHandle {
    handle: rocblas_handle,
}

impl RocBlasHandle {
    pub fn new(stream: &Stream) -> Result<Self> {
        let mut handle: rocblas_handle = std::ptr::null_mut();
        unsafe {
            ffi::check_rocblas_status(ffi::rocblas_create_handle(&mut handle))
                .map_err(|e| HipError::Hip(format!("Failed to create rocBLAS handle: {}", e)))
                .w()?;
            ffi::check_rocblas_status(ffi::rocblas_set_stream(handle, stream.as_ptr()))
                .map_err(|e| HipError::Hip(format!("Failed to set rocBLAS stream: {}", e)))
                .w()?;
        }
        Ok(Self { handle })
    }

    pub fn handle(&self) -> rocblas_handle {
        self.handle
    }
}

impl Drop for RocBlasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::rocblas_destroy_handle(self.handle);
            }
        }
    }
}

unsafe impl Send for RocBlasHandle {}
unsafe impl Sync for RocBlasHandle {}

#[derive(Clone)]
pub struct HipDevice {
    id: DeviceId,
    ordinal: usize,
    modules: Arc<RwLock<ModuleStore>>,
    stream: Arc<Stream>,
    blas: Arc<RocBlasHandle>,
    seed_value: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for HipDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipDevice({:?})", self.id)
    }
}

impl HipDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        // Set the device
        unsafe {
            ffi::check_hip_error(ffi::hipSetDevice(ordinal as i32))
                .map_err(|e| HipError::Hip(format!("Failed to set device: {}", e)))
                .w()?;
        }

        let stream = Stream::new()?;
        let blas = RocBlasHandle::new(&stream)?;

        Ok(Self {
            id: DeviceId::new(),
            ordinal,
            modules: Arc::new(RwLock::new(ModuleStore::new())),
            stream: Arc::new(stream),
            blas: Arc::new(blas),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }

    pub fn blas_handle(&self) -> rocblas_handle {
        self.blas.handle()
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Get the number of HIP devices
    pub fn device_count() -> Result<usize> {
        let mut count: i32 = 0;
        unsafe {
            ffi::check_hip_error(ffi::hipGetDeviceCount(&mut count))
                .map_err(|e| HipError::Hip(e))
                .w()?;
        }
        Ok(count as usize)
    }

    /// Allocate device memory
    pub fn alloc<T>(&self, len: usize) -> Result<DeviceMemory<T>> {
        DeviceMemory::alloc(len)
    }

    /// Allocate zeroed device memory
    pub fn alloc_zeros<T: Default + Clone>(&self, len: usize) -> Result<DeviceMemory<T>> {
        let mut mem = self.alloc::<T>(len)?;
        mem.memset_zero()?;
        Ok(mem)
    }

    /// Copy from host to device
    pub fn memcpy_htod<T: Clone>(&self, src: &[T], dst: &mut DeviceMemory<T>) -> Result<()> {
        dst.copy_from_host(src)
    }

    /// Copy from device to host
    pub fn memcpy_dtoh<T: Clone + Default>(&self, src: &DeviceMemory<T>, dst: &mut [T]) -> Result<()> {
        src.copy_to_host(dst)
    }

    /// Copy device to vector
    pub fn memcpy_dtov<T: Clone + Default>(&self, src: &DeviceMemory<T>) -> Result<Vec<T>> {
        src.copy_to_vec()
    }

    /// Copy device to device
    pub fn memcpy_dtod<T>(&self, src: &DeviceMemory<T>, dst: &mut DeviceMemory<T>, len: usize) -> Result<()> {
        let size = len * std::mem::size_of::<T>();
        unsafe {
            ffi::check_hip_error(ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                size,
                ffi::hipMemcpyKind::hipMemcpyDeviceToDevice,
            ))
            .map_err(|e| HipError::Hip(format!("Device to device copy failed: {}", e)))
            .w()?;
        }
        Ok(())
    }

    /// Load a kernel module from HSACO data
    pub fn load_module(&self, name: &str, hsaco: &[u8]) -> Result<Arc<Module>> {
        let mut modules = self.modules.write().unwrap();

        if let Some(m) = modules.modules.get(name) {
            return Ok(m.clone());
        }

        let module = Module::load_data(hsaco)?;
        let module = Arc::new(module);
        modules.modules.insert(name.to_string(), module.clone());
        Ok(module)
    }

    /// Get a function from a module
    pub fn get_func(&self, module: &Module, name: &str) -> Result<Function> {
        module.get_function(name)
    }

    /// Synchronize the device
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            ffi::check_hip_error(ffi::hipDeviceSynchronize())
                .map_err(|e| HipError::Hip(e))
                .w()?;
        }
        Ok(())
    }
}

impl BackendDevice for HipDevice {
    type Storage = HipStorage;

    fn new(ordinal: usize) -> Result<Self> {
        HipDevice::new(ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Hip { gpu_id: self.ordinal }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                HipStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                HipStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                HipStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                HipStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                HipStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                HipStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                HipStorageSlice::F64(data)
            }
            dtype => crate::bail!("dtype {dtype:?} is not supported on HIP"),
        };
        Ok(HipStorage {
            slice: storage,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<HipStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => HipStorageSlice::U8(self.alloc::<u8>(elem_count)?),
            DType::U32 => HipStorageSlice::U32(self.alloc::<u32>(elem_count)?),
            DType::I64 => HipStorageSlice::I64(self.alloc::<i64>(elem_count)?),
            DType::BF16 => HipStorageSlice::BF16(self.alloc::<bf16>(elem_count)?),
            DType::F16 => HipStorageSlice::F16(self.alloc::<f16>(elem_count)?),
            DType::F32 => HipStorageSlice::F32(self.alloc::<f32>(elem_count)?),
            DType::F64 => HipStorageSlice::F64(self.alloc::<f64>(elem_count)?),
            dtype => crate::bail!("dtype {dtype:?} is not supported on HIP"),
        };
        Ok(HipStorage {
            slice: storage,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<HipStorage> {
        let slice = HipStorageSlice::from_host_slice(self, data)?;
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<HipStorage> {
        let slice = HipStorageSlice::from_cpu_storage(self, storage)?;
        Ok(HipStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<HipStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<HipStorage> {
        // TODO: Implement proper RNG on device
        // For now, generate on CPU and copy
        use rand::Rng;
        let elem_count = shape.elem_count();
        let mut rng = rand::rng();

        let storage = match dtype {
            DType::F32 => {
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| rng.random_range(lo as f32..up as f32))
                    .collect();
                let mut mem = self.alloc::<f32>(elem_count)?;
                self.memcpy_htod(&data, &mut mem)?;
                HipStorageSlice::F32(mem)
            }
            DType::F64 => {
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| rng.random_range(lo..up))
                    .collect();
                let mut mem = self.alloc::<f64>(elem_count)?;
                self.memcpy_htod(&data, &mut mem)?;
                HipStorageSlice::F64(mem)
            }
            dtype => crate::bail!("rand_uniform not supported for dtype {dtype:?}"),
        };
        Ok(HipStorage {
            slice: storage,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<HipStorage> {
        // TODO: Implement proper RNG on device
        use rand_distr::{Distribution, Normal};
        let elem_count = shape.elem_count();
        let mut rng = rand::rng();

        let storage = match dtype {
            DType::F32 => {
                let normal = Normal::new(mean as f32, std as f32)
                    .map_err(|e| crate::Error::wrap(HipError::Internal(format!("{:?}", e))))?;
                let data: Vec<f32> = (0..elem_count)
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                let mut mem = self.alloc::<f32>(elem_count)?;
                self.memcpy_htod(&data, &mut mem)?;
                HipStorageSlice::F32(mem)
            }
            DType::F64 => {
                let normal = Normal::new(mean, std)
                    .map_err(|e| crate::Error::wrap(HipError::Internal(format!("{:?}", e))))?;
                let data: Vec<f64> = (0..elem_count)
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                let mut mem = self.alloc::<f64>(elem_count)?;
                self.memcpy_htod(&data, &mut mem)?;
                HipStorageSlice::F64(mem)
            }
            dtype => crate::bail!("rand_normal not supported for dtype {dtype:?}"),
        };
        Ok(HipStorage {
            slice: storage,
            device: self.clone(),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        HipDevice::synchronize(self)
    }
}
