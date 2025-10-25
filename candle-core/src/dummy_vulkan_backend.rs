use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Error, Result, Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId;

impl DeviceId {
    pub(crate) fn new() -> Self {
        DeviceId
    }
}

#[derive(Debug, Clone)]
pub struct VulkanDevice;

#[derive(Debug, Clone)]
pub struct VulkanStorage;

impl VulkanStorage {
    pub fn dtype(&self) -> DType {
        unreachable!()
    }

    pub fn device(&self) -> &VulkanDevice {
        unreachable!()
    }

    pub fn to_cpu_storage(&self) -> Result<CpuStorage> {
        unreachable!()
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        unreachable!()
    }

    fn same_device(&self, _: &Self) -> bool {
        unreachable!()
    }

    fn zeros_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    unsafe fn alloc_uninit(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}
