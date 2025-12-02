//! Dummy Vulkan quantized storage when vulkan feature is not enabled
use super::GgmlDType;
use crate::{Result, Shape};

pub struct QVulkanStorage;

impl QVulkanStorage {
    pub fn zeros(
        _device: &crate::dummy_vulkan_backend::VulkanDevice,
        _elem_count: usize,
        _dtype: GgmlDType,
    ) -> Result<Self> {
        unreachable!()
    }

    pub fn dtype(&self) -> GgmlDType {
        unreachable!()
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<crate::dummy_vulkan_backend::VulkanStorage> {
        unreachable!()
    }

    pub fn quantize(&mut self, _src: &crate::dummy_vulkan_backend::VulkanStorage) -> Result<()> {
        unreachable!()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        unreachable!()
    }

    pub fn device(&self) -> &crate::dummy_vulkan_backend::VulkanDevice {
        unreachable!()
    }

    pub fn fwd(
        &self,
        _self_shape: &Shape,
        _storage: &crate::dummy_vulkan_backend::VulkanStorage,
        _layout: &crate::Layout,
    ) -> Result<(crate::dummy_vulkan_backend::VulkanStorage, Shape)> {
        unreachable!()
    }
}
