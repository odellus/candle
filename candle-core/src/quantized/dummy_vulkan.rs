//! Dummy Vulkan quantized storage when vulkan feature is not enabled
use super::GgmlDType;
use crate::Result;

pub struct QVulkanStorage;

impl QVulkanStorage {
    pub fn dtype(&self) -> GgmlDType {
        unreachable!()
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<crate::dummy_vulkan_backend::VulkanStorage> {
        unreachable!()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        unreachable!()
    }

    pub fn device(&self) -> &crate::dummy_vulkan_backend::VulkanDevice {
        unreachable!()
    }
}
