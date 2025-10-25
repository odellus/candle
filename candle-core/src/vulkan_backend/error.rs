use crate::Error;

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("Vulkan error: {0}")]
    VkError(#[from] ash::vk::Result),

    #[error("GPU allocator error: {0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),

    #[error("Kernel error: {0}")]
    KernelError(#[from] candle_vulkan_kernels::VulkanError),

    #[error("Message: {0}")]
    Message(String),

    #[error("Unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: crate::DType, op: &'static str },

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<String> for VulkanError {
    fn from(s: String) -> Self {
        VulkanError::Message(s)
    }
}

impl From<&str> for VulkanError {
    fn from(s: &str) -> Self {
        VulkanError::Message(s.to_string())
    }
}

pub trait WrapErr<T> {
    fn w(self) -> Result<T, Error>;
}

impl<T, E: Into<VulkanError>> WrapErr<T> for Result<T, E> {
    fn w(self) -> Result<T, Error> {
        self.map_err(|e| Error::Vulkan(Box::new(e.into())))
    }
}

impl<T> WrapErr<T> for Option<T> {
    fn w(self) -> Result<T, Error> {
        self.ok_or_else(|| Error::Vulkan(Box::new(VulkanError::Message("None value".to_string()))))
    }
}
