//! Vulkan related errors

use thiserror::Error;

/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

/// Vulkan related errors
#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    AshError(#[from] ash::vk::Result),
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
    #[error("Buffer mapping failed: {0}")]
    BufferMapping(String),
    #[error("Invalid device index: {0}")]
    InvalidDeviceIndex(usize),
    #[error("Unsupported Vulkan extension: {0}")]
    UnsupportedExtension(String),
    #[error("Unsupported Vulkan feature: {0}")]
    UnsupportedFeature(String),
    #[error("Descriptor set layout binding {binding} out of range for pipeline layout")]
    DescriptorBindingOutOfRange { binding: u32 },
    #[error("Invalid dtype for operation: expected {expected:?}, got {got:?}")]
    InvalidDType { expected: DType, got: DType },
    #[error("Lock error: {0}")]
    LockError(LockError),
    #[error("Failed to load Vulkan entry points")]
    EntryLoadingFailed,
    #[error("String contains null bytes")]
    NullError,
    #[error("Shader compiler not available")]
    CompilerNotAvailable,
    #[error("No Vulkan devices found")]
    NoVulkanDevices,
    #[error("No compute queue family found")]
    NoComputeQueueFamily,
}

impl From<std::ffi::NulError> for VulkanError {
    fn from(_: std::ffi::NulError) -> Self {
        VulkanError::NullError
    }
}

impl From<ash::LoadingError> for VulkanError {
    fn from(_: ash::LoadingError) -> Self {
        VulkanError::EntryLoadingFailed
    }
}

impl From<shaderc::Error> for VulkanError {
    fn from(e: shaderc::Error) -> Self {
        VulkanError::ShaderCompilation(e.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    U8,
    U32,
    I64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

pub type Result<T> = std::result::Result<T, VulkanError>;
