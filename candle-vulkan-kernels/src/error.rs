//! Error types for Vulkan kernels

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("No Vulkan devices found")]
    NoDevices,

    #[error("Invalid device index: {0}")]
    InvalidDeviceIndex(usize),

    #[error("No compute queue family found")]
    NoComputeQueue,

    #[error("Failed to load shader: {0}")]
    ShaderLoad(String),

    #[error("Failed to create pipeline: {0}")]
    PipelineCreation(String),

    #[error("Buffer error: {0}")]
    Buffer(String),

    #[error("Kernel execution error: {0}")]
    Execution(String),

    #[error("Allocator error: {0}")]
    Allocator(String),
}

pub type Result<T> = std::result::Result<T, VulkanError>;
