pub mod vulkan {
    pub use ash;
    pub use ash::vk;
    pub use gpu_allocator;
}

// Re-export common types
pub use ash::vk::DeviceMemory;
pub use ash::{Device, Entry, Instance};
pub use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
pub use gpu_allocator::MemoryLocation;

include!(concat!(env!("OUT_DIR"), "/compiled_shaders.rs"));

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("Vulkan error: {0}")]
    VkError(#[from] ash::vk::Result),

    #[error("Shader not found: {0}")]
    ShaderNotFound(String),

    #[error("Allocation error: {0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

pub type Result<T> = std::result::Result<T, VulkanError>;

/// Get compiled shader SPIR-V code by name
pub fn get_shader(name: &str) -> Result<&'static [u8]> {
    static SHADERS: std::sync::OnceLock<CompiledShaders> = std::sync::OnceLock::new();
    let shaders = SHADERS.get_or_init(CompiledShaders::new);

    shaders
        .get(name)
        .ok_or_else(|| VulkanError::ShaderNotFound(name.to_string()))
}

/// Available shader operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderOp {
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,

    // Unary ops
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Abs,
    Neg,
    Sqr,
    Gelu,
    Relu,
    Tanh,

    // Reduce ops
    Sum,
    ReduceMin,
    ReduceMax,
    ArgMin,
    ArgMax,
}

impl ShaderOp {
    pub fn to_op_code(&self) -> u32 {
        match self {
            // Binary ops
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Div => 3,
            Self::Min => 4,
            Self::Max => 5,

            // Unary ops
            Self::Exp => 0,
            Self::Log => 1,
            Self::Sin => 2,
            Self::Cos => 3,
            Self::Sqrt => 4,
            Self::Abs => 5,
            Self::Neg => 6,
            Self::Sqr => 7,
            Self::Gelu => 8,
            Self::Relu => 9,
            Self::Tanh => 10,

            // Reduce ops
            Self::Sum => 0,
            Self::ReduceMin => 1,
            Self::ReduceMax => 2,
            Self::ArgMin => 3,
            Self::ArgMax => 4,
        }
    }

    pub fn shader_name(&self) -> &'static str {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Min | Self::Max => "binary",
            Self::Exp | Self::Log | Self::Sin | Self::Cos | Self::Sqrt | Self::Abs
            | Self::Neg | Self::Sqr | Self::Gelu | Self::Relu | Self::Tanh => "unary",
            Self::Sum | Self::ReduceMin | Self::ReduceMax | Self::ArgMin | Self::ArgMax => "reduce",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_loading() {
        // This test will pass if shaders are compiled correctly
        assert!(get_shader("binary").is_ok() || std::fs::read_dir("src/shaders").is_err());
    }
}
