//! Vulkan compute kernels for Candle
//!
//! This crate provides Vulkan backend compute kernels following the same
//! patterns as candle-metal-kernels.

mod context;
mod error;
mod kernels;
pub mod ops;

pub use context::VulkanContext;
pub use error::{VulkanError, Result};
pub use kernels::Kernels;

/// Data types supported by kernels
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
            Self::BF16 | Self::F16 => 2,
            Self::U32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}
