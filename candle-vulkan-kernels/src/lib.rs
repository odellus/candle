//! Vulkan kernels for Candle ML framework
//!
//! This crate provides Vulkan backend implementation for Candle following the same
//! architecture as the Metal backend.

pub mod device;
pub mod kernels;
pub mod quant_types;
pub mod storage;

pub use device::VulkanContext;
pub use kernels::Kernels;
pub use quant_types::{BlockQ4_0, GgmlDType};
pub use storage::VulkanStorage;
