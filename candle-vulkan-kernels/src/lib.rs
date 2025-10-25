//! Vulkan kernels for Candle ML framework
//!
//! This crate provides Vulkan backend implementation for Candle following the same
//! architecture as the Metal backend.

pub mod error;
pub mod device;
pub mod storage;
pub mod quant_types;
pub mod kernels;

pub use error::{VulkanError, Result};
pub use device::{VulkanDevice, DeviceInfo};
pub use storage::VulkanStorage;
pub use quant_types::{BlockQ4_0, GgmlDType};
pub use kernels::{Kernels, KernelHandle};
