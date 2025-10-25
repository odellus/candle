//! Vulkan kernels for Candle ML framework
//!
//! This crate provides Vulkan backend implementation for Candle following the same
//! architecture as the Metal backend.

pub mod quant_types;

pub use quant_types::{BlockQ4_0, GgmlDType};
