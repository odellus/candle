//! Vulkan Backend Basics
//!
//! A minimal example showing end-to-end tensor operations on Vulkan.
//!
//! Run with:
//!   cargo run --example vulkan_basics --features vulkan

use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    // Initialize Vulkan
    let device = Device::new_vulkan(0)?;
    println!("Vulkan device initialized");

    // Create tensors
    let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
    println!("A = {}", a.to_string());
    println!("B = {}", b.to_string());

    // Binary ops
    let sum = (&a + &b)?;
    let product = (&a * &b)?;
    println!("A + B = {}", sum.to_string());
    println!("A * B = {}", product.to_string());

    // Unary ops
    let exp_a = a.exp()?;
    let relu_b = b.relu()?;
    println!("exp(A) = {}", exp_a.to_string());
    println!("relu(B) = {}", relu_b.to_string());

    // Transfer back to CPU and verify
    let sum_cpu: Vec<f32> = sum.flatten_all()?.to_vec1()?;
    println!("Sum values on CPU: {:?}", sum_cpu);

    Ok(())
}
