//! HIP (ROCm) Backend Examples
//!
//! This example demonstrates the HIP backend for AMD GPUs.
//! Run with: cargo run --example hip_basics --features hip

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    println!("=== Candle HIP (ROCm) Backend Demo ===\n");

    // Create HIP device
    let device = Device::new_hip(0)?;
    println!("HIP device created successfully!");

    // Basic tensor creation
    println!("\n--- Basic Tensor Operations ---");
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;
    let b = Tensor::new(&[[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]], &device)?;
    println!("Tensor a:\n{:?}", a.to_vec2::<f32>()?);
    println!("Tensor b:\n{:?}", b.to_vec2::<f32>()?);

    // Element-wise operations
    let sum = (&a + &b)?;
    let diff = (&a - &b)?;
    let prod = (&a * &b)?;
    let div = (&a / &b)?;
    println!("\na + b:\n{:?}", sum.to_vec2::<f32>()?);
    println!("a - b:\n{:?}", diff.to_vec2::<f32>()?);
    println!("a * b:\n{:?}", prod.to_vec2::<f32>()?);
    println!("a / b:\n{:?}", div.to_vec2::<f32>()?);

    // Unary operations
    println!("\n--- Unary Operations ---");
    let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    println!("x: {:?}", x.to_vec1::<f32>()?);
    println!("exp(x): {:?}", x.exp()?.to_vec1::<f32>()?);
    println!("log(x): {:?}", x.log()?.to_vec1::<f32>()?);
    println!("sqrt(x): {:?}", x.sqrt()?.to_vec1::<f32>()?);
    println!("tanh(x): {:?}", x.tanh()?.to_vec1::<f32>()?);

    // Matrix multiplication with rocBLAS
    println!("\n--- Matrix Multiplication (rocBLAS) ---");
    let m1 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let m2 = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
    println!("M1:\n{:?}", m1.to_vec2::<f32>()?);
    println!("M2:\n{:?}", m2.to_vec2::<f32>()?);
    let result = m1.matmul(&m2)?;
    println!("M1 @ M2:\n{:?}", result.to_vec2::<f32>()?);

    // Larger matmul benchmark
    println!("\n--- Matmul Performance ---");
    let size = 512;
    let mat_a = Tensor::randn(0f32, 1.0, (size, size), &device)?;
    let mat_b = Tensor::randn(0f32, 1.0, (size, size), &device)?;

    // Warmup
    let _ = mat_a.matmul(&mat_b)?;
    device.synchronize()?;

    // Timed runs
    let iterations = 10;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = mat_a.matmul(&mat_b)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_secs_f64() / iterations as f64;
    let gflops = (2.0 * (size as f64).powi(3)) / avg_time / 1e9;
    println!(
        "{}x{} matmul: {:.3}ms avg ({:.1} GFLOPS)",
        size,
        size,
        avg_time * 1000.0,
        gflops
    );

    // Reduction operations
    println!("\n--- Reduction Operations ---");
    let data = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)?;
    println!("data: {:?}", data.to_vec1::<f32>()?);
    println!("sum: {}", data.sum_all()?.to_scalar::<f32>()?);
    println!("max: {}", data.max_all()?.to_scalar::<f32>()?);
    println!("min: {}", data.min_all()?.to_scalar::<f32>()?);

    // Type casting
    println!("\n--- Type Casting ---");
    let float_tensor = Tensor::new(&[1.5f32, 2.7, 3.9], &device)?;
    println!("f32: {:?}", float_tensor.to_vec1::<f32>()?);
    let f64_tensor = float_tensor.to_dtype(DType::F64)?;
    println!("f64: {:?}", f64_tensor.to_vec1::<f64>()?);

    // Comparison operations
    println!("\n--- Comparison Operations ---");
    let v1 = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let v2 = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &device)?;
    println!("v1: {:?}", v1.to_vec1::<f32>()?);
    println!("v2: {:?}", v2.to_vec1::<f32>()?);
    println!("v1 < v2: {:?}", v1.lt(&v2)?.to_vec1::<u8>()?);
    println!("v1 == v2: {:?}", v1.eq(&v2)?.to_vec1::<u8>()?);
    println!("v1 > v2: {:?}", v1.gt(&v2)?.to_vec1::<u8>()?);

    device.synchronize()?;
    println!("\n=== HIP operations completed successfully! ===");

    Ok(())
}
