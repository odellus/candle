//! HIP (ROCm) Neural Network Operations Example
//!
//! This example demonstrates neural network operations on the HIP backend for AMD GPUs.
//! Run with: cargo run --example hip_neural_net --features hip

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

fn main() -> Result<()> {
    println!("=== Candle HIP Neural Network Operations Demo ===\n");

    // Create HIP device
    let device = Device::new_hip(0)?;
    println!("HIP device created successfully!");

    // Convolution 1D
    println!("\n--- Conv1D ---");
    let input_1d = Tensor::new(
        &[[[1.0f32, 2.0, 3.0, 4.0, 5.0]]],  // (batch=1, channels=1, length=5)
        &device,
    )?;
    let kernel_1d = Tensor::new(
        &[[[1.0f32, 0.5]]],  // (out_channels=1, in_channels=1, kernel=2)
        &device,
    )?;
    println!("Input shape: {:?}", input_1d.dims());
    println!("Kernel shape: {:?}", kernel_1d.dims());
    let conv1d_out = input_1d.conv1d(&kernel_1d, 0, 1, 1, 1)?;
    println!("Conv1D output: {:?}", conv1d_out.flatten_all()?.to_vec1::<f32>()?);
    println!("Output shape: {:?}", conv1d_out.dims());

    // Convolution 2D
    println!("\n--- Conv2D ---");
    let input_2d = Tensor::new(
        &[[[[1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]]]],  // (batch=1, channels=1, height=3, width=3)
        &device,
    )?;
    let kernel_2d = Tensor::new(
        &[[[[1.0f32, 0.0],
            [0.0, 1.0]]]],  // (out_channels=1, in_channels=1, kh=2, kw=2)
        &device,
    )?;
    println!("Input shape: {:?}", input_2d.dims());
    println!("Kernel shape: {:?}", kernel_2d.dims());
    let conv2d_out = input_2d.conv2d(&kernel_2d, 0, 1, 1, 1)?;
    println!("Conv2D output:\n{:?}", conv2d_out.flatten_all()?.to_vec1::<f32>()?);
    println!("Output shape: {:?}", conv2d_out.dims());

    // Pooling Operations
    println!("\n--- Pooling Operations ---");
    let pool_input = Tensor::new(
        &[[[[1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]]]],  // (batch=1, channels=1, h=4, w=4)
        &device,
    )?;
    println!("Pool input shape: {:?}", pool_input.dims());

    let avg_pooled = pool_input.avg_pool2d((2, 2))?;
    println!("Avg Pool 2x2: {:?}", avg_pooled.flatten_all()?.to_vec1::<f32>()?);

    let max_pooled = pool_input.max_pool2d((2, 2))?;
    println!("Max Pool 2x2: {:?}", max_pooled.flatten_all()?.to_vec1::<f32>()?);

    // Upsampling
    println!("\n--- Upsampling ---");
    let upsample_input = Tensor::new(
        &[[[[1.0f32, 2.0],
            [3.0, 4.0]]]],  // (batch=1, channels=1, h=2, w=2)
        &device,
    )?;
    println!("Upsample input: {:?}", upsample_input.flatten_all()?.to_vec1::<f32>()?);
    let upsampled = upsample_input.upsample_nearest2d(4, 4)?;
    println!("Upsampled to 4x4:");
    for i in 0..4 {
        let row: Vec<f32> = upsampled.i((0, 0, i))?.flatten_all()?.to_vec1()?;
        println!("  {:?}", row);
    }

    // Transposed Convolutions
    println!("\n--- Transposed Convolutions ---");
    let conv_t_input = Tensor::new(
        &[[[1.0f32, 2.0, 3.0]]],  // (batch=1, channels=1, length=3)
        &device,
    )?;
    let conv_t_kernel = Tensor::new(
        &[[[1.0f32, 1.0]]],  // (in_channels=1, out_channels=1, kernel=2)
        &device,
    )?;
    println!("ConvTranspose1D input: {:?}", conv_t_input.flatten_all()?.to_vec1::<f32>()?);
    let conv_t1d_out = conv_t_input.conv_transpose1d(&conv_t_kernel, 0, 0, 1, 1, 1)?;
    println!("ConvTranspose1D output: {:?}", conv_t1d_out.flatten_all()?.to_vec1::<f32>()?);
    println!("Output shape: {:?}", conv_t1d_out.dims());

    // Index Operations
    println!("\n--- Index Operations ---");
    let data = Tensor::zeros((5, 3), DType::F32, &device)?;
    let src = Tensor::new(
        &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],  // 2x3
        &device,
    )?;
    let indices = Tensor::new(&[1u32, 3], &device)?;

    println!("Base tensor (zeros 5x3)");
    println!("Source values:\n{:?}", src.to_vec2::<f32>()?);
    println!("Indices: {:?}", indices.to_vec1::<u32>()?);

    let result = data.index_add(&indices, &src, 0)?;
    println!("After index_add at rows [1, 3]:");
    for i in 0..5 {
        let row: Vec<f32> = result.i(i)?.flatten_all()?.to_vec1()?;
        println!("  Row {}: {:?}", i, row);
    }

    // Scatter Add
    println!("\n--- Scatter Add ---");
    let scatter_dst = Tensor::zeros((4,), DType::F32, &device)?;
    let scatter_src = Tensor::new(&[10.0f32, 20.0, 30.0], &device)?;
    let scatter_idx = Tensor::new(&[0u32, 2, 0], &device)?;

    println!("Scatter destination (zeros): {:?}", scatter_dst.to_vec1::<f32>()?);
    println!("Scatter source: {:?}", scatter_src.to_vec1::<f32>()?);
    println!("Scatter indices: {:?}", scatter_idx.to_vec1::<u32>()?);

    let scattered = scatter_dst.scatter_add(&scatter_idx, &scatter_src, 0)?;
    println!("After scatter_add: {:?}", scattered.to_vec1::<f32>()?);
    println!("  (10+30=40 at idx 0, 20 at idx 2)");

    // Gather
    println!("\n--- Gather ---");
    let gather_src = Tensor::new(
        &[[10.0f32, 11.0, 12.0],
          [20.0, 21.0, 22.0],
          [30.0, 31.0, 32.0]],
        &device,
    )?;
    let gather_idx = Tensor::new(&[0u32, 2, 1], &device)?;
    println!("Gather source:\n{:?}", gather_src.to_vec2::<f32>()?);
    println!("Gather indices: {:?}", gather_idx.to_vec1::<u32>()?);
    let gathered = gather_src.gather(&gather_idx.unsqueeze(1)?.broadcast_as((3, 3))?.contiguous()?, 0)?;
    println!("Gathered rows: {:?}", gathered.to_vec2::<f32>()?);

    // Performance test: Conv2D
    println!("\n--- Conv2D Performance ---");
    let batch_size = 16;
    let channels = 64;
    let height = 56;
    let width = 56;
    let out_channels = 128;
    let kernel_size = 3;

    let perf_input = Tensor::randn(0f32, 1.0, (batch_size, channels, height, width), &device)?;
    let perf_kernel = Tensor::randn(0f32, 1.0, (out_channels, channels, kernel_size, kernel_size), &device)?;

    // Warmup
    let _ = perf_input.conv2d(&perf_kernel, 1, 1, 1, 1)?;
    device.synchronize()?;

    // Timed runs
    let iterations = 10;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = perf_input.conv2d(&perf_kernel, 1, 1, 1, 1)?;
    }
    device.synchronize()?;
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_secs_f64() / iterations as f64;

    println!("Input: {}x{}x{}x{}", batch_size, channels, height, width);
    println!("Kernel: {}x{}x{}x{}", out_channels, channels, kernel_size, kernel_size);
    println!("Conv2D time: {:.3}ms avg", avg_time * 1000.0);

    device.synchronize()?;
    println!("\n=== HIP neural network operations completed successfully! ===");

    Ok(())
}
