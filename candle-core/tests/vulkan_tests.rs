//! Vulkan backend integration tests
//!
//! These tests verify that Vulkan compute kernels work correctly.

#![cfg(feature = "vulkan")]

use candle_core::{DType, Device, Module, Result, Tensor};

fn get_vulkan_device() -> Result<Device> {
    Device::new_vulkan(0)
}

#[test]
fn test_vulkan_basic_transfer() -> Result<()> {
    let device = get_vulkan_device()?;
    println!("Using device: {:?}", device);

    // Create tensor on CPU, transfer to Vulkan, transfer back
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cpu_tensor = Tensor::from_vec(data.clone(), (5,), &Device::Cpu)?;
    let vulkan_tensor = cpu_tensor.to_device(&device)?;
    let result = vulkan_tensor.to_vec1::<f32>()?;

    assert_eq!(result, data);
    println!("Basic transfer test passed!");
    Ok(())
}

#[test]
fn test_vulkan_zeros() -> Result<()> {
    let device = get_vulkan_device()?;

    let tensor = Tensor::zeros((4, 4), DType::F32, &device)?;
    let result = tensor.to_vec2::<f32>()?;

    for row in &result {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    println!("Zeros test passed!");
    Ok(())
}

#[test]
fn test_vulkan_exp() -> Result<()> {
    let device = get_vulkan_device()?;

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0];
    let tensor = Tensor::from_vec(data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.exp()?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-5,
            "exp mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("Exp test passed!");
    Ok(())
}

#[test]
fn test_vulkan_silu() -> Result<()> {
    let device = get_vulkan_device()?;

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0, -2.0];
    let tensor = Tensor::from_vec(data.clone(), (5,), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.silu()?;
    let result_vec = result.to_vec1::<f32>()?;

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let expected: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-5,
            "silu mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("SiLU test passed!");
    Ok(())
}

#[test]
fn test_vulkan_gelu() -> Result<()> {
    let device = get_vulkan_device()?;

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0];
    let tensor = Tensor::from_vec(data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.gelu()?;
    let result_vec = result.to_vec1::<f32>()?;

    // Compute expected on CPU for comparison
    let cpu_tensor = Tensor::from_vec(data, (4,), &Device::Cpu)?;
    let cpu_result = cpu_tensor.gelu()?;
    let expected = cpu_result.to_vec1::<f32>()?;

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "gelu mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("GELU test passed!");
    Ok(())
}

#[test]
fn test_vulkan_relu() -> Result<()> {
    let device = get_vulkan_device()?;

    let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let tensor = Tensor::from_vec(data.clone(), (5,), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.relu()?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-6,
            "relu mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("ReLU test passed!");
    Ok(())
}

#[test]
fn test_vulkan_add() -> Result<()> {
    let device = get_vulkan_device()?;

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

    let a = Tensor::from_vec(a_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;

    let result = (&a + &b)?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();

    assert_eq!(result_vec, expected);
    println!("Add test passed!");
    Ok(())
}

#[test]
fn test_vulkan_mul() -> Result<()> {
    let device = get_vulkan_device()?;

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];

    let a = Tensor::from_vec(a_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;

    let result = (&a * &b)?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();

    assert_eq!(result_vec, expected);
    println!("Mul test passed!");
    Ok(())
}

#[test]
fn test_vulkan_div() -> Result<()> {
    let device = get_vulkan_device()?;

    let a_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let b_data: Vec<f32> = vec![2.0, 4.0, 5.0, 8.0];

    let a = Tensor::from_vec(a_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;

    let result = (&a / &b)?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a / b).collect();

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-6,
            "div mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("Div test passed!");
    Ok(())
}

#[test]
fn test_vulkan_larger_tensor() -> Result<()> {
    let device = get_vulkan_device()?;

    // Test with a larger tensor (1024 elements)
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();

    let tensor = Tensor::from_vec(data.clone(), (size,), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.exp()?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();

    for (i, (got, exp)) in result_vec.iter().zip(expected.iter()).enumerate() {
        // Use relative tolerance for larger values
        let tol = (exp.abs() * 1e-5).max(1e-4);
        assert!(
            (got - exp).abs() < tol,
            "exp mismatch at index {}: got {}, expected {}, diff {}",
            i,
            got,
            exp,
            (got - exp).abs()
        );
    }
    println!("Larger tensor exp test passed (1024 elements)!");
    Ok(())
}

#[test]
fn test_vulkan_2d_tensor() -> Result<()> {
    let device = get_vulkan_device()?;

    // Test with 2D tensor
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let tensor = Tensor::from_vec(data.clone(), (3, 4), &Device::Cpu)?.to_device(&device)?;
    let result = tensor.relu()?;
    let result_vec = result.to_vec2::<f32>()?;

    // All values are positive, so relu should be identity
    let expected = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ];

    assert_eq!(result_vec, expected);
    println!("2D tensor test passed!");
    Ok(())
}

#[test]
fn test_vulkan_chained_ops() -> Result<()> {
    let device = get_vulkan_device()?;

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let tensor = Tensor::from_vec(data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;

    // Chain: exp -> relu -> add with itself
    let after_exp = tensor.exp()?;
    let after_relu = after_exp.relu()?;
    let result = (&after_relu + &after_relu)?;
    let result_vec = result.to_vec1::<f32>()?;

    // Compute expected
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| {
            let e = x.exp();
            let r = e.max(0.0);
            r + r
        })
        .collect();

    for (got, exp) in result_vec.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "chained ops mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("Chained ops test passed!");
    Ok(())
}

// ============ Strided tensor tests ============

#[test]
fn test_vulkan_strided_unary_transpose() -> Result<()> {
    let device = get_vulkan_device()?;

    // Create a 2D tensor and transpose it (creates non-contiguous view)
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, (2, 3), &Device::Cpu)?.to_device(&device)?;

    // Transpose creates a strided tensor
    let transposed = tensor.t()?;

    // Apply exp to the strided tensor
    let result = transposed.exp()?;
    let result_vec = result.to_vec2::<f32>()?;

    // Expected: exp of transposed values
    // Original: [[1,2,3], [4,5,6]]
    // Transposed: [[1,4], [2,5], [3,6]]
    let expected = vec![
        vec![1.0_f32.exp(), 4.0_f32.exp()],
        vec![2.0_f32.exp(), 5.0_f32.exp()],
        vec![3.0_f32.exp(), 6.0_f32.exp()],
    ];

    for (i, (got_row, exp_row)) in result_vec.iter().zip(expected.iter()).enumerate() {
        for (j, (got, exp)) in got_row.iter().zip(exp_row.iter()).enumerate() {
            // Use relative tolerance for larger values (exp can produce large numbers)
            let tol = 1e-5 * exp.abs().max(1.0);
            assert!(
                (got - exp).abs() < tol,
                "strided exp mismatch at [{},{}]: got {}, expected {}, diff {}",
                i,
                j,
                got,
                exp,
                (got - exp).abs()
            );
        }
    }
    println!("Strided unary (transpose) test passed!");
    Ok(())
}

#[test]
fn test_vulkan_strided_unary_narrow() -> Result<()> {
    let device = get_vulkan_device()?;

    // Create a tensor and narrow it (creates non-contiguous view)
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(data, (4, 6), &Device::Cpu)?.to_device(&device)?;

    // Narrow on dimension 1: take columns 1-3 (creates strided view)
    let narrowed = tensor.narrow(1, 1, 3)?;

    // Apply relu to the strided tensor
    let result = narrowed.relu()?;
    let result_vec = result.to_vec2::<f32>()?;

    // Expected values
    let expected = vec![
        vec![1.0, 2.0, 3.0],
        vec![7.0, 8.0, 9.0],
        vec![13.0, 14.0, 15.0],
        vec![19.0, 20.0, 21.0],
    ];

    assert_eq!(result_vec, expected);
    println!("Strided unary (narrow) test passed!");
    Ok(())
}

#[test]
fn test_vulkan_strided_silu() -> Result<()> {
    let device = get_vulkan_device()?;

    // Test SiLU on a transposed tensor
    let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let tensor = Tensor::from_vec(data.clone(), (2, 3), &Device::Cpu)?.to_device(&device)?;
    let transposed = tensor.t()?;

    let result = transposed.silu()?;
    let result_vec = result.to_vec2::<f32>()?;

    // Compute expected on CPU with same transposition
    let cpu_tensor = Tensor::from_vec(data, (2, 3), &Device::Cpu)?;
    let cpu_transposed = cpu_tensor.t()?;
    let cpu_result = cpu_transposed.silu()?;
    let expected = cpu_result.to_vec2::<f32>()?;

    for (i, (got_row, exp_row)) in result_vec.iter().zip(expected.iter()).enumerate() {
        for (j, (got, exp)) in got_row.iter().zip(exp_row.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "strided silu mismatch at [{},{}]: got {}, expected {}",
                i,
                j,
                got,
                exp
            );
        }
    }
    println!("Strided SiLU test passed!");
    Ok(())
}

// ============ Broadcasting tests ============

#[test]
fn test_vulkan_broadcast_add_scalar() -> Result<()> {
    let device = get_vulkan_device()?;

    // Add a scalar (1D tensor with 1 element) to a larger tensor
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![10.0];

    let a = Tensor::from_vec(a_data.clone(), (4,), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data.clone(), (1,), &Device::Cpu)?.to_device(&device)?;

    let result = a.broadcast_add(&b)?;
    let result_vec = result.to_vec1::<f32>()?;

    let expected: Vec<f32> = a_data.iter().map(|x| x + 10.0).collect();
    assert_eq!(result_vec, expected);
    println!("Broadcast add (scalar) test passed!");
    Ok(())
}

#[test]
fn test_vulkan_broadcast_add_row() -> Result<()> {
    let device = get_vulkan_device()?;

    // Add a row vector to a matrix (broadcasting along rows)
    let a = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &Device::Cpu,
    )?
    .to_device(&device)?;

    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (1, 3), &Device::Cpu)?.to_device(&device)?;

    let result = a.broadcast_add(&b)?;
    let result_vec = result.to_vec2::<f32>()?;

    let expected = vec![vec![11.0, 22.0, 33.0], vec![14.0, 25.0, 36.0]];
    assert_eq!(result_vec, expected);
    println!("Broadcast add (row) test passed!");
    Ok(())
}

#[test]
fn test_vulkan_broadcast_mul_column() -> Result<()> {
    let device = get_vulkan_device()?;

    // Multiply a column vector with a matrix (broadcasting along columns)
    let a = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &Device::Cpu,
    )?
    .to_device(&device)?;

    let b = Tensor::from_vec(vec![10.0f32, 100.0], (2, 1), &Device::Cpu)?.to_device(&device)?;

    let result = a.broadcast_mul(&b)?;
    let result_vec = result.to_vec2::<f32>()?;

    let expected = vec![vec![10.0, 20.0, 30.0], vec![400.0, 500.0, 600.0]];
    assert_eq!(result_vec, expected);
    println!("Broadcast mul (column) test passed!");
    Ok(())
}

#[test]
fn test_vulkan_broadcast_3d() -> Result<()> {
    let device = get_vulkan_device()?;

    // 3D broadcasting: (2, 3, 4) + (1, 3, 1) -> (2, 3, 4)
    let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_data: Vec<f32> = vec![100.0f32, 200.0, 300.0];

    let a = Tensor::from_vec(a_data, (2, 3, 4), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data, (1, 3, 1), &Device::Cpu)?.to_device(&device)?;

    let result = a.broadcast_add(&b)?;
    let result_vec = result.to_vec3::<f32>()?;

    // Verify a few values
    // a[0,0,0] = 0, b broadcasted = 100 -> 100
    assert_eq!(result_vec[0][0][0], 100.0);
    // a[0,1,0] = 4, b broadcasted = 200 -> 204
    assert_eq!(result_vec[0][1][0], 204.0);
    // a[0,2,0] = 8, b broadcasted = 300 -> 308
    assert_eq!(result_vec[0][2][0], 308.0);
    // a[1,0,0] = 12, b broadcasted = 100 -> 112
    assert_eq!(result_vec[1][0][0], 112.0);

    println!("Broadcast 3D test passed!");
    Ok(())
}

#[test]
fn test_vulkan_strided_binary() -> Result<()> {
    let device = get_vulkan_device()?;

    // Test binary op with both operands strided (transposed)
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let a = Tensor::from_vec(a_data.clone(), (2, 3), &Device::Cpu)?.to_device(&device)?;
    let b = Tensor::from_vec(b_data.clone(), (2, 3), &Device::Cpu)?.to_device(&device)?;

    // Transpose both
    let a_t = a.t()?;
    let b_t = b.t()?;

    let result = (&a_t + &b_t)?;
    let result_vec = result.to_vec2::<f32>()?;

    // CPU reference
    let cpu_a = Tensor::from_vec(a_data, (2, 3), &Device::Cpu)?;
    let cpu_b = Tensor::from_vec(b_data, (2, 3), &Device::Cpu)?;
    let cpu_result = (cpu_a.t()? + cpu_b.t()?)?;
    let expected = cpu_result.to_vec2::<f32>()?;

    assert_eq!(result_vec, expected);
    println!("Strided binary test passed!");
    Ok(())
}

// ============ Quantization tests ============

use candle_core::quantized::{GgmlDType, QTensor};

#[test]
fn test_vulkan_quantize_dequantize_q4_0() -> Result<()> {
    let device = get_vulkan_device()?;

    // Create a tensor with values suitable for quantization
    // Q4_0 has block_size of 32, so use multiples of 32
    let n = 128;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let cpu_tensor = Tensor::from_vec(data.clone(), (n,), &Device::Cpu)?;

    // Quantize on CPU first, then test dequantization on Vulkan
    let qtensor = QTensor::quantize(&cpu_tensor, GgmlDType::Q4_0)?;

    // Get the expected dequantized result from CPU
    let cpu_dequant = qtensor.dequantize(&Device::Cpu)?;
    let expected = cpu_dequant.to_vec1::<f32>()?;

    // Now dequantize to Vulkan device
    let vulkan_dequant = qtensor.dequantize(&device)?;
    let result = vulkan_dequant.to_vec1::<f32>()?;

    // Compare results - they should be identical since both use CPU dequant for now
    for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "Q4_0 dequant mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
    println!("Q4_0 quantize/dequantize test passed!");
    Ok(())
}

#[test]
fn test_vulkan_quantize_dequantize_q8_0() -> Result<()> {
    let device = get_vulkan_device()?;

    // Q8_0 also has block_size of 32
    let n = 128;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.05).collect();
    let cpu_tensor = Tensor::from_vec(data.clone(), (n,), &Device::Cpu)?;

    // Quantize
    let qtensor = QTensor::quantize(&cpu_tensor, GgmlDType::Q8_0)?;

    // Get expected from CPU
    let cpu_dequant = qtensor.dequantize(&Device::Cpu)?;
    let expected = cpu_dequant.to_vec1::<f32>()?;

    // Dequantize to Vulkan
    let vulkan_dequant = qtensor.dequantize(&device)?;
    let result = vulkan_dequant.to_vec1::<f32>()?;

    for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "Q8_0 dequant mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
    println!("Q8_0 quantize/dequantize test passed!");
    Ok(())
}

#[test]
fn test_vulkan_qtensor_dequantize_to_cpu_matmul() -> Result<()> {
    let device = get_vulkan_device()?;

    // Create weight matrix (will be quantized) - n x k (transposed storage)
    // Using dimensions that are multiples of 32 for Q4_0
    let n = 64; // output features
    let k = 128; // input features
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
        .collect();
    let weight_tensor = Tensor::from_vec(weight_data.clone(), (n, k), &Device::Cpu)?;

    // Quantize weights on CPU
    let qweight = QTensor::quantize(&weight_tensor, GgmlDType::Q4_0)?;

    // Dequantize to Vulkan device, then transfer back to CPU for matmul
    // (Since Vulkan matmul isn't implemented yet)
    let dequant_weight_vulkan = qweight.dequantize(&device)?;
    let dequant_weight = dequant_weight_vulkan.to_device(&Device::Cpu)?;

    // Create input on CPU
    let batch = 4;
    let input_data: Vec<f32> = (0..(batch * k)).map(|i| (i % 50) as f32 * 0.02).collect();
    let input = Tensor::from_vec(input_data.clone(), (batch, k), &Device::Cpu)?;

    // Compute matmul on CPU with Vulkan-dequantized weights
    let result = input.matmul(&dequant_weight.t()?)?;
    let result_vec = result.to_vec2::<f32>()?;

    // Compute reference on CPU with CPU-dequantized weights
    let cpu_dequant = qweight.dequantize(&Device::Cpu)?;
    let cpu_result = input.matmul(&cpu_dequant.t()?)?;
    let expected = cpu_result.to_vec2::<f32>()?;

    // Compare - should be identical since both use same dequantization
    let mut max_diff = 0.0f32;
    for (i, (got_row, exp_row)) in result_vec.iter().zip(expected.iter()).enumerate() {
        for (j, (got, exp)) in got_row.iter().zip(exp_row.iter()).enumerate() {
            let diff = (got - exp).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-5,
                "Dequant matmul mismatch at [{},{}]: got {}, expected {}, diff {}",
                i,
                j,
                got,
                exp,
                diff
            );
        }
    }
    println!(
        "QTensor Vulkan dequantize + CPU matmul test passed! Max diff: {:.6}",
        max_diff
    );
    Ok(())
}

/// Stress test for descriptor pool management
/// This would previously fail with ERROR_OUT_OF_POOL_MEMORY after ~64 operations
#[test]
fn test_vulkan_descriptor_pool_stress() -> Result<()> {
    let device = get_vulkan_device()?;

    let a = Tensor::randn(0f32, 1.0, (64, 64), &device)?;
    let b = Tensor::randn(0f32, 1.0, (64, 64), &device)?;

    // Run many operations without explicit sync - tests pool growth
    for i in 0..100 {
        let _ = (&a + &b)?;
        let _ = (&a * &b)?;
        let _ = a.exp()?;
        let _ = a.relu()?;
        if i % 20 == 0 {
            // Periodic sync resets the descriptor set index
            device.synchronize()?;
        }
    }

    // Final sync
    device.synchronize()?;

    println!("Descriptor pool stress test passed! (400 ops without exhaustion)");
    Ok(())
}

/// Test Q4K quantize, dequantize roundtrip
#[test]
fn test_vulkan_q4k_quantize_dequantize() -> Result<()> {
    let device = get_vulkan_device()?;

    // Q4K has block size of 256, so use dimensions divisible by 256
    let nrows = 4;
    let ncols = 256;

    // Create random f32 data on CPU
    let cpu_tensor = Tensor::randn(0f32, 1.0, (nrows, ncols), &Device::Cpu)?;

    // Quantize to Q4K
    let qtensor = QTensor::quantize(&cpu_tensor, GgmlDType::Q4K)?;

    // Dequantize on CPU (reference)
    let cpu_dequant = qtensor.dequantize(&Device::Cpu)?;

    // Dequantize on Vulkan
    let vulkan_dequant = qtensor.dequantize(&device)?;
    let vulkan_result = vulkan_dequant.to_device(&Device::Cpu)?;

    // Compare
    let cpu_vals = cpu_dequant.flatten_all()?.to_vec1::<f32>()?;
    let vulkan_vals = vulkan_result.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff: f32 = 0.0;
    for (i, (&cpu_val, &vulkan_val)) in cpu_vals.iter().zip(vulkan_vals.iter()).enumerate() {
        let diff = (cpu_val - vulkan_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-4,
            "Q4K dequantize mismatch at {}: cpu={}, vulkan={}, diff={}",
            i, cpu_val, vulkan_val, diff
        );
    }

    println!("Q4K quantize/dequantize test passed! Max diff: {:.6}", max_diff);
    Ok(())
}

/// Test Q4K fused matmul kernel against CPU reference
#[test]
fn test_vulkan_q4k_matmul() -> Result<()> {
    use candle_core::quantized::QMatMul;

    let device = get_vulkan_device()?;

    // Q4K has block size of 256, so k must be divisible by 256
    let n = 64;   // output features (rows in weight matrix)
    let k = 256;  // input features (must be multiple of 256 for Q4K)
    let batch = 4;

    // Create random weight matrix on CPU
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
        .collect();
    let cpu_weight_tensor = Tensor::from_vec(weight_data.clone(), (n, k), &Device::Cpu)?;

    // Create input data
    let input_data: Vec<f32> = (0..(batch * k)).map(|i| (i % 50) as f32 * 0.02).collect();
    let cpu_input = Tensor::from_vec(input_data.clone(), (batch, k), &Device::Cpu)?;

    // Quantize weights to Q4K on CPU and run CPU reference
    let cpu_qweight = QTensor::quantize(&cpu_weight_tensor, GgmlDType::Q4K)?;
    let cpu_qmatmul = QMatMul::from_qtensor(cpu_qweight)?;
    let cpu_result = cpu_qmatmul.forward(&cpu_input)?;
    let expected = cpu_result.to_vec2::<f32>()?;

    // Now create weight tensor on Vulkan and quantize there
    let vulkan_weight_tensor = Tensor::from_vec(weight_data, (n, k), &device)?;
    let vulkan_qweight = QTensor::quantize(&vulkan_weight_tensor, GgmlDType::Q4K)?;
    let vulkan_qmatmul = QMatMul::from_qtensor(vulkan_qweight)?;

    // Create input on Vulkan
    let vulkan_input = Tensor::from_vec(input_data, (batch, k), &device)?;

    let vulkan_result = vulkan_qmatmul.forward(&vulkan_input)?;
    let vulkan_result_cpu = vulkan_result.to_device(&Device::Cpu)?;
    let result_vec = vulkan_result_cpu.to_vec2::<f32>()?;

    // Compare results - allow some tolerance for quantization differences
    let mut max_diff: f32 = 0.0;
    let mut total_error: f32 = 0.0;
    for (i, (got_row, exp_row)) in result_vec.iter().zip(expected.iter()).enumerate() {
        for (j, (got, exp)) in got_row.iter().zip(exp_row.iter()).enumerate() {
            let diff = (got - exp).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            total_error += diff;
            // Allow reasonable tolerance for GPU vs CPU quantized matmul
            // Similar to CUDA/Metal tolerances in quantized_tests.rs
            assert!(
                diff < 1.0,
                "Q4K matmul mismatch at [{},{}]: got {}, expected {}, diff {}",
                i, j, got, exp, diff
            );
        }
    }

    let avg_error = total_error / (batch * n) as f32;
    println!(
        "Q4K Vulkan matmul test passed! Max diff: {:.4}, Avg diff: {:.4}",
        max_diff, avg_error
    );

    // Also verify the shape is correct
    assert_eq!(vulkan_result.dims(), &[batch, n]);

    Ok(())
}

/// Test Q4K matmul with larger dimensions (stress test)
#[test]
fn test_vulkan_q4k_matmul_large() -> Result<()> {
    use candle_core::quantized::QMatMul;

    let device = get_vulkan_device()?;

    // Larger dimensions similar to typical LLM layers
    let n = 512;   // output features
    let k = 512;   // input features (multiple of 256)
    let batch = 8;

    // Create random weight matrix on CPU first for reference
    let cpu_weight_tensor = Tensor::randn(0f32, 0.1, (n, k), &Device::Cpu)?;

    // Quantize weights to Q4K on CPU and create CPU reference
    let cpu_qweight = QTensor::quantize(&cpu_weight_tensor, GgmlDType::Q4K)?;
    let cpu_qmatmul = QMatMul::from_qtensor(cpu_qweight)?;

    // Create weight tensor on Vulkan and quantize there
    let vulkan_weight_tensor = cpu_weight_tensor.to_device(&device)?;
    let vulkan_qweight = QTensor::quantize(&vulkan_weight_tensor, GgmlDType::Q4K)?;
    let vulkan_qmatmul = QMatMul::from_qtensor(vulkan_qweight)?;

    // Create random input on Vulkan
    let vulkan_input = Tensor::randn(0f32, 1.0, (batch, k), &device)?;
    let cpu_input = vulkan_input.to_device(&Device::Cpu)?;

    // Run on Vulkan
    let vulkan_result = vulkan_qmatmul.forward(&vulkan_input)?;
    let vulkan_result_cpu = vulkan_result.to_device(&Device::Cpu)?;

    // Run on CPU for reference
    let cpu_result = cpu_qmatmul.forward(&cpu_input)?;

    // Compare
    let vulkan_vals = vulkan_result_cpu.flatten_all()?.to_vec1::<f32>()?;
    let cpu_vals = cpu_result.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff: f32 = 0.0;
    for (&v, &c) in vulkan_vals.iter().zip(cpu_vals.iter()) {
        let diff = (v - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Compute relative error
    let cpu_abs_sum: f32 = cpu_vals.iter().map(|x| x.abs()).sum();
    let diff_sum: f32 = vulkan_vals.iter().zip(cpu_vals.iter()).map(|(v, c)| (v - c).abs()).sum();
    let rel_error = diff_sum / cpu_abs_sum;

    println!(
        "Q4K large matmul test: max_diff={:.4}, rel_error={:.6}",
        max_diff, rel_error
    );

    // Allow 2% relative error (similar to quantized_tests.rs tolerance)
    assert!(
        rel_error < 0.02,
        "Relative error {:.4} exceeds threshold",
        rel_error
    );

    Ok(())
}
