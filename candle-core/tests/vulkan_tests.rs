//! Vulkan backend integration tests
//!
//! These tests verify that Vulkan compute kernels work correctly.

#![cfg(feature = "vulkan")]

use candle_core::{DType, Device, Result, Tensor};

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
