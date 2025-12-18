//! Basic tests for the HIP backend

use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "hip")]
fn get_hip_device() -> Result<Device> {
    Device::new_hip(0)
}

#[cfg(feature = "hip")]
#[test]
fn hip_device_creation() -> Result<()> {
    let device = get_hip_device()?;
    assert!(device.is_hip());
    println!("HIP device created successfully");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_zeros() -> Result<()> {
    let device = get_hip_device()?;
    let tensor = Tensor::zeros((5, 3), DType::F32, &device)?;
    let (dim1, dim2) = tensor.dims2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 3);

    // Convert back to CPU and check values
    let cpu_tensor = tensor.to_device(&Device::Cpu)?;
    let values = cpu_tensor.to_vec2::<f32>()?;
    for row in &values {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    println!("HIP zeros test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_ones() -> Result<()> {
    let device = get_hip_device()?;
    let tensor = Tensor::ones((4, 2), DType::F32, &device)?;

    // Convert back to CPU and check values
    let cpu_tensor = tensor.to_device(&Device::Cpu)?;
    let values = cpu_tensor.to_vec2::<f32>()?;
    for row in &values {
        for &val in row {
            assert_eq!(val, 1.0);
        }
    }
    println!("HIP ones test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_from_vec() -> Result<()> {
    let device = get_hip_device()?;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data.clone(), (2, 3), &device)?;

    // Convert back to CPU and check values
    let cpu_tensor = tensor.to_device(&Device::Cpu)?;
    let values = cpu_tensor.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(values, data);
    println!("HIP from_vec test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_to_cpu_roundtrip() -> Result<()> {
    let device = get_hip_device()?;
    let data: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];

    // Create on CPU, move to HIP, move back to CPU
    let cpu_tensor = Tensor::from_vec(data.clone(), (2, 2), &Device::Cpu)?;
    let hip_tensor = cpu_tensor.to_device(&device)?;
    let back_to_cpu = hip_tensor.to_device(&Device::Cpu)?;

    let result = back_to_cpu.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, data);
    println!("HIP roundtrip test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_add() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;

    let c = (&a + &b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    println!("HIP add test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_mul() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], (2, 2), &device)?;

    let c = (&a * &b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    println!("HIP mul test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_sub() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;

    let c = (&a - &b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
    println!("HIP sub test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_div() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![2.0f32, 4.0, 5.0, 8.0], (2, 2), &device)?;

    let c = (&a / &b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    println!("HIP div test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_affine() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;

    // affine: mul by 2, add 1
    let c = a.affine(2.0, 1.0)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);
    println!("HIP affine test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_dtypes() -> Result<()> {
    let device = get_hip_device()?;

    // Test F32
    let f32_tensor = Tensor::ones((2, 2), DType::F32, &device)?;
    assert_eq!(f32_tensor.dtype(), DType::F32);

    // Test F64
    let f64_tensor = Tensor::ones((2, 2), DType::F64, &device)?;
    assert_eq!(f64_tensor.dtype(), DType::F64);

    // Test F16
    let f16_tensor = Tensor::ones((2, 2), DType::F16, &device)?;
    assert_eq!(f16_tensor.dtype(), DType::F16);

    // Test BF16
    let bf16_tensor = Tensor::ones((2, 2), DType::BF16, &device)?;
    assert_eq!(bf16_tensor.dtype(), DType::BF16);

    // Test U32
    let u32_tensor = Tensor::ones((2, 2), DType::U32, &device)?;
    assert_eq!(u32_tensor.dtype(), DType::U32);

    // Test I64
    let i64_tensor = Tensor::ones((2, 2), DType::I64, &device)?;
    assert_eq!(i64_tensor.dtype(), DType::I64);

    println!("HIP dtypes test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_exp() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, -1.0], (2, 2), &device)?;
    let c = a.exp()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // Check exp values with tolerance
    let expected = [1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E, 1.0 / std::f32::consts::E];
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "exp mismatch: {} vs {}", got, exp);
    }
    println!("HIP exp test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_log() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, std::f32::consts::E, 10.0, 100.0], (2, 2), &device)?;
    let c = a.log()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    let expected = [0.0f32, 1.0, 10.0f32.ln(), 100.0f32.ln()];
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "log mismatch: {} vs {}", got, exp);
    }
    println!("HIP log test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_sin_cos() -> Result<()> {
    let device = get_hip_device()?;
    let pi = std::f32::consts::PI;
    let a = Tensor::from_vec(vec![0.0f32, pi / 2.0, pi, 3.0 * pi / 2.0], (2, 2), &device)?;

    let sin_result = a.sin()?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let cos_result = a.cos()?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // sin(0) = 0, sin(pi/2) = 1, sin(pi) = 0, sin(3pi/2) = -1
    let sin_expected = [0.0f32, 1.0, 0.0, -1.0];
    for (got, exp) in sin_result.iter().zip(sin_expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "sin mismatch: {} vs {}", got, exp);
    }

    // cos(0) = 1, cos(pi/2) = 0, cos(pi) = -1, cos(3pi/2) = 0
    let cos_expected = [1.0f32, 0.0, -1.0, 0.0];
    for (got, exp) in cos_result.iter().zip(cos_expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "cos mismatch: {} vs {}", got, exp);
    }

    println!("HIP sin/cos test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_sqrt() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 4.0, 9.0, 16.0], (2, 2), &device)?;
    let c = a.sqrt()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    let expected = [1.0f32, 2.0, 3.0, 4.0];
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "sqrt mismatch: {} vs {}", got, exp);
    }
    println!("HIP sqrt test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_neg() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], (2, 2), &device)?;
    let c = a.neg()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
    println!("HIP neg test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_unary_relu() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0], (2, 3), &device)?;
    let c = a.relu()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    println!("HIP relu test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_comparison() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], (2, 2), &device)?;

    // a < b: [1<2, 2<2, 3<2, 4<2] = [1, 0, 0, 0]
    let lt = a.lt(&b)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    assert_eq!(lt, vec![1, 0, 0, 0]);

    // a == b: [1==2, 2==2, 3==2, 4==2] = [0, 1, 0, 0]
    let eq = a.eq(&b)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    assert_eq!(eq, vec![0, 1, 0, 0]);

    // a > b: [1>2, 2>2, 3>2, 4>2] = [0, 0, 1, 1]
    let gt = a.gt(&b)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u8>()?;
    assert_eq!(gt, vec![0, 0, 1, 1]);

    println!("HIP comparison test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_to_dtype() -> Result<()> {
    let device = get_hip_device()?;

    // F32 -> F64
    let f32_tensor = Tensor::from_vec(vec![1.5f32, 2.5, 3.5, 4.5], (2, 2), &device)?;
    let f64_tensor = f32_tensor.to_dtype(DType::F64)?;
    assert_eq!(f64_tensor.dtype(), DType::F64);
    let result = f64_tensor.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f64>()?;
    assert_eq!(result, vec![1.5f64, 2.5, 3.5, 4.5]);

    // F32 -> U32
    let f32_tensor2 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let u32_tensor = f32_tensor2.to_dtype(DType::U32)?;
    assert_eq!(u32_tensor.dtype(), DType::U32);
    let result = u32_tensor.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
    assert_eq!(result, vec![1u32, 2, 3, 4]);

    println!("HIP to_dtype test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_clone() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let b = a.clone();

    let a_cpu = a.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let b_cpu = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a_cpu, b_cpu);

    println!("HIP clone test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_where_cond() -> Result<()> {
    let device = get_hip_device()?;
    let cond = Tensor::from_vec(vec![1u8, 0, 1, 0], (2, 2), &device)?;
    let t = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
    let f = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;

    let result = cond.where_cond(&t, &f)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // cond[0]=1 -> t[0]=10, cond[1]=0 -> f[1]=2, cond[2]=1 -> t[2]=30, cond[3]=0 -> f[3]=4
    assert_eq!(result, vec![10.0, 2.0, 30.0, 4.0]);

    println!("HIP where_cond test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_powf() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    let c = a.powf(2.0)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![1.0, 4.0, 9.0, 16.0]);
    println!("HIP powf test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_tanh() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0], (2, 2), &device)?;
    let c = a.tanh()?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    let expected: Vec<f32> = vec![0.0f32, 1.0, -1.0, 2.0].iter().map(|x| x.tanh()).collect();
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "tanh mismatch: {} vs {}", got, exp);
    }
    println!("HIP tanh test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_reduce_sum() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;

    // Sum all elements
    let sum_all = a.sum_all()?.to_device(&Device::Cpu)?.to_scalar::<f32>()?;
    assert!((sum_all - 21.0).abs() < 1e-5, "sum_all mismatch: {}", sum_all);

    println!("HIP reduce_sum test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_reduce_max() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 2.0, 6.0, 4.0], (2, 3), &device)?;

    // Max of all elements
    let max_all = a.max_all()?.to_device(&Device::Cpu)?.to_scalar::<f32>()?;
    assert!((max_all - 6.0).abs() < 1e-5, "max_all mismatch: {}", max_all);

    println!("HIP reduce_max test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_reduce_min() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![3.0f32, 1.0, 5.0, 2.0, 6.0, 4.0], (2, 3), &device)?;

    // Min of all elements
    let min_all = a.min_all()?.to_device(&Device::Cpu)?.to_scalar::<f32>()?;
    assert!((min_all - 1.0).abs() < 1e-5, "min_all mismatch: {}", min_all);

    println!("HIP reduce_min test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_argmax() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 2.0, 6.0, 4.0], (6,), &device)?;

    // Argmax - index of max element (6.0 at index 4)
    let argmax_result = a.argmax(0)?.to_device(&Device::Cpu)?.to_scalar::<u32>()?;
    assert_eq!(argmax_result, 4, "argmax mismatch");

    println!("HIP argmax test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_argmin() -> Result<()> {
    let device = get_hip_device()?;
    let a = Tensor::from_vec(vec![3.0f32, 1.0, 5.0, 2.0, 6.0, 4.0], (6,), &device)?;

    // Argmin - index of min element (1.0 at index 1)
    let argmin_result = a.argmin(0)?.to_device(&Device::Cpu)?.to_scalar::<u32>()?;
    assert_eq!(argmin_result, 1, "argmin mismatch");

    println!("HIP argmin test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_index_select() -> Result<()> {
    let device = get_hip_device()?;
    // 2D tensor: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3), &device)?;
    // Select rows 0 and 2
    let indices = Tensor::from_vec(vec![0u32, 2], (2,), &device)?;

    let result = a.index_select(&indices, 0)?;
    let result_cpu = result.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // Should get rows 0 and 2: [[1, 2, 3], [7, 8, 9]]
    assert_eq!(result_cpu, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);

    println!("HIP index_select test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_index_select_dim1() -> Result<()> {
    let device = get_hip_device()?;
    // 2D tensor: [[1, 2, 3], [4, 5, 6]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;
    // Select columns 0 and 2
    let indices = Tensor::from_vec(vec![0u32, 2], (2,), &device)?;

    let result = a.index_select(&indices, 1)?;
    let result_cpu = result.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // Should get columns 0 and 2: [[1, 3], [4, 6]]
    assert_eq!(result_cpu, vec![1.0, 3.0, 4.0, 6.0]);

    println!("HIP index_select dim1 test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_gather() -> Result<()> {
    let device = get_hip_device()?;
    // 2D tensor: [[1, 2], [3, 4]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
    // Gather indices for dim 1: [[0, 1], [1, 0]]
    let indices = Tensor::from_vec(vec![0u32, 1, 1, 0], (2, 2), &device)?;

    let result = a.gather(&indices, 1)?;
    let result_cpu = result.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // Row 0: [a[0,0], a[0,1]] = [1, 2]
    // Row 1: [a[1,1], a[1,0]] = [4, 3]
    assert_eq!(result_cpu, vec![1.0, 2.0, 4.0, 3.0]);

    println!("HIP gather test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_matmul() -> Result<()> {
    let device = get_hip_device()?;
    // Simple 2x2 matmul: A(2x3) @ B(3x2) = C(2x2)
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // C = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //   = [[22, 28], [49, 64]]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;
    let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &device)?;

    let c = a.matmul(&b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);

    println!("HIP matmul test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_matmul_batch() -> Result<()> {
    let device = get_hip_device()?;
    // Batched matmul: batch=2, A(2x2) @ B(2x2) = C(2x2)
    // Batch 0: [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
    // Batch 1: [[1, 0], [0, 1]] @ [[5, 6], [7, 8]] = [[5, 6], [7, 8]]
    let a = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0],
        (2, 2, 2),
        &device,
    )?;
    let b = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0],
        (2, 2, 2),
        &device,
    )?;

    let c = a.matmul(&b)?;
    let result = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    println!("HIP batched matmul test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_conv1d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, length=5) = [1, 2, 3, 4, 5]
    // Kernel: (out_channels=1, in_channels=1, kernel_size=3) = [1, 0, 1]
    // Conv1d with stride=1, padding=0: output length = 5 - 3 + 1 = 3
    // output[0] = 1*1 + 2*0 + 3*1 = 4
    // output[1] = 2*1 + 3*0 + 4*1 = 6
    // output[2] = 3*1 + 4*0 + 5*1 = 8
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &device)?;
    let kernel = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], (1, 1, 3), &device)?;

    let output = input.conv1d(&kernel, 0, 1, 1, 1)?;
    let result = output.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(result, vec![4.0, 6.0, 8.0]);

    println!("HIP conv1d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_conv2d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, height=3, width=3)
    // Kernel: (out_channels=1, in_channels=1, kh=2, kw=2) = [[1, 0], [0, 1]]
    // Conv2d with stride=1, padding=0: output size = 3 - 2 + 1 = 2
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        (1, 1, 3, 3),
        &device,
    )?;
    let kernel = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, 2), &device)?;

    let output = input.conv2d(&kernel, 0, 1, 1, 1)?;
    let result = output.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    // output[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
    // output[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8
    // output[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12
    // output[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14
    assert_eq!(result, vec![6.0, 8.0, 12.0, 14.0]);

    println!("HIP conv2d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_avg_pool2d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, height=4, width=4)
    // Pool: kernel_size=2x2, stride=2x2
    // Output: (1, 1, 2, 2)
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        (1, 1, 4, 4),
        &device,
    )?;

    let output = input.avg_pool2d((2, 2))?.to_device(&Device::Cpu)?;
    let result = output.flatten_all()?.to_vec1::<f32>()?;
    // Pool 0,0: (1+2+5+6)/4 = 3.5
    // Pool 0,1: (3+4+7+8)/4 = 5.5
    // Pool 1,0: (9+10+13+14)/4 = 11.5
    // Pool 1,1: (11+12+15+16)/4 = 13.5
    let expected = vec![3.5, 5.5, 11.5, 13.5];
    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "avg_pool2d mismatch: {} vs {}", got, exp);
    }

    println!("HIP avg_pool2d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_max_pool2d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, height=4, width=4)
    // Pool: kernel_size=2x2, stride=2x2
    // Output: (1, 1, 2, 2)
    let input = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ],
        (1, 1, 4, 4),
        &device,
    )?;

    let output = input.max_pool2d((2, 2))?.to_device(&Device::Cpu)?;
    let result = output.flatten_all()?.to_vec1::<f32>()?;
    // Max of each 2x2 block
    // Pool 0,0: max(1,2,5,6) = 6
    // Pool 0,1: max(3,4,7,8) = 8
    // Pool 1,0: max(9,10,13,14) = 14
    // Pool 1,1: max(11,12,15,16) = 16
    assert_eq!(result, vec![6.0, 8.0, 14.0, 16.0]);

    println!("HIP max_pool2d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_upsample_nearest2d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, height=2, width=2)
    // Output: (batch=1, channels=1, height=4, width=4)
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0],
        (1, 1, 2, 2),
        &device,
    )?;

    let output = input.upsample_nearest2d(4, 4)?.to_device(&Device::Cpu)?;
    let result = output.flatten_all()?.to_vec1::<f32>()?;
    // Each input pixel gets replicated in a 2x2 block
    // [[1, 2], [3, 4]] -> [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]
    let expected = vec![
        1.0, 1.0, 2.0, 2.0,
        1.0, 1.0, 2.0, 2.0,
        3.0, 3.0, 4.0, 4.0,
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(result, expected);

    println!("HIP upsample_nearest2d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_index_add() -> Result<()> {
    let device = get_hip_device()?;
    // Test index_add: dst[indices[i]] += src[i]
    // dst shape: (4, 3), src shape: (2, 3)
    // indices: [1, 3] means add src[0] to dst[1] and src[1] to dst[3]
    let dst = Tensor::zeros((4, 3), DType::F32, &device)?;
    let src = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &device,
    )?;
    let indices = Tensor::from_vec(vec![1u32, 3], (2,), &device)?;

    let result = dst.index_add(&indices, &src, 0)?.to_device(&Device::Cpu)?;
    let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
    // dst[0] = [0, 0, 0], dst[1] = [1, 2, 3], dst[2] = [0, 0, 0], dst[3] = [4, 5, 6]
    let expected = vec![
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
        0.0, 0.0, 0.0,
        4.0, 5.0, 6.0,
    ];
    assert_eq!(result_vec, expected);

    println!("HIP index_add test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_scatter_add() -> Result<()> {
    let device = get_hip_device()?;
    // Test scatter_add: dst[src_indices[i]] += src[i]
    // dst shape: (4,), src shape: (3,)
    // src_indices: [0, 1, 0] means add src[0] and src[2] to dst[0], src[1] to dst[1]
    let dst = Tensor::zeros((4,), DType::F32, &device)?;
    let src = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)?;
    let indices = Tensor::from_vec(vec![0u32, 1, 0], (3,), &device)?;

    let result = dst.scatter_add(&indices, &src, 0)?.to_device(&Device::Cpu)?;
    let result_vec = result.flatten_all()?.to_vec1::<f32>()?;
    // dst[0] = 1.0 + 3.0 = 4.0, dst[1] = 2.0, dst[2] = 0, dst[3] = 0
    assert_eq!(result_vec, vec![4.0, 2.0, 0.0, 0.0]);

    println!("HIP scatter_add test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_conv_transpose1d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=2, length=3)
    // Kernel: (in_channels=2, out_channels=1, kernel_size=2)
    // ConvTranspose with stride=2, padding=0: output length = (3-1)*2 + 2 = 6
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],  // 2 channels, 3 elements each
        (1, 2, 3),
        &device,
    )?;
    // Kernel: [[1, 0], [0, 1]] shape (2, 1, 2)
    let kernel = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 1.0],
        (2, 1, 2),
        &device,
    )?;

    // conv_transpose1d(kernel, padding, output_padding, stride, dilation, groups)
    let output = input.conv_transpose1d(&kernel, 0, 0, 2, 1, 1)?;
    let result = output.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;

    // Transposed convolution should produce output of length 6
    assert_eq!(result.len(), 6);

    println!("HIP conv_transpose1d test passed");
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_conv_transpose2d() -> Result<()> {
    let device = get_hip_device()?;
    // Input: (batch=1, channels=1, height=2, width=2)
    // Kernel: (in_channels=1, out_channels=1, kh=2, kw=2)
    // ConvTranspose with stride=2, padding=0: output size = (2-1)*2 + 2 = 4
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0],
        (1, 1, 2, 2),
        &device,
    )?;
    // Kernel: identity-like [[1, 0], [0, 0]]
    let kernel = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0],
        (1, 1, 2, 2),
        &device,
    )?;

    // conv_transpose2d(kernel, padding, output_padding, stride, dilation)
    let output = input.conv_transpose2d(&kernel, 0, 0, 2, 1)?;
    let shape = output.shape().dims();

    // Transposed convolution should produce output of size 4x4
    assert_eq!(shape, &[1, 1, 4, 4]);

    println!("HIP conv_transpose2d test passed");
    Ok(())
}
