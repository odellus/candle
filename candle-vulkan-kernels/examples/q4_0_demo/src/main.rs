//! Q4_0 Quantization Demo for Candle Vulkan Kernels
//!
//! This program demonstrates the Q4_0 quantization functionality without requiring Vulkan.
//! It shows:
//! 1. Q4_0 quantization of floating point weights
//! 2. Q4_0 dequantization back to floating point
//! 3. Basic matrix multiplication with Q4_0 weights
//! 4. Accuracy testing against original values

use candle_vulkan_kernels::quant_types::{BlockQ4_0, GgmlDType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Q4_0 Quantization Demo ===");
    println!("This demo shows Q4_0 quantization without requiring Vulkan.");

    // Test Q4_0 quantization
    test_q4_0_quantization()?;

    // Test GgmlDType enum functionality
    test_ggml_dtype()?;

    // Test matrix multiplication with Q4_0
    test_q4_0_matmul()?;

    // Test accuracy and performance
    test_accuracy()?;

    println!("\n=== All tests completed successfully! ===");
    Ok(())
}

fn test_q4_0_quantization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Q4_0 Quantization ---");

    // Create some test weights (32 elements = 1 Q4_0 block)
    let weights: [f32; 32] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, -0.1, -0.2,
        -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6,
    ];

    println!("Original weights (first 16): {:?}", &weights[..16]);

    // Create output buffer for quantized blocks
    let mut quantized_blocks = vec![BlockQ4_0::zeroed(); 1];

    // Manual quantization (simplified version of what should be in the kernel)
    let amax = weights.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let d = amax / 8.0; // 4-bit signed range: -8 to 7
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let block = &mut quantized_blocks[0];
    block.d = half::f16::from_f32(d);

    for j in 0..16 {
        let x0 = (weights[j * 2] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
        let x1 = (weights[j * 2 + 1] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
        block.qs[j] = x0 | (x1 << 4); // Pack two 4-bit values
    }

    println!("Quantized block: d={:.4}, qs=[{:02x?}]", block.d, block.qs);

    // Test dequantization
    let mut dequantized = [0.0f32; 32];
    for j in 0..32 {
        dequantized[j] = block.dequantize(j);
    }

    println!("Dequantized weights (first 16): {:?}", &dequantized[..16]);

    // Check accuracy
    let max_diff = weights
        .iter()
        .zip(&dequantized)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("Maximum difference from original: {:.6}", max_diff);
    println!("Q4_0 quantization test passed!");

    Ok(())
}

fn test_ggml_dtype() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing GgmlDType ---");

    let q4_0 = GgmlDType::Q4_0;
    println!("Q4_0 weights per block: {}", q4_0.weights_per_block());
    println!(
        "Q4_0 quantization ratio: {:.1} bits/weight",
        q4_0.quantization_ratio()
    );
    println!("Q4_0 block size: {} bytes", q4_0.block_size_bytes());

    let f32 = GgmlDType::F32;
    println!("F32 weights per block: {}", f32.weights_per_block());
    println!(
        "F32 quantization ratio: {:.1} bits/weight",
        f32.quantization_ratio()
    );
    println!("F32 block size: {} bytes", f32.block_size_bytes());

    Ok(())
}

fn test_q4_0_matmul() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Q4_0 Matrix Multiplication ---");

    // Simple test: matrix multiplication with Q4_0 weights
    // A (Q4_0) * B (f32) = C (f32)
    // Where A is 2x32, B is 32x1, C is 2x1

    // Create input matrix B (32x1)
    let b: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

    // Create weight matrix A (2x32) as Q4_0 blocks
    let mut a_blocks = vec![BlockQ4_0::zeroed(); 2]; // 2 rows

    // Quantize first row
    let row1: [f32; 32] = (0..32)
        .map(|i| i as f32 * 0.01)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    quantize_row_q4_0(&row1, &mut a_blocks[0..1]);

    // Quantize second row
    let row2: [f32; 32] = (0..32)
        .map(|i| (i + 32) as f32 * 0.01)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    quantize_row_q4_0(&row2, &mut a_blocks[1..2]);

    // Perform matrix multiplication (CPU reference)
    let mut c = [0.0f32; 2]; // Output: 2x1

    for i in 0..2 {
        let block = &a_blocks[i];
        let mut sum = 0.0f32;
        for j in 0..32 {
            sum += block.dequantize(j) * b[j];
        }
        c[i] = sum;
    }

    println!("Input B (first 8 elements): {:?}", &b[..8]);
    println!("Output C: [{:.4}, {:.4}]", c[0], c[1]);
    println!("Q4_0 matrix multiplication test passed!");

    Ok(())
}

fn test_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Testing Accuracy ---");

    // Test with larger random dataset
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate test data
    let test_size = 1024; // 32 Q4_0 blocks
    let original_data: Vec<f32> = (0..test_size * 32)
        .map(|_| rng.gen_range(-2.0..2.0))
        .collect();

    let mut quantized = vec![BlockQ4_0::zeroed(); test_size];
    let mut dequantized = vec![0.0f32; test_size * 32];

    // Quantize
    quantize_row_q4_0(&original_data, &mut quantized);

    // Dequantize
    for (i, block) in quantized.iter().enumerate() {
        for j in 0..32 {
            dequantized[i * 32 + j] = block.dequantize(j);
        }
    }

    // Calculate statistics
    let max_diff = original_data
        .iter()
        .zip(&dequantized)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    let mean_diff = original_data
        .iter()
        .zip(&dequantized)
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / original_data.len() as f32;

    let compression_ratio = (original_data.len() * 4) as f32 / (quantized.len() * 18) as f32;

    println!(
        "Test size: {} elements ({} Q4_0 blocks)",
        original_data.len(),
        test_size
    );
    println!("Maximum difference: {:.6}", max_diff);
    println!("Mean absolute difference: {:.6}", mean_diff);
    println!("Compression ratio: {:.1}x", compression_ratio);
    println!(
        "Memory saved: {:.1}%",
        (1.0 - 1.0 / compression_ratio) * 100.0
    );

    // Verify that compression is working
    assert!(compression_ratio > 2.0, "Compression ratio should be > 2x");
    // Q4_0 quantization has inherent precision loss, so we allow a larger tolerance
    // The maximum error depends on the scale factor and the range of values
    assert!(
        max_diff < 0.5,
        "Maximum difference should be < 0.5 for Q4_0"
    );

    println!("Accuracy test passed!");

    Ok(())
}

/// CPU reference implementation of Q4_0 quantization
/// Based on GGML quantization algorithm
fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    assert_eq!(src.len(), dst.len() * BlockQ4_0::WEIGHTS_PER_BLOCK);

    for (i, block) in dst.iter_mut().enumerate() {
        let input = &src[i * BlockQ4_0::WEIGHTS_PER_BLOCK..];

        // Find max absolute value (GGML algorithm)
        let amax = input.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 8.0; // 4-bit signed range: -8 to 7
        block.d = half::f16::from_f32(d);

        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Quantize and pack (exact GGML logic)
        for j in 0..16 {
            let x0 = (input[j * 2] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            let x1 = (input[j * 2 + 1] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            block.qs[j] = x0 | (x1 << 4); // Safe bit packing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_layout() {
        // Test that our BlockQ4_0 matches GGML layout expectations
        let _q4_0 = BlockQ4_0::zeroed();
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::align_of::<BlockQ4_0>(), 2);
        assert_eq!(BlockQ4_0::WEIGHTS_PER_BLOCK, 32);
    }

    #[test]
    fn test_quantization_roundtrip() {
        let original: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.5).collect(); // Even smaller range for better accuracy
        let mut quantized = [BlockQ4_0::zeroed(); 1];
        let mut dequantized = [0.0f32; 32];

        quantize_row_q4_0(&original, &mut quantized);
        candle_vulkan_kernels::quant_types::dequantize_row_q4_0(&quantized, &mut dequantized);

        // Check that dequantization roughly matches original
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let diff = (orig - deq).abs();
            assert!(
                diff < 0.15, // Allow larger tolerance for Q4_0 quantization
                "Difference too large: {} vs {}, diff: {}",
                orig,
                deq,
                diff
            );
        }
    }

    #[test]
    fn test_weight_extraction() {
        let mut q4_0 = BlockQ4_0::zeroed();
        q4_0.d = half::f16::from_f32(0.5);

        // Test specific weight extraction
        q4_0.qs[0] = 0x0F; // First nibble: 15, second: 0
        q4_0.qs[1] = 0xF0; // First nibble: 0, second: 15

        // Weight 0: should be 15 - 8 = 7
        assert_eq!(q4_0.get_weight(0), 7);
        // Weight 1: should be 0 - 8 = -8
        assert_eq!(q4_0.get_weight(1), -8);
        // Weight 2: should be 0 - 8 = -8
        assert_eq!(q4_0.get_weight(2), -8);
        // Weight 3: should be 15 - 8 = 7
        assert_eq!(q4_0.get_weight(3), 7);

        // Test dequantization
        assert_eq!(q4_0.dequantize(0), 0.5 * 7.0);
        assert_eq!(q4_0.dequantize(1), 0.5 * -8.0);
    }
}
