//! Simple test to verify Vulkan backend integration works
use candle_core::{Device, Tensor, DType};

fn main() -> candle_core::Result<()> {
    println!("Testing Vulkan backend integration...");
    
    // Check if Vulkan is available
    if !candle_core::utils::vulkan_is_available() {
        println!("Vulkan is not available on this system");
        return Ok(());
    }
    
    println!("Vulkan is available!");
    
    // Create a Vulkan device
    let device = Device::new_vulkan(0)?;
    println!("Created Vulkan device: {:?}", device);
    
    // Create a simple tensor on CPU and move it to Vulkan
    let cpu_tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
    println!("CPU tensor: {:?}", cpu_tensor.to_vec1::<f32>()?);
    
    // Move to Vulkan device
    let vulkan_tensor = cpu_tensor.to_device(&device)?;
    println!("Moved tensor to Vulkan device");
    
    // Read back from Vulkan
    let result = vulkan_tensor.to_vec1::<f32>()?;
    println!("Read back from Vulkan: {:?}", result);
    
    // Verify values
    assert_eq!(result, vec![1.0f32, 2.0, 3.0, 4.0]);
    println!("Values verified correctly!");
    
    // Test creating a tensor directly on Vulkan
    let vulkan_zeros = Tensor::zeros((2, 3), DType::F32, &device)?;
    println!("Created zeros tensor on Vulkan: shape {:?}", vulkan_zeros.shape());
    let zeros_result = vulkan_zeros.to_vec2::<f32>()?;
    println!("Zeros content: {:?}", zeros_result);
    
    println!("\nVulkan integration test passed!");
    Ok(())
}
