//! Integration test for matrix-vector multiplication

use ash::vk;
use candle_vulkan_kernels::{ops::call_matvec, Kernels, VulkanContext};
use vk_mem::{Alloc, AllocationCreateFlags, AllocationCreateInfo};

fn create_buffer(
    allocator: &std::mem::ManuallyDrop<vk_mem::Allocator>,
    size: usize,
) -> (vk::Buffer, vk_mem::Allocation) {
    let buffer_info = vk::BufferCreateInfo {
        size: size as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let alloc_info = AllocationCreateInfo {
        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
        flags: AllocationCreateFlags::MAPPED,
        ..Default::default()
    };

    unsafe { allocator.create_buffer(&buffer_info, &alloc_info).unwrap() }
}

fn write_buffer<T: Copy>(allocator: &std::mem::ManuallyDrop<vk_mem::Allocator>, allocation: &mut vk_mem::Allocation, data: &[T]) {
    unsafe {
        let ptr = allocator.map_memory(allocation).unwrap() as *mut T;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        allocator.unmap_memory(allocation);
    }
}

fn read_buffer<T: Copy + Default>(
    allocator: &std::mem::ManuallyDrop<vk_mem::Allocator>,
    allocation: &mut vk_mem::Allocation,
    count: usize,
) -> Vec<T> {
    let mut result = vec![T::default(); count];
    unsafe {
        let ptr = allocator.map_memory(allocation).unwrap() as *const T;
        std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), count);
        allocator.unmap_memory(allocation);
    }
    result
}

#[test]
fn test_matvec() {
    // Create Vulkan context
    let ctx = VulkanContext::new(0).expect("Failed to create Vulkan context");
    println!("Using device: {}", ctx.device_name());

    println!("Creating kernels...");
    let kernels = Kernels::new(ctx.clone()).expect("Failed to create kernels");
    println!("Kernels created");

    // Test dimensions
    const NROWS: usize = 64;
    const NCOLS: usize = 512;

    // Create buffers
    let matrix_size = NROWS * NCOLS * std::mem::size_of::<f32>();
    let vector_size = NCOLS * std::mem::size_of::<f32>();
    let output_size = NROWS * std::mem::size_of::<f32>();

    println!("Creating buffers...");
    let (matrix_buf, mut matrix_alloc) = create_buffer(&ctx.allocator, matrix_size);
    println!("  matrix buffer created");
    let (vector_buf, mut vector_alloc) = create_buffer(&ctx.allocator, vector_size);
    println!("  vector buffer created");
    let (output_buf, mut output_alloc) = create_buffer(&ctx.allocator, output_size);
    println!("  output buffer created");

    // Initialize data: matrix[i][j] = i + j, vector = all 1s
    println!("Initializing data...");
    let matrix_data: Vec<f32> = (0..NROWS)
        .flat_map(|row| (0..NCOLS).map(move |col| (row + col) as f32))
        .collect();
    let vector_data: Vec<f32> = vec![1.0; NCOLS];
    println!("  data generated");

    println!("Writing to buffers...");
    write_buffer(&ctx.allocator, &mut matrix_alloc, &matrix_data);
    println!("  matrix written");
    write_buffer(&ctx.allocator, &mut vector_alloc, &vector_data);
    println!("  vector written");

    // Expected: row i dot all-ones = sum of row i = i*ncols + (0+1+...+(ncols-1))
    println!("Computing expected values...");
    let expected: Vec<f32> = (0..NROWS)
        .map(|row| (row * NCOLS) as f32 + (NCOLS * (NCOLS - 1) / 2) as f32)
        .collect();
    println!("  expected computed");

    // Record and execute
    println!("Allocating command buffer...");
    let cmd = ctx.allocate_command_buffer().unwrap();
    println!("  command buffer allocated");

    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    println!("Recording commands...");
    unsafe {
        ctx.device.begin_command_buffer(cmd, &begin_info).unwrap();
    }
    println!("  begin_command_buffer done");

    call_matvec(&kernels, cmd, NROWS, NCOLS, matrix_buf, vector_buf, output_buf)
        .expect("Failed to dispatch matvec");
    println!("  call_matvec done");

    unsafe {
        ctx.device.end_command_buffer(cmd).unwrap();
    }
    println!("  end_command_buffer done");

    println!("Submitting...");
    ctx.submit_and_wait(cmd).expect("Failed to execute");
    println!("  submit done");

    // Read back and verify
    let output: Vec<f32> = read_buffer(&ctx.allocator, &mut output_alloc, NROWS);

    println!("Output[0..5]: {:?}", &output[0..5]);
    println!("Expected[0..5]: {:?}", &expected[0..5]);

    for i in 0..NROWS {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff < 0.01,
            "Mismatch at {}: expected {}, got {} (diff {})",
            i,
            expected[i],
            output[i],
            diff
        );
    }

    println!("SUCCESS: All {} outputs match!", NROWS);

    // Cleanup
    unsafe {
        ctx.allocator.destroy_buffer(matrix_buf, &mut matrix_alloc);
        ctx.allocator.destroy_buffer(vector_buf, &mut vector_alloc);
        ctx.allocator.destroy_buffer(output_buf, &mut output_alloc);
    }
}
