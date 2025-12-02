//! Matrix-vector multiplication kernels
//!
//! Includes both f32 matvec and quantized Q4K matvec

use crate::error::Result;
use crate::kernels::{source, Kernels};
use ash::vk;

/// Q4K block size (256 elements per block)
pub const Q4K_BLOCK_SIZE: usize = 256;

/// Push constants for matvec shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatVecParams {
    pub nrows: u32,
    pub ncols: u32,
}

/// Execute matrix-vector multiplication: output = matrix * vector
///
/// # Arguments
/// * `kernels` - Kernel manager
/// * `command_buffer` - Command buffer to record into
/// * `nrows` - Number of rows in matrix (output size)
/// * `ncols` - Number of columns in matrix (input vector size)
/// * `matrix` - Matrix buffer [nrows x ncols], row-major
/// * `vector` - Input vector buffer [ncols]
/// * `output` - Output vector buffer [nrows]
pub fn call_matvec(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    nrows: usize,
    ncols: usize,
    matrix: vk::Buffer,
    vector: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    let device = &kernels.context().device;

    // Load pipeline (3 buffers, 8 bytes push constants)
    let cached = kernels.load_pipeline(
        "matvec",
        source::MATVEC,
        3,
        std::mem::size_of::<MatVecParams>() as u32,
    )?;

    // Allocate and update descriptor set
    let descriptor_set = kernels.allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(matrix)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(vector)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(output)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];

    let writes: Vec<_> = buffer_infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    // Record commands
    let params = MatVecParams {
        nrows: nrows as u32,
        ncols: ncols as u32,
    };

    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, cached.pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[descriptor_set],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&params),
        );

        // One workgroup per row
        device.cmd_dispatch(command_buffer, nrows as u32, 1, 1);
    }

    Ok(())
}

/// Push constants for Q4K matmul shader (matches mul_mat_vec_base.glsl)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MulMatVecQ4KParams {
    pub ncols: u32,
    pub stride_a: u32,
    pub stride_b: u32,
    pub stride_d: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub ne02: u32,
    pub ne12: u32,
    pub broadcast2: u32,
    pub broadcast3: u32,
}

/// Execute Q4K quantized matrix-vector multiplication: output = q4k_matrix @ vector
///
/// This performs matrix-vector multiplication where the matrix is stored in Q4K
/// quantized format (4-bit k-quant with 256 elements per block).
///
/// # Arguments
/// * `kernels` - Kernel manager
/// * `command_buffer` - Command buffer to record into
/// * `nrows` - Number of rows in matrix (output dimension)
/// * `ncols` - Number of columns in matrix (must be multiple of 256)
/// * `q4k_matrix` - Q4K quantized matrix buffer
/// * `vector` - Input f32 vector buffer [ncols]
/// * `output` - Output f32 vector buffer [nrows]
pub fn call_mul_mat_vec_q4k(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    nrows: usize,
    ncols: usize,
    q4k_matrix: vk::Buffer,
    vector: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    assert!(ncols % Q4K_BLOCK_SIZE == 0, "ncols must be multiple of Q4K block size (256)");

    let device = &kernels.context().device;

    // Load pipeline (3 buffers, push constants size)
    let cached = kernels.load_pipeline(
        "mul_mat_vec_q4_k_f32",
        source::MUL_MAT_VEC_Q4_K_F32,
        3,
        std::mem::size_of::<MulMatVecQ4KParams>() as u32,
    )?;

    // Allocate and update descriptor set
    let descriptor_set = kernels.allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(q4k_matrix)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(vector)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(output)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];

    let writes: Vec<_> = buffer_infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    // Set up push constants for simple single-batch case
    let params = MulMatVecQ4KParams {
        ncols: ncols as u32,
        stride_a: ncols as u32,
        stride_b: ncols as u32,
        stride_d: nrows as u32,
        batch_stride_a: (nrows * ncols) as u32,
        batch_stride_b: ncols as u32,
        batch_stride_d: nrows as u32,
        ne02: 1,
        ne12: 1,
        broadcast2: 1,
        broadcast3: 1,
    };

    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, cached.pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[descriptor_set],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&params),
        );

        // Dispatch: one workgroup per row, batch_size=1
        // The shader uses NUM_ROWS spec constant (default 1)
        device.cmd_dispatch(command_buffer, nrows as u32, 1, 1);
    }

    Ok(())
}

/// Execute batched Q4K quantized matrix-vector multiplication
///
/// For each batch: output[batch] = q4k_matrix @ vector[batch]
///
/// # Arguments
/// * `kernels` - Kernel manager
/// * `command_buffer` - Command buffer to record into
/// * `batch_size` - Number of batches
/// * `nrows` - Number of rows in matrix (output dimension, n)
/// * `ncols` - Number of columns in matrix (k, must be multiple of 256)
/// * `q4k_matrix` - Q4K quantized matrix buffer [nrows x ncols]
/// * `vector` - Input f32 vector buffer [batch_size x ncols]
/// * `vector_offset` - Byte offset into vector buffer
/// * `output` - Output f32 vector buffer [batch_size x nrows]
/// * `output_offset` - Byte offset for each batch output (typically batch_id * nrows * 4)
pub fn call_mul_mat_vec_q4k_batched(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    batch_size: usize,
    nrows: usize,
    ncols: usize,
    q4k_matrix: vk::Buffer,
    vector: vk::Buffer,
    vector_offset: usize,
    output: vk::Buffer,
) -> Result<()> {
    assert!(ncols % Q4K_BLOCK_SIZE == 0, "ncols must be multiple of Q4K block size (256)");

    let device = &kernels.context().device;

    // Load pipeline
    let cached = kernels.load_pipeline(
        "mul_mat_vec_q4_k_f32",
        source::MUL_MAT_VEC_Q4_K_F32,
        3,
        std::mem::size_of::<MulMatVecQ4KParams>() as u32,
    )?;

    // Allocate and update descriptor set
    let descriptor_set = kernels.allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(q4k_matrix)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(vector)
            .offset(vector_offset as u64)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(output)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];

    let writes: Vec<_> = buffer_infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    // Set up push constants for batched case
    // The shader handles batching via gl_GlobalInvocationID.y
    let params = MulMatVecQ4KParams {
        ncols: ncols as u32,
        stride_a: ncols as u32,
        stride_b: ncols as u32,
        stride_d: nrows as u32,
        batch_stride_a: (nrows * ncols) as u32,  // Not used for single weight matrix
        batch_stride_b: ncols as u32,            // Stride between batch inputs
        batch_stride_d: nrows as u32,            // Stride between batch outputs
        ne02: 1,
        ne12: batch_size as u32,
        broadcast2: 1,
        broadcast3: 1,
    };

    unsafe {
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, cached.pipeline);

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[descriptor_set],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&params),
        );

        // Dispatch: x=num rows, y=batch_size, z=1
        device.cmd_dispatch(command_buffer, nrows as u32, batch_size as u32, 1);
    }

    Ok(())
}
