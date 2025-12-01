//! Matrix-vector multiplication kernel

use crate::error::Result;
use crate::kernels::{source, Kernels};
use ash::vk;

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
