//! Unary operation kernels (exp, silu, gelu, relu, sqrt, sin, cos, etc.)

use crate::error::Result;
use crate::kernels::{source, Kernels};
use ash::vk;

/// Push constants for simple unary ops (using generic_head.glsl)
/// Matches the layout in generic_head.glsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnarySimpleParams {
    pub kx: u32,    // number of elements
    pub ky: u32,    // unused for simple unary
    pub param1: f32,
    pub param2: f32,
}

/// Unary operation types
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Exp,
    Silu,
    Gelu,
    Relu,
}

/// Execute a simple unary operation: output = op(input)
/// Uses shaders with generic_head.glsl layout
pub fn call_unary_simple(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    op: UnaryOp,
    num_elements: usize,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    let device = &kernels.context().device;

    // Select shader based on operation
    let (name, spirv): (&'static str, &'static [u8]) = match op {
        UnaryOp::Exp => ("exp_f32", source::EXP_F32),
        UnaryOp::Silu => ("silu_f32", source::SILU_F32),
        UnaryOp::Gelu => ("gelu_f32", source::GELU_F32),
        UnaryOp::Relu => ("relu_f32", source::RELU_F32),
    };

    // Load pipeline (2 buffers: input, output; 16 bytes push constants)
    let cached = kernels.load_pipeline(
        name,
        spirv,
        2,
        std::mem::size_of::<UnarySimpleParams>() as u32,
    )?;

    // Allocate and update descriptor set
    let descriptor_set = kernels.allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(input)
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
    let params = UnarySimpleParams {
        kx: num_elements as u32,
        ky: 0,
        param1: 0.0,
        param2: 0.0,
    };

    // Calculate dispatch size: workgroup size is 512, each thread handles one element
    let num_workgroups_x = (num_elements as u32 + 511) / 512;
    let num_workgroups_y = 1;
    let num_workgroups_z = 1;

    // Handle very large tensors by spreading across y and z dimensions
    let (wx, wy, wz) = if num_workgroups_x > 65535 {
        let total = num_workgroups_x;
        let z = (total + 262143) / 262144;
        let remaining = (total + z - 1) / z;
        let y = (remaining + 511) / 512;
        let x = 512.min(remaining);
        (x, y.min(512), z.min(65535))
    } else {
        (num_workgroups_x, num_workgroups_y, num_workgroups_z)
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

        device.cmd_dispatch(command_buffer, wx, wy, wz);
    }

    Ok(())
}

/// Execute exp: output = exp(input)
pub fn call_exp(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_unary_simple(kernels, command_buffer, UnaryOp::Exp, num_elements, input, output)
}

/// Execute silu: output = silu(input)
pub fn call_silu(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_unary_simple(kernels, command_buffer, UnaryOp::Silu, num_elements, input, output)
}

/// Execute gelu: output = gelu(input)
pub fn call_gelu(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_unary_simple(kernels, command_buffer, UnaryOp::Gelu, num_elements, input, output)
}

/// Execute relu: output = relu(input)
pub fn call_relu(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_unary_simple(kernels, command_buffer, UnaryOp::Relu, num_elements, input, output)
}
