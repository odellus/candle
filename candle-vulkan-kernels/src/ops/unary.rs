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

/// Push constants for strided unary ops (using generic_unary_head.glsl)
/// Matches the layout in generic_unary_head.glsl exactly
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnaryStridedParams {
    pub ne: u32,  // total number of output elements

    // Source tensor shape and strides (in elements)
    pub ne00: u32, pub ne01: u32, pub ne02: u32, pub ne03: u32,
    pub nb00: u32, pub nb01: u32, pub nb02: u32, pub nb03: u32,

    // Destination tensor shape and strides (in elements)
    pub ne10: u32, pub ne11: u32, pub ne12: u32, pub ne13: u32,
    pub nb10: u32, pub nb11: u32, pub nb12: u32, pub nb13: u32,

    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,

    // Fast division magic values for source tensor
    pub ne0_012mp: u32, pub ne0_012L: u32,
    pub ne0_01mp: u32, pub ne0_01L: u32,
    pub ne0_0mp: u32, pub ne0_0L: u32,

    // Fast division magic values for destination tensor
    pub ne1_012mp: u32, pub ne1_012L: u32,
    pub ne1_01mp: u32, pub ne1_01L: u32,
    pub ne1_0mp: u32, pub ne1_0L: u32,
}

/// Compute magic multiplier and shift for fast division
/// Implements: n/d = (mulhi(n, mp) + n) >> L
/// See init_fastdiv_values in ggml-vulkan.cpp
fn init_fastdiv_values(d: u32) -> (u32, u32) {
    if d == 0 {
        return (0, 0);
    }

    // Compute L = ceil(log2(d))
    let mut l: u32 = 0;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }

    // Compute mp = (2^32 * (2^L - d) / d) + 1
    let mp = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64 + 1) as u32;

    (mp, l)
}

impl UnaryStridedParams {
    /// Create params from shape and strides with offset
    /// shape/strides are in candle order: [outermost..innermost]
    /// src_offset is the element offset into the source buffer (from layout.start_offset())
    pub fn from_shape_strides_offset(
        src_shape: &[usize],
        src_strides: &[usize],
        dst_shape: &[usize],
        dst_strides: &[usize],
        src_offset: usize,
    ) -> Self {
        // Map from candle's [outermost..innermost] to ggml's [innermost..outermost]
        let ndim = src_shape.len().min(4);

        let mut ne0 = [1u32; 4];
        let mut nb0 = [0u32; 4];
        let mut ne1 = [1u32; 4];
        let mut nb1 = [0u32; 4];

        for i in 0..ndim {
            let ggml_idx = ndim - 1 - i;
            ne0[ggml_idx] = src_shape[i] as u32;
            nb0[ggml_idx] = src_strides[i] as u32;
            ne1[ggml_idx] = dst_shape[i] as u32;
            nb1[ggml_idx] = dst_strides[i] as u32;
        }

        // Fill remaining with sensible defaults
        for i in ndim..4 {
            ne0[i] = 1;
            nb0[i] = nb0[i.saturating_sub(1)];
            ne1[i] = 1;
            nb1[i] = nb1[i.saturating_sub(1)];
        }

        let num_elements: u32 = dst_shape.iter().map(|&x| x as u32).product();

        // Compute fastdiv values for source tensor
        let (ne0_012mp, ne0_012L) = init_fastdiv_values(ne0[2] * ne0[1] * ne0[0]);
        let (ne0_01mp, ne0_01L) = init_fastdiv_values(ne0[1] * ne0[0]);
        let (ne0_0mp, ne0_0L) = init_fastdiv_values(ne0[0]);

        // Compute fastdiv values for destination tensor
        let (ne1_012mp, ne1_012L) = init_fastdiv_values(ne1[2] * ne1[1] * ne1[0]);
        let (ne1_01mp, ne1_01L) = init_fastdiv_values(ne1[1] * ne1[0]);
        let (ne1_0mp, ne1_0L) = init_fastdiv_values(ne1[0]);

        // Pack offsets: src_offset in upper 16 bits, dst_offset in lower 16 bits
        // In generic_unary_head.glsl: get_aoffset() = misalign_offsets >> 16
        //                             get_doffset() = misalign_offsets & 0xFFFF
        let misalign_offsets = ((src_offset as u32 & 0xFFFF) << 16) | 0u32;

        Self {
            ne: num_elements,
            ne00: ne0[0], ne01: ne0[1], ne02: ne0[2], ne03: ne0[3],
            nb00: nb0[0], nb01: nb0[1], nb02: nb0[2], nb03: nb0[3],
            ne10: ne1[0], ne11: ne1[1], ne12: ne1[2], ne13: ne1[3],
            nb10: nb1[0], nb11: nb1[1], nb12: nb1[2], nb13: nb1[3],
            misalign_offsets,
            param1: 0.0,
            param2: 0.0,
            ne0_012mp, ne0_012L,
            ne0_01mp, ne0_01L,
            ne0_0mp, ne0_0L,
            ne1_012mp, ne1_012L,
            ne1_01mp, ne1_01L,
            ne1_0mp, ne1_0L,
        }
    }

    /// Create params from shape and strides (no offset)
    /// shape/strides are in candle order: [outermost..innermost]
    pub fn from_shape_strides(
        src_shape: &[usize],
        src_strides: &[usize],
        dst_shape: &[usize],
        dst_strides: &[usize],
    ) -> Self {
        Self::from_shape_strides_offset(src_shape, src_strides, dst_shape, dst_strides, 0)
    }

    /// Create params for contiguous tensor (both src and dst have same contiguous layout)
    pub fn contiguous(shape: &[usize]) -> Self {
        let ndim = shape.len().min(4);

        let mut ne = [1u32; 4];
        let mut nb = [1u32; 4];

        // Map to ggml order (innermost first)
        for i in 0..ndim {
            let ggml_idx = ndim - 1 - i;
            ne[ggml_idx] = shape[i] as u32;
        }

        // Compute contiguous strides
        nb[0] = 1;
        for i in 1..4 {
            nb[i] = nb[i - 1] * ne[i - 1];
        }

        let num_elements: u32 = shape.iter().map(|&x| x as u32).product();

        // Compute fastdiv values
        let (ne0_012mp, ne0_012L) = init_fastdiv_values(ne[2] * ne[1] * ne[0]);
        let (ne0_01mp, ne0_01L) = init_fastdiv_values(ne[1] * ne[0]);
        let (ne0_0mp, ne0_0L) = init_fastdiv_values(ne[0]);

        Self {
            ne: num_elements,
            ne00: ne[0], ne01: ne[1], ne02: ne[2], ne03: ne[3],
            nb00: nb[0], nb01: nb[1], nb02: nb[2], nb03: nb[3],
            ne10: ne[0], ne11: ne[1], ne12: ne[2], ne13: ne[3],
            nb10: nb[0], nb11: nb[1], nb12: nb[2], nb13: nb[3],
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            ne0_012mp, ne0_012L,
            ne0_01mp, ne0_01L,
            ne0_0mp, ne0_0L,
            ne1_012mp: ne0_012mp, ne1_012L: ne0_012L,
            ne1_01mp: ne0_01mp, ne1_01L: ne0_01L,
            ne1_0mp: ne0_0mp, ne1_0L: ne0_0L,
        }
    }
}

/// Unary operation types
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Exp,
    Silu,
    Gelu,
    Relu,
}

/// Execute a simple unary operation (contiguous tensors): output = op(input)
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
    let (wx, wy, wz) = compute_dispatch_size(num_elements);

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

/// Execute a strided unary operation: output = op(input)
/// Uses shaders with generic_unary_head.glsl layout for full stride support
pub fn call_unary_strided(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    op: UnaryOp,
    params: &UnaryStridedParams,
    input: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    let device = &kernels.context().device;

    // Select strided shader based on operation
    let (name, spirv): (&'static str, &'static [u8]) = match op {
        UnaryOp::Exp => ("exp_f32_strided", source::EXP_F32_STRIDED),
        UnaryOp::Silu => ("silu_f32_strided", source::SILU_F32_STRIDED),
        UnaryOp::Gelu => ("gelu_f32_strided", source::GELU_F32_STRIDED),
        UnaryOp::Relu => ("relu_f32_strided", source::RELU_F32_STRIDED),
    };

    // Load pipeline (2 buffers: input, output; full push constants)
    let cached = kernels.load_pipeline(
        name,
        spirv,
        2,
        std::mem::size_of::<UnaryStridedParams>() as u32,
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

    let num_elements = params.ne as usize;
    let (wx, wy, wz) = compute_dispatch_size(num_elements);

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
            bytemuck::bytes_of(params),
        );

        device.cmd_dispatch(command_buffer, wx, wy, wz);
    }

    Ok(())
}

/// Compute dispatch dimensions for a given number of elements
/// Workgroup size is 512, spreads across x, y, z for large tensors
fn compute_dispatch_size(num_elements: usize) -> (u32, u32, u32) {
    let num_workgroups_x = (num_elements as u32 + 511) / 512;

    if num_workgroups_x <= 65535 {
        (num_workgroups_x.max(1), 1, 1)
    } else {
        // Spread across y and z for very large tensors
        // Global ID = z * 262144 + y * 512 + x
        let total = num_workgroups_x;
        let z = (total + 262143) / 262144;
        let remaining = (total + z - 1) / z;
        let y = (remaining + 511) / 512;
        let x = 512.min(remaining);
        (x, y.min(512), z.min(65535))
    }
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
