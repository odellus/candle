//! Binary operation kernels (add, mul, div, sub)

use crate::error::Result;
use crate::kernels::{source, Kernels};
use ash::vk;

/// Push constants for binary ops (using generic_binary_head.glsl)
/// This matches the layout in generic_binary_head.glsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinaryParams {
    pub ne: u32,           // total number of elements in output
    // Source 0 (lhs) shape and strides (in elements, not bytes)
    pub ne00: u32, pub ne01: u32, pub ne02: u32, pub ne03: u32,
    pub nb00: u32, pub nb01: u32, pub nb02: u32, pub nb03: u32,
    // Source 1 (rhs) shape and strides
    pub ne10: u32, pub ne11: u32, pub ne12: u32, pub ne13: u32,
    pub nb10: u32, pub nb11: u32, pub nb12: u32, pub nb13: u32,
    // Destination shape and strides
    pub ne20: u32, pub ne21: u32, pub ne22: u32, pub ne23: u32,
    pub nb20: u32, pub nb21: u32, pub nb22: u32, pub nb23: u32,
    pub misalign_offsets: u32,
    pub param1: f32,
    pub param2: f32,
    pub param3: i32,
}

/// Shape and stride info for a tensor (up to 4D)
#[derive(Debug, Clone, Copy)]
pub struct TensorLayout {
    /// Shape dimensions (ne0 is innermost/fastest)
    pub shape: [u32; 4],
    /// Strides in elements (nb0 is innermost)
    pub strides: [u32; 4],
    /// Element offset into the buffer
    pub offset: u32,
}

impl TensorLayout {
    /// Create layout for a contiguous 1D tensor
    pub fn contiguous_1d(n: u32) -> Self {
        Self {
            shape: [n, 1, 1, 1],
            strides: [1, n, n, n],
            offset: 0,
        }
    }

    /// Create layout from shape, strides, and offset (candle uses row-major, outermost first)
    /// shape: [dim0, dim1, ...] where dim0 is outermost
    /// strides: [stride0, stride1, ...] where stride0 is outermost
    /// offset: element offset into the buffer (from layout.start_offset())
    pub fn from_shape_strides_offset(shape: &[usize], strides: &[usize], offset: usize) -> Self {
        let mut s = [1u32; 4];
        let mut st = [0u32; 4];

        let ndim = shape.len().min(4);

        // Map from candle's [outermost..innermost] to ggml's [innermost..outermost]
        // candle: shape[0] is outermost, strides[0] is outermost
        // ggml: ne00 is innermost, ne03 is outermost
        for i in 0..ndim {
            let ggml_idx = ndim - 1 - i; // reverse order
            s[ggml_idx] = shape[i] as u32;
            st[ggml_idx] = strides[i] as u32;
        }

        // Fill remaining dimensions
        for i in ndim..4 {
            s[i] = 1;
            st[i] = st[i.saturating_sub(1)];
        }

        Self { shape: s, strides: st, offset: offset as u32 }
    }

    /// Create layout from shape and strides (no offset)
    pub fn from_shape_strides(shape: &[usize], strides: &[usize]) -> Self {
        Self::from_shape_strides_offset(shape, strides, 0)
    }

    /// Check if this layout is contiguous
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1u32;
        for i in 0..4 {
            if self.strides[i] != expected_stride && self.shape[i] > 1 {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Total number of elements
    pub fn num_elements(&self) -> u32 {
        self.shape.iter().product()
    }
}

impl BinaryParams {
    /// Create params for simple contiguous tensors of the same shape
    pub fn contiguous(num_elements: usize) -> Self {
        let ne = num_elements as u32;
        Self {
            ne,
            ne00: ne, ne01: 1, ne02: 1, ne03: 1,
            nb00: 1, nb01: ne, nb02: ne, nb03: ne,
            ne10: ne, ne11: 1, ne12: 1, ne13: 1,
            nb10: 1, nb11: ne, nb12: ne, nb13: ne,
            ne20: ne, ne21: 1, ne22: 1, ne23: 1,
            nb20: 1, nb21: ne, nb22: ne, nb23: ne,
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        }
    }

    /// Create params from tensor layouts with broadcasting support
    pub fn from_layouts(lhs: &TensorLayout, rhs: &TensorLayout, out: &TensorLayout) -> Self {
        // Pack offsets into misalign_offsets:
        // In generic_binary_head.glsl:
        //   get_aoffset() = misalign_offsets >> 16       (bits 16-31, but really only 8 bits used)
        //   get_boffset() = (misalign_offsets >> 8) & 0xFF (bits 8-15)
        //   get_doffset() = misalign_offsets & 0xFF      (bits 0-7)
        // Note: ggml uses 8-bit offsets for binary ops (max 255)
        let misalign_offsets = ((lhs.offset & 0xFF) << 16)
                             | ((rhs.offset & 0xFF) << 8)
                             | (out.offset & 0xFF);

        Self {
            ne: out.num_elements(),
            // lhs (src0)
            ne00: lhs.shape[0], ne01: lhs.shape[1], ne02: lhs.shape[2], ne03: lhs.shape[3],
            nb00: lhs.strides[0], nb01: lhs.strides[1], nb02: lhs.strides[2], nb03: lhs.strides[3],
            // rhs (src1)
            ne10: rhs.shape[0], ne11: rhs.shape[1], ne12: rhs.shape[2], ne13: rhs.shape[3],
            nb10: rhs.strides[0], nb11: rhs.strides[1], nb12: rhs.strides[2], nb13: rhs.strides[3],
            // output (dst)
            ne20: out.shape[0], ne21: out.shape[1], ne22: out.shape[2], ne23: out.shape[3],
            nb20: out.strides[0], nb21: out.strides[1], nb22: out.strides[2], nb23: out.strides[3],
            misalign_offsets,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        }
    }
}

/// Binary operation types
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Mul,
    Div,
}

/// Execute a binary operation with full layout support: output = a op b
pub fn call_binary_strided(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    op: BinaryOp,
    params: &BinaryParams,
    input_a: vk::Buffer,
    input_b: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    let device = &kernels.context().device;

    // Select shader based on operation
    let (name, spirv): (&'static str, &'static [u8]) = match op {
        BinaryOp::Add => ("add_f32", source::ADD_F32),
        BinaryOp::Mul => ("mul_f32", source::MUL_F32),
        BinaryOp::Div => ("div_f32", source::DIV_F32),
    };

    // Load pipeline (3 buffers: input_a, input_b, output; push constants)
    let cached = kernels.load_pipeline(
        name,
        spirv,
        3,
        std::mem::size_of::<BinaryParams>() as u32,
    )?;

    // Allocate and update descriptor set
    let descriptor_set = kernels.allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buffer_infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(input_a)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(input_b)
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

    // Calculate dispatch size
    // The shader uses: gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x
    // Each workgroup is 256 threads, processing 2 elements each (num_iter = 2)
    let elements_per_workgroup = 256 * 2;
    let total_workgroups = (num_elements as u32 + elements_per_workgroup - 1) / elements_per_workgroup;

    let (wx, wy, wz) = if total_workgroups <= 512 {
        (total_workgroups.max(1), 1, 1)
    } else if total_workgroups <= 512 * 512 {
        (512, (total_workgroups + 511) / 512, 1)
    } else {
        let z = (total_workgroups + 262143) / 262144;
        (512, 512, z.min(65535))
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
            bytemuck::bytes_of(params),
        );

        device.cmd_dispatch(command_buffer, wx, wy, wz);
    }

    Ok(())
}

/// Execute a binary operation: output = a op b (simple contiguous version)
pub fn call_binary(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    op: BinaryOp,
    num_elements: usize,
    input_a: vk::Buffer,
    input_b: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    let params = BinaryParams::contiguous(num_elements);
    call_binary_strided(kernels, command_buffer, op, &params, input_a, input_b, output)
}

/// Execute add: output = a + b
pub fn call_add(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input_a: vk::Buffer,
    input_b: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_binary(kernels, command_buffer, BinaryOp::Add, num_elements, input_a, input_b, output)
}

/// Execute mul: output = a * b
pub fn call_mul(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input_a: vk::Buffer,
    input_b: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_binary(kernels, command_buffer, BinaryOp::Mul, num_elements, input_a, input_b, output)
}

/// Execute div: output = a / b
pub fn call_div(
    kernels: &Kernels,
    command_buffer: vk::CommandBuffer,
    num_elements: usize,
    input_a: vk::Buffer,
    input_b: vk::Buffer,
    output: vk::Buffer,
) -> Result<()> {
    call_binary(kernels, command_buffer, BinaryOp::Div, num_elements, input_a, input_b, output)
}
