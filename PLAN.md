// THE idiomatic Rust way for GPU data structures
#[repr(C)]  // Stable layout matching GGML and GPU expectations
#[derive(Copy, Clone, Pod, Zeroable)]  // Safe for GPU, no padding
pub struct BlockQ4_0 {
    pub d: f16,           // 2 bytes: scale factor
    pub qs: [u8; 16],     // 16 bytes: 32 4-bit weights packed
}

// Compile-time validation (Rust's safety guarantee)
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
```

This gives us:
- **Exact GGML layout compatibility** (18 bytes, no surprises)
- **GPU safety** (Pod trait ensures valid bit patterns)
- **Zero-cost abstractions** (direct field access)
- **Type safety** (no manual byte manipulation)

## Technology Stack

```toml
[dependencies]
ash = "0.38"                    # Thin Vulkan wrapper (no abstraction)
gpu-allocator = "0.27"          # Safe GPU memory allocation  
bytemuck = { version = "1.14", features = ["derive"] }  # GPU casting
half = "2.3"                    # f16 support
thiserror = "1.0"               # Error handling
parking_lot = "0.12"            # Better mutexes

[build-dependencies]  
shaderc = "0.8"                 # GLSL -> SPIR-V compilation
```

## Reference Files & Implementation Plan

### Week 1: Quantization Types & CPU Reference
**Study:**
- `ggml/include/ggml.h` lines 350-450: Block definitions
- `ggml/src/ggml-quants.h` lines 1-200: Detailed structs
- `ggml/src/ggml-quants.c` lines 500-800: CPU quantization

**Implementation:**
```rust
// candle-vulkan-kernels/src/quant_types.rs

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Q4_0: 32 4-bit weights per block, 4.5 bits/weight
/// Layout MUST match ggml-quants.h for GPU compatibility
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct BlockQ4_0 {
    /// Scale factor (delta) for dequantization
    pub d: f16,
    /// 32 4-bit signed integers packed: weight[2*i] | (weight[2*i+1] << 4)
    pub qs: [u8; 16],
}

// Compile-time layout validation (Rust safety!)
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::align_of::<BlockQ4_0>() == 2);

impl BlockQ4_0 {
    pub const WEIGHTS_PER_BLOCK: usize = 32;
    
    pub const fn zeroed() -> Self {
        Self { d: f16::ZERO, qs: [0; 16] }
    }
    
    /// Get weight at index (safe wrapper around bit manipulation)
    pub fn get_weight(&self, idx: usize) -> i8 {
        assert!(idx < Self::WEIGHTS_PER_BLOCK);
        let byte_idx = idx / 2;
        let shift = (idx & 1) * 4;
        let nibble = (self.qs[byte_idx] >> shift) & 0x0F;
        (nibble as i8) - 8  // Convert to signed: -8 to 7
    }
    
    /// Dequantize single weight to f32
    pub fn dequantize(&self, idx: usize) -> f32 {
        self.d.to_f32() * self.get_weight(idx) as f32
    }
}

/// CPU reference implementation (matches ggml-quants.c)
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    assert_eq!(src.len(), dst.len() * BlockQ4_0::WEIGHTS_PER_BLOCK);
    
    for (i, block) in dst.iter_mut().enumerate() {
        let input = &src[i * BlockQ4_0::WEIGHTS_PER_BLOCK..];
        
        // Find max absolute value (GGML algorithm)
        let amax = input.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 8.0;  // 4-bit signed range: -8 to 7
        block.d = f16::from_f32(d);
        
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        
        // Quantize and pack (exact GGML logic)
        for j in 0..16 {
            let x0 = (input[j * 2] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            let x1 = (input[j * 2 + 1] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            block.qs[j] = x0 | (x1 << 4);  // Safe bit packing
        }
    }
}
```

### Week 2: SPIR-V Shaders & Pipeline Creation
**Study:**
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_0_f32.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/types.comp`

**Implementation:**
```glsl
// candle-vulkan-kernels/src/shaders/mul_mat_vec_q4_0_f32.comp
// Copy exactly from GGML - portable GPU code

#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define QK4_0 32

// Must match Rust BlockQ4_0 layout exactly
struct BlockQ4_0 {
    float16_t d;
    uint8_t qs[16];
};

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { BlockQ4_0 data_a[]; };
layout(binding = 1) readonly buffer B { float data_b[]; };
layout(binding = 2) writeonly buffer D { float data_d[]; };

layout(push_constant) uniform PushConstants {
    uint ncols_x;
    uint nrows_x;
    uint nrows_y;
    uint nrows_dst;
    uint row_stride_x;
    uint channel_stride_x;
} p;

shared float tmp[32];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    
    const uint num_blocks_per_row = p.ncols_x / QK4_0;
    float sum = 0.0;
    
    // Process blocks assigned to this thread
    for (uint i = tid; i < num_blocks_per_row; i += 32) {
        const uint block_idx = row * num_blocks_per_row + i;
        BlockQ4_0 block = data_a[block_idx];
        
        float block_sum = 0.0;
        for (uint j = 0; j < QK4_0; ++j) {
            const uint b_idx = i * QK4_0 + j;
            float b_val = data_b[b_idx];
            float a_val = float(block.d) * float(((block.qs[j / 2] >> ((j & 1) * 4)) & 0x0F) - 8);
            block_sum += a_val * b_val;
        }
        sum += block_sum;
    }
    
    // Reduction in shared memory (same as GGML)
    tmp[tid] = sum;
    barrier();
    
    if (tid < 16) tmp[tid] += tmp[tid + 16];
    barrier();
    if (tid < 8) tmp[tid] += tmp[tid + 8];
    barrier();
    if (tid < 4) tmp[tid] += tmp[tid + 4];
    barrier();
    if (tid < 2) tmp[tid] += tmp[tid + 2];
    barrier();
    if (tid < 1) tmp[tid] += tmp[tid + 1];
    
    if (tid == 0) {
        data_d[row] = tmp[0];
    }
}
```

### Week 3: Vulkan Device Context
**Study:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` lines 100-300, 500-800

**Implementation:**
```rust
// candle-vulkan-kernels/src/device.rs

use anyhow::Result;
use ash::{vk, Device, Entry, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct VulkanContext {
    // Vulkan objects (RAII via Drop)
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    
    pub compute_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    
    pub allocator: Arc<Mutex<Allocator>>,
    pub pipelines: Pipelines,
    
    pub device_info: DeviceInfo,
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub subgroup_size: u32,
    pub max_workgroup_size: [u32; 3],
}

impl VulkanContext {
    pub fn new(device_index: usize) -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let physical_device = Self::select_device(&instance, device_index)?;
        
        let (device, compute_queue, compute_queue_family) = 
            Self::create_device(&instance, physical_device)?;
            
        let compute_pool = Self::create_command_pool(&device, compute_queue_family)?;
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        let allocator = Self::create_allocator(&instance, &device, physical_device)?;
        
        let device_info = Self::query_device_info(&instance, physical_device)?;
        let pipelines = Pipelines::new(&device)?;
        
        Ok(Self {
            instance, device, physical_device,
            compute_queue, compute_queue_family,
            compute_pool, descriptor_pool,
            allocator: Arc::new(Mutex::new(allocator)),
            pipelines,
            device_info,
        })
    }
    
    // Follow GGML's initialization pattern exactly
    fn create_instance(entry: &Entry) -> Result<Instance> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name("Candle Vulkan Kernels")
            .api_version(vk::API_VERSION_1_3);
        
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);
        
        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }
    
    // ... other methods follow GGML pattern ...
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.compute_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
```

### Week 4: Operation Dispatch
**Study:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` lines 1500-2000

**Implementation:**
```rust
// candle-vulkan-kernels/src/ops/matmul.rs

use crate::{device::VulkanContext, buffer::VulkanBuffer};
use ash::vk;

pub struct MatMulParams {
    pub m: usize,  // rows
    pub n: usize,  // columns  
    pub k: usize,  // inner dimension
}

pub fn mul_mat_vec_q4_0_f32(
    ctx: &VulkanContext,
    weights: &VulkanBuffer,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
    params: MatMulParams,
) -> Result<()> {
    assert_eq!(params.k % 32, 0, "K must be multiple of QK4_0");
    
    let cmd = ctx.allocate_command_buffer()?;
    
    unsafe {
        ctx.device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        
        // Allocate descriptor set (matches GGML pattern)
        let desc_set = ctx.allocate_descriptor_set(
            ctx.pipelines.mul_mat_vec_q4_0.descriptor_layout
        )?;
        
        // Update descriptors (slice API for safety)
        let buffer_infos = [
            vk::DescriptorBufferInfo { buffer: weights.handle(), offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: input.handle(), offset: 0, range: vk::WHOLE_SIZE },
            vk::DescriptorBufferInfo { buffer: output.handle(), offset: 0, range: vk::WHOLE_SIZE },
        ];
        
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[0..1]).build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[1..2]).build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(desc_set).dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[2..3]).build(),
        ];
        
        ctx.device.update_descriptor_sets(&writes, &[]);
        
        // Push constants (match shader layout exactly)
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct PushConstants {
            ncols_x: u32, nrows_x: u32, nrows_y: u32,
            nrows_dst: u32, row_stride_x: u32, channel_stride_x: u32,
        }
        
        let push = PushConstants {
            ncols_x: params.k as u32, nrows_x: params.m as u32, nrows_y: 1,
            nrows_dst: params.m as u32, row_stride_x: params.k as u32, channel_stride_x: 0,
        };
        
        // Bind and dispatch (GGML pattern)
        ctx.device.cmd_bind_pipeline(
            cmd, vk::PipelineBindPoint::COMPUTE,
            ctx.pipelines.mul_mat_vec_q4_0.pipeline,
        );
        
        ctx.device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE,
            ctx.pipelines.mul_mat_vec_q4_0.layout, 0,
            &[desc_set], &[],
        );
        
        ctx.device.cmd_push_constants(
            cmd, ctx.pipelines.mul_mat_vec_q4_0.layout,
            vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::bytes_of(&push),
        );
        
        ctx.device.cmd_dispatch(cmd, params.m as u32, 1, 1);
        ctx.device.end_command_buffer(cmd)?;
        
        ctx.submit_and_wait(&[cmd])?;
    }
    
    Ok(())
}
```

## File Structure

```
candle-vulkan-kernels/
├── Cargo.toml
├── build.rs                    # Compile shaders to SPIR-V
├── src/
│   ├── lib.rs                  # Public API
│   ├── quant_types.rs          # GPU-compatible data structures  
│   ├── device.rs               # Vulkan context (RAII)
│   ├── buffer.rs               # Safe Vulkan buffer wrapper
│   ├── pipelines/
│   │   ├── mod.rs
│   │   ├── mul_mat_vec_q4_0.rs # One pipeline per kernel
│   │   └── rope.rs
│   └── ops/
│       ├── mod.rs
│       ├── matmul.rs           # Operation dispatch
│       └── rope.rs
├── src/shaders/
│   ├── types.comp
│   ├── mul_mat_vec_q4_0_f32.comp
│   └── rope.comp
└── tests/
    ├── correctness.rs          # Validate against GGML
    └── performance.rs          # Benchmark CPU vs GPU
```

## Testing Strategy

```rust
#[test]
fn test_q4_0_layout_compatibility() {
    // Ensure our Rust struct matches GPU expectations
    let block = BlockQ4_0 {
        d: f16::from_f32(1.0),
        qs: [0xFF; 16],  // Max values
    };
    
    // Should be exactly 18 bytes like GGML
    assert_eq!(std::mem::size_of_val(&block), 18);
    
    // Should be 2-byte aligned for GPU
    assert_eq!(std::mem::align_of_val(&block), 2);
    
    // Field access should work (no padding)
    assert_eq!(block.qs[0], 0xFF);
}

#[test] 
fn test_quantization_roundtrip() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32) / 10.0).collect();
    let mut blocks = vec![BlockQ4_0::zeroed(); input.len() / 32];
    
    quantize_row_q4_0(&input, &mut blocks);
    
    let mut reconstructed = vec![0.0f32; input.len()];
    for (i, block) in blocks.iter().enumerate() {
        for j in 0..32 {
            reconstructed[i * 32 + j] = block.dequantize(j);
        }
    }
    
    // Should be close (quantization error)
    for (orig, recon) in input.iter().zip(&reconstructed) {
        let error = (orig - recon).abs();
        assert!(error < 0.1, "Quantization error too large: {}", error);
    }
}

#[test]
fn test_matmul_correctness() {
    let ctx = VulkanContext::new(0).unwrap();
    
    // Small test case for debugging
    let weights = create_q4_0_matrix(64, 64);
    let input = create_f32_vector(64);
    
    let expected = matmul_q4_0_cpu_reference(&weights, &input);
    let actual = mul_mat_vec_q4_0_f32(
        &ctx, &weights, &input, &mut create_output_buffer(64),
        MatMulParams { m: 64, n: 64, k: 64 }
    ).unwrap();
    
    assert_vectors_close(&expected, &actual, 1e-3);
}
```

## Success Criteria

### ✅ You're doing it right if:
- `#[repr(C)] + bytemuck::Pod` for all GPU data structures
- Shaders copied exactly from GGML GLSL
- Block sizes match (18 bytes for Q4_0, etc.)
- Using `gpu-allocator` for memory management
- RAII cleanup via `Drop` trait
- Tests validate against CPU reference implementations
- Only unsafe is for Vulkan API calls

### ❌ Red flags:
- Fighting the compiler over memory layout (use `#[repr(C)]`)
- Raw pointers where safe abstractions work
- Manual memory allocation (no VMA, no malloc)
- Generic abstractions over proven patterns
- Skipping layout validation tests
- Rewriting GGML algorithms instead of porting them

## Key Decisions

### Why ash instead of vulkano?
- ash lets us mirror GGML's exact Vulkan patterns
- No abstraction hiding performance-critical details
- Industry standard for low-level Vulkan in Rust

### Why gpu-allocator instead of VMA?
- Safe Rust API (no raw handles)
- Better error handling with `Result<T>`
- Automatic cleanup via `Drop` trait
- Same performance as VMA

### Why #[repr(C)] + Pod?
- Documented, idiomatic Rust for FFI/GPU
- Stable layout guarantee (GGML compatibility)
- Zero-cost abstractions
- Type safety (no manual byte manipulation)

## Quick Start

1. **Start with `quant_types.rs`** - Foundation for everything
2. **Copy shaders from GGML** - Portable GPU code  
3. **Follow GGML's Vulkan patterns** - ash lets us match them exactly
4. **Test against CPU reference** - Validate correctness before GPU
5. **Profile with `tracing`** - Identify bottlenecks early
6. **Use `#[repr(C)] + Pod`** - This is the idiomatic Rust way for GPU data

## Timeline

- **Week 1**: Quantization types + CPU reference implementation
- **Week 2**: SPIR-V shaders + pipeline creation with ash
- **Week 3**: Vulkan device context + memory management
- **Week 4**: Operation dispatch + integration testing  
- **Week 5**: Performance optimization + Candle integration

Total: ~5 weeks for a working, tested, performant implementation that matches GGML's performance characteristics.