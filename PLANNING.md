# PLANNING.md - GGML-Vulkan Port to Rust/Candle

## Philosophy: GGML Performance with Idiomatic Rust

**Goal:** Port ggml-vulkan's battle-tested quantized inference to Candle using modern Rust patterns, NOT C++ translation. We want GGML's performance with Rust's safety and maintainability.

## The Approach

### ✅ DO THIS:
- **SPIR-V shaders**: Copy GLSL exactly (portable GPU code)
- **Data layouts**: Match GGML logic exactly but use Rust types (arrays, structs, no C padding)
- **Algorithms**: Copy proven GGML approaches (same logic, Rust idioms)
- **Memory management**: Use safe Rust (gpu-allocator, Arc, smart pointers)
- **Vulkan calls**: Mirror GGML's exact patterns using ash crate
- **Error handling**: Result<T> instead of error codes
- **Resource management**: RAII (Drop trait) instead of manual cleanup

### ❌ DON'T DO THIS:
- Literal C++ to Rust syntax translation
- `#[repr(C)]` and manual memory layout management
- Raw pointers where safe abstractions work
- Manual memory management (VMA, etc.)
- Generic abstractions over proven patterns
- "Improving" algorithms before proving correctness
- Using unsafe to bypass Rust's safety guarantees

## Why This Matters

GGML achieves 100-200x speedups through:
- Carefully optimized quantization blocks
- On-GPU dequantization (never move unquantized data to CPU)
- Specialized kernels for each quantization type
- Zero-copy memory management

We want:
- **Same performance**: Proven algorithms and GPU patterns
- **Rust safety**: No undefined behavior, leverage type system
- **Maintainability**: Idiomatic Rust that future developers love

## Technology Stack

```toml
[dependencies]
ash = "0.38"                    # Vulkan bindings (thin wrapper, no abstraction)
gpu-allocator = "0.27"          # Safe GPU memory allocation
bytemuck = { version = "1.14", features = ["derive"] }  # Safe casting for GPU data
half = "2.3"                    # f16 support
thiserror = "1.0"               # Error handling
parking_lot = "0.12"            # Better mutexes than std::sync
anyhow = "1.0"                  # Error propagation

[build-dependencies]
shaderc = "0.8"                 # GLSL -> SPIR-V compilation
```

## Reference Files in llama.cpp Repository

### Phase 1: Data Structures (Week 1)
**Study:**
- `ggml/include/ggml.h` lines 350-450: Quantization block definitions
- `ggml/src/ggml-quants.h` lines 1-200: Detailed quantization structs  
- `ggml/src/ggml-quants.c` lines 500-800: CPU quantization reference

**Your Implementation:**
```rust
// candle-vulkan-kernels/src/quant_types.rs

use bytemuck::{Pod, Zeroable};
use half::f16;

pub const QK4_0: usize = 32;

// Safe Rust version matching GGML logic exactly
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockQ4_0 {
    pub d: f16,           // Scale factor (half-precision float)
    pub qs: [u8; 16],     // Packed 4-bit quant values (32 values in 16 bytes)
}

// Compile-time verification (Rust way)
static_assert!(std::mem::size_of::<BlockQ4_0>() == 18, 
    "BlockQ4_0 must be exactly 18 bytes like GGML");

// CPU reference implementation (port GGML logic safely)
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    assert_eq!(src.len(), dst.len() * QK4_0, "Input size must match block count");
    
    for (i, block) in dst.iter_mut().enumerate() {
        let input_slice = &src[i * QK4_0..(i + 1) * QK4_0];
        
        // Find max absolute value (GGML algorithm)
        let amax = input_slice.iter().copied().fold(0.0f32, |a, &x| a.max(x.abs()));
        let d = amax / 8.0; // 4-bit signed range: -8 to 7
        block.d = f16::from_f32(d);
        
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        
        // Quantize and pack (bit manipulation in Rust)
        for j in 0..16 {
            let x0 = (input_slice[j * 2] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            let x1 = (input_slice[j * 2 + 1] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
            block.qs[j] = x0 | (x1 << 4); // Safe bit packing
        }
    }
}

// Safe dequantization
pub fn dequantize_row_q4_0(block: &BlockQ4_0, dst: &mut [f32]) {
    assert_eq!(dst.len(), QK4_0, "Output must be exactly QK4_0 elements");
    
    let d = f16::to_f32(block.d);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    
    for j in 0..16 {
        let packed = block.qs[j];
        // Extract 4-bit values safely
        let x0 = ((packed & 0x0F) as i8 - 8) as f32 * id;
        let x1 = ((packed >> 4) as i8 - 8) as f32 * id;
        dst[j * 2] = x0;
        dst[j * 2 + 1] = x1;
    }
}
```

### Phase 2: SPIR-V Shaders (Week 2)
**Study:**
- `ggml/src/ggml-vulkan/vulkan-shaders/types.comp` lines 1-100
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_0_f32.comp` lines 1-150
- `ggml/src/ggml-vulkan/vulkan-shaders/rope.comp` lines 1-120

**Your Implementation:**
```glsl
// candle-vulkan-kernels/src/shaders/mul_mat_vec_q4_0_f32.comp
// Copy exactly from GGML - this is portable GPU code

#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define QK4_0 32

struct BlockQ4_0 {
    float16_t d;
    uint8_t qs[16];
};

// Helper to extract 4-bit signed value
int get_q4_0(uint8_t packed, int idx) {
    int shift = (idx & 1) * 4;
    return int((packed >> shift) & 0x0F) - 8; // Signed: -8 to 7
}

// Dequantize a single element
float dequantize_q4_0(BlockQ4_0 block, int idx) {
    int byte_idx = idx / 2;
    int nibble_idx = idx & 1;
    int q = get_q4_0(block.qs[byte_idx], nibble_idx);
    return float(block.d) * float(q);
}

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
            float a_val = dequantize_q4_0(block, int(j));
            block_sum += a_val * b_val;
        }
        sum += block_sum;
    }
    
    // Reduction in shared memory
    tmp[tid] = sum;
    barrier();
    
    // Tree reduction (same as GGML)
    if (tid < 16) tmp[tid] += tmp[tid + 16];
    barrier();
    if (tid < 8) tmp[tid] += tmp[tid + 8];
    barrier();
    if (tid < 4) tmp[tid] += tmp[tid + 4];
    barrier();
    if (tid < 2) tmp[tid] += tmp[tid + 2];
    barrier();
    if (tid < 1) tmp[tid] += tmp[tid + 1];
    
    // Write result
    if (tid == 0) {
        data_d[row] = tmp[0];
    }
}
```

**Build script (compile GLSL to SPIR-V):**
```rust
// build.rs
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Compile shaders
    let shader_files = [
        "src/shaders/types.comp",
        "src/shaders/mul_mat_vec_q4_0_f32.comp",
        "src/shaders/rope.comp",
    ];
    
    for shader in shader_files {
        let output_path = Path::new(&out_dir).join(shader.split('/').last().unwrap());
        
        let shaderc = shaderc::Shaderc::new().unwrap();
        let compiled = shaderc.compile_into_spirv(
            shader,
            shaderc::ShaderKind::Vertex,
            "main",
            "vertex",
            None,
            None,
        ).unwrap();
        
        std::fs::write(&output_path, compiled.as_binary()).unwrap();
    }
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shaders/");
}
```

### Phase 3: Vulkan Device Context (Week 3)
**Study:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` lines 100-300 (device init)
- lines 500-800 (pipeline creation)  
- lines 2000-2200 (memory management)

**Your Implementation:**
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
    
    // Queues
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    
    // Command resources
    pub compute_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    
    // Safe memory allocator
    pub allocator: Arc<Mutex<Allocator>>,
    
    // Pipeline collection
    pub pipelines: Pipelines,
    
    // Device properties for optimization
    pub subgroup_size: u32,
    pub max_workgroup_size: [u32; 3],
}

impl VulkanContext {
    pub fn new(device_index: usize) -> Result<Self> {
        // Initialize Vulkan following GGML's pattern
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let physical_device = Self::select_physical_device(&instance, device_index)?;
        
        let (device, compute_queue, compute_queue_family) = 
            Self::create_device(&instance, physical_device)?;
            
        let compute_pool = Self::create_command_pool(&device, compute_queue_family)?;
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        
        let allocator = Self::create_allocator(&instance, &device, physical_device)?;
        
        let pipelines = Pipelines::new(&device)?;
        
        // Query device properties
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let subgroup_props = Self::query_subgroup_properties(&instance, physical_device)?;
        
        Ok(Self {
            instance,
            device,
            physical_device,
            compute_queue,
            compute_queue_family,
            compute_pool,
            descriptor_pool,
            allocator: Arc::new(Mutex::new(allocator)),
            pipelines,
            subgroup_size: subgroup_props.subgroup_size,
            max_workgroup_size: props.limits.max_compute_work_group_size,
        })
    }
    
    // Helper methods (follow GGML pattern, safe Rust)
    fn create_instance(entry: &Entry) -> Result<Instance> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name("Candle Vulkan Kernels")
            .application_version(vk::make_version(1, 0, 0))
            .api_version(vk::API_VERSION_1_3);
        
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);
        
        unsafe {
            Ok(entry.create_instance(&create_info, None)?)
        }
    }
    
    fn select_physical_device(instance: &Instance, index: usize) -> Result<vk::PhysicalDevice> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        
        devices.get(index)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Device {} not found", index))
    }
    
    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(Device, vk::Queue, u32)> {
        // Find compute queue (GGML pattern)
        let queue_families = unsafe { 
            instance.get_physical_device_queue_family_properties(physical_device) 
        };
        
        let compute_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or("No compute queue found")?;
        
        let queue_priority = 1.0;
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_family)
            .queue_priorities(&[queue_priority]);
        
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_create_info]);
        
        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)?
        };
        
        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };
        
        Ok((device, compute_queue, compute_family))
    }
    
    fn create_command_pool(device: &Device, queue_family: u32) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
            
        unsafe {
            Ok(device.create_command_pool(&pool_info, None)?)
        }
    }
    
    fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 256,
            },
        ];
        
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(128);
            
        unsafe {
            Ok(device.create_descriptor_pool(&pool_info, None)?)
        }
    }
    
    fn create_allocator(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Allocator> {
        let alloc_desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };
        
        Allocator::new(&alloc_desc)
    }
    
    fn query_subgroup_properties(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<vk::PhysicalDeviceSubgroupProperties> {
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut device_props2 = vk::PhysicalDeviceProperties2::builder();
        device_props2.push_next(&mut subgroup_props);
        
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut device_props2);
        }
        
        Ok(subgroup_props.build())
    }
}

// RAII cleanup (Rust idiom)
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.compute_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            // Allocator drops automatically
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
```

### Phase 4: Operation Dispatch (Week 4)
**Study:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` lines 1500-1700 (matmul dispatch)
- lines 1800-2000 (command buffer recording)

**Your Implementation:**
```rust
// candle-vulkan-kernels/src/ops/matmul.rs

use crate::device::VulkanContext;
use crate::buffer::VulkanBuffer;
use ash::vk;

pub struct MatMulParams {
    pub m: usize,  // rows of weights matrix
    pub n: usize,  // columns of weights matrix  
    pub k: usize,  // inner dimension (must be multiple of QK4_0)
}

pub fn mul_mat_vec_q4_0_f32(
    ctx: &VulkanContext,
    weights: &VulkanBuffer,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
    params: MatMulParams,
) -> Result<()> {
    // Validation (Rust can catch these at compile time where possible)
    assert_eq!(params.k % super::quant_types::QK4_0, 0, 
        "K dimension must be multiple of QK4_0 (32)");
    
    // RAII command buffer (automatically reset)
    let cmd_buffer = ctx.allocate_command_buffer()?;
    
    unsafe {
        // Begin command buffer (safe via ash builders)
        ctx.device.begin_command_buffer(
            cmd_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        
        // Allocate descriptor set (matches GGML pattern)
        let descriptor_set = ctx.allocate_descriptor_set(
            ctx.pipelines.mul_mat_vec_q4_0.descriptor_layout
        )?;
        
        // Update descriptors (safe slice API)
        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: weights.handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: input.handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: output.handle(),
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[0..1])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[1..2])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos[2..3])
                .build(),
        ];
        
        ctx.device.update_descriptor_sets(&writes, &[]);
        
        // Push constants (match shader layout exactly)
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        struct PushConstants {
            ncols_x: u32,
            nrows_x: u32,
            nrows_y: u32,
            nrows_dst: u32,
            row_stride_x: u32,
            channel_stride_x: u32,
        }
        
        let push_constants = PushConstants {
            ncols_x: params.k as u32,
            nrows_x: params.m as u32,
            nrows_y: 1,
            nrows_dst: params.m as u32,
            row_stride_x: params.k as u32,
            channel_stride_x: 0,
        };
        
        // Bind pipeline and dispatch (GGML pattern)
        ctx.device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            ctx.pipelines.mul_mat_vec_q4_0.pipeline,
        );
        
        ctx.device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            ctx.pipelines.mul_mat_vec_q4_0.layout,
            0,
            &[descriptor_set],
            &[],
        );
        
        ctx.device.cmd_push_constants(
            cmd_buffer,
            ctx.pipelines.mul_mat_vec_q4_0.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );
        
        // Dispatch (one workgroup per row, like GGML)
        ctx.device.cmd_dispatch(cmd_buffer, params.m as u32, 1, 1);
        
        ctx.device.end_command_buffer(cmd_buffer)?;
        
        // Submit and wait (simple for now, can be optimized later)
        ctx.submit_and_wait(&[cmd_buffer])?;
    }
    
    Ok(())
}
```

## File Structure

```
candle-vulkan-kernels/
├── Cargo.toml                     # Dependencies: ash, gpu-allocator, bytemuck
├── build.rs                       # Compile shaders to SPIR-V
├── src/
│   ├── lib.rs                     # Public API exports
│   ├── quant_types.rs             # Block structs (GGML logic, Rust types)
│   ├── device.rs                  # VulkanContext (safe RAII)
│   ├── buffer.rs                  # VulkanBuffer wrapper
│   ├── pipelines/
│   │   ├── mod.rs
│   │   ├── mul_mat_vec_q4_0.rs   # One file per pipeline
│   │   ├── mul_mat_q4_0.rs
│   │   └── rope.rs
│   └── ops/
│       ├── mod.rs
│       ├── matmul.rs              # Operation dispatch
│       └── rope.rs
└── tests/
    ├── correctness.rs             # Validate against GGML
    └── performance.rs             # Benchmark against CPU reference
```

## Testing Strategy

### Correctness Tests (Week 5)
```rust
#[test]
fn test_q4_0_quantization_roundtrip() {
    let input: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]; // Test vector
    
    // Quantize
    let mut blocks = vec![super::quant_types::BlockQ4_0::default(); input.len() / super::quant_types::QK4_0];
    super::quant_types::quantize_row_q4_0(&input, &mut blocks);
    
    // Dequantize
    let mut reconstructed = vec![0.0f32; input.len()];
    for (i, block) in blocks.iter().enumerate() {
        super::quant_types::dequantize_row_q4_0(block, &mut reconstructed[i * super::quant_types::QK4_0..]);
    }
    
    // Should be close but not exact (quantization loss)
    for (original, reconstructed) in input.iter().zip(&reconstructed) {
        let diff = (original - reconstructed).abs();
        assert!(diff < 0.01, "Quantization error too large: {}", diff);
    }
}

#[test]
fn test_q4_0_matmul_matches_cpu_reference() {
    let ctx = VulkanContext::new(0).unwrap();
    
    // Create test data (small for debugging)
    let weights = generate_q4_0_matrix(64, 64); // 64x64 matrix
    let input = generate_f32_vector(64);        // 64-element vector
    
    // CPU reference implementation (our safe version)
    let expected = matmul_q4_0_cpu_reference(&weights, &input);
    
    // Vulkan implementation
    let actual = mul_mat_vec_q4_0_f32(
        &ctx,
        &weights,
        &input,
        MatMulParams { m: 64, n: 64, k: 64 }
    ).unwrap();
    
    // Allow small numerical error (floating point precision)
    assert_vectors_close(&expected, &actual, 1e-3);
}
```

## Success Criteria

### ✅ You're doing it right if:
- Block sizes match GGML (18 bytes for Q4_0)
- Shaders are copied from GGML GLSL exactly
- Using `bytemuck::Pod` for GPU data, not raw pointers
- Using `gpu-allocator` for memory management
- Tests pass against CPU reference implementations
- Only unsafe code is for Vulkan API calls (no unnecessary unsafe)
- RAII cleanup via Drop trait instead of manual destruction

### ❌ Red flags:
- Writing new quantization algorithms instead of porting GGML
- Using `#[repr(C)]` and manual layout management
- Raw pointers where safe abstractions work
- Manual memory allocation (VMA, malloc, etc.)
- Generic abstractions over proven patterns
- Different workgroup sizes than GGML shaders
- Skipping validation tests

## Decision Points

### Why start with Q4_0?
- Most common quantization format in LLMs
- Best balance of compression ratio and accuracy
- GGML's most optimized kernel

### Why ash instead of vulkano?
- ash lets us mirror GGML's exact Vulkan patterns
- No abstraction hiding performance-critical details
- Industry standard for low-level Vulkan in Rust
- Works perfectly with gpu-allocator

### Why gpu-allocator instead of VMA?
- Safe Rust API (no raw handles)
- Better error handling with Result<T>
- Automatic cleanup via Drop trait
- Still has the same performance as VMA

## Quick Start for New Developers

1. **Read `quant_types.rs` first** - Understand the data structures
2. **Study `shaders/mul_mat_vec_q4_0_f32.comp`** - This is the critical kernel
3. **Run `cargo test`** - Make sure basic tests pass
4. **Profile with `tracing`** - Identify bottlenecks early
5. **Compare with GGML** - Use reference implementations to validate

## Glossary

- **Q4_0**: 4-bit quantization with 32 values per block, 18 bytes total
- **SPIR-V**: Vulkan's intermediate shader representation (portable)
- **Workgroup**: GPU compute unit (32x1x1 in our case)
- **Push constants**: Small data passed directly to shaders (faster than buffers)
- **Descriptor set**: Binding points for shader resources (buffers, textures)
```
