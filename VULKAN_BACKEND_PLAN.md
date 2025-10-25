# Candle Vulkan Backend - Implementation Plan

## Overview

This document outlines the implementation of a new Vulkan backend for the Candle ML framework, bringing the lessons learned from llama.cpp's successful Vulkan implementation to enable fast inference with exotic quantization formats.

## Goals

1. **Cross-Platform GPU Acceleration**: Run candle models on any GPU supporting Vulkan 1.2+ (NVIDIA, AMD, Intel, mobile GPUs)
2. **Quantization Support**: Enable fast inference with Q4_0, Q8_1, IQ2_S, IQ3_XXS, and other quantization formats
3. **Performance**: Match or exceed CUDA/Metal performance on respective hardware
4. **Compatibility**: Seamlessly integrate with existing candle architecture

---

## ✅ Phase 1: Foundation (COMPLETED)

### 1.1 candle-vulkan-kernels Crate

**Purpose**: Manage GLSL compute shaders and SPIR-V compilation

**Location**: `/candle-vulkan-kernels/`

**Files Created**:
```
candle-vulkan-kernels/
├── Cargo.toml              # Dependencies: ash, gpu-allocator, shaderc
├── build.rs                # GLSL → SPIR-V compiler at build time
├── src/
│   ├── lib.rs              # Shader loading API, ShaderOp enum
│   └── shaders/
│       ├── binary.comp     # Binary ops: add, sub, mul, div, min, max
│       ├── unary.comp      # Unary ops: exp, log, sin, cos, sqrt, abs, neg, sqr, gelu, relu, tanh
│       ├── reduce.comp     # Reductions: sum, min, max, argmin, argmax
│       └── matmul_f32.comp # Matrix multiplication (F32 baseline)
```

**Key Features**:
- Automatic SPIR-V compilation using `shaderc`
- Embedded shader binaries as `&'static [u8]`
- Zero-copy shader access at runtime
- `ShaderOp` enum for operation dispatch

**Shader Highlights**:
```glsl
// binary.comp - Strided binary operations
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {
    uint n_elements;
    uint op_type;  // 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
    uint stride_a;
    uint stride_b;
};

// reduce.comp - Shared memory parallel reduction
shared float shared_values[256];
shared uint shared_indices[256];
// Tree reduction with barrier synchronization
```

### 1.2 VulkanDevice Implementation

**Purpose**: Core device management and backend infrastructure

**Location**: `/candle-core/src/vulkan_backend/device.rs`

**Structure**:
```rust
pub struct VulkanDevice {
    id: DeviceId,                      // Unique device identifier
    entry: Arc<ash::Entry>,            // Vulkan entry point
    instance: Arc<ash::Instance>,      // Vulkan instance
    physical_device: vk::PhysicalDevice,
    device: Arc<ash::Device>,          // Logical device
    compute_queue: Arc<Mutex<vk::Queue>>,
    compute_queue_family: u32,
    command_pool: Arc<Mutex<vk::CommandPool>>,
    descriptor_pool: Arc<Mutex<vk::DescriptorPool>>,
    allocator: Arc<Mutex<Allocator>>,  // gpu-allocator
    pipeline_cache: Arc<RwLock<PipelineCache>>,
}
```

**Capabilities**:
- ✅ Device initialization with Vulkan 1.2+
- ✅ Compute queue and command pool setup
- ✅ Descriptor pool for resource binding
- ✅ Memory allocation (CPU-to-GPU, GPU-only, GPU-to-CPU)
- ✅ Buffer read/write operations
- ✅ Shader module caching
- ✅ Pipeline caching for compiled shaders
- ✅ Proper cleanup in Drop implementation

**Key Methods**:
```rust
impl VulkanDevice {
    pub fn new(ordinal: usize) -> Result<Self>
    pub fn alloc_buffer<T>(&self, len: usize, location: MemoryLocation) -> Result<Arc<VulkanBuffer<T>>>
    pub fn write_buffer<T>(&self, buffer: &VulkanBuffer<T>, data: &[T]) -> Result<()>
    pub fn read_buffer<T>(&self, buffer: &VulkanBuffer<T>) -> Result<Vec<T>>
    pub fn get_or_create_shader_module(&self, name: &str) -> Result<vk::ShaderModule>
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;
    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<VulkanStorage>
    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<VulkanStorage>
    // ... other BackendDevice methods
}
```

### 1.3 VulkanStorage Type System

**Purpose**: Type-safe GPU buffer management

**Location**: `/candle-core/src/vulkan_backend/mod.rs`

**Types**:
```rust
pub struct VulkanBuffer<T> {
    buffer: vk::Buffer,
    allocation: Allocation,
    len: usize,
    device: VulkanDevice,
    _phantom: PhantomData<T>,
}

pub enum VulkanStorageSlice {
    U8(Arc<VulkanBuffer<u8>>),
    U32(Arc<VulkanBuffer<u32>>),
    I64(Arc<VulkanBuffer<i64>>),
    BF16(Arc<VulkanBuffer<bf16>>),
    F16(Arc<VulkanBuffer<f16>>),
    F32(Arc<VulkanBuffer<f32>>),
    F64(Arc<VulkanBuffer<f64>>),
}

pub struct VulkanStorage {
    slice: VulkanStorageSlice,
    device: VulkanDevice,
}
```

**Supported DTypes**: U8, U32, I64, F16, BF16, F32, F64, F8E4M3

### 1.4 Integration with Candle Core

**Modified Files**:
- `candle-core/src/device.rs` - Added `Vulkan` variant to `Device` and `DeviceLocation` enums
- `candle-core/src/storage.rs` - Added `Vulkan` variant to `Storage` enum
- `candle-core/src/error.rs` - Added `NotCompiledWithVulkanSupport` and `Vulkan` error variants
- `candle-core/src/lib.rs` - Added `vulkan_backend` and `dummy_vulkan_backend` modules
- `Cargo.toml` - Added workspace dependency for `candle-vulkan-kernels`
- `candle-core/Cargo.toml` - Added `vulkan` feature flag

**Feature Flag**:
```toml
[features]
vulkan = [
    "dep:ash",
    "dep:gpu-allocator",
    "dep:candle-vulkan-kernels",
]
```

**Usage**:
```rust
// Enable in Cargo.toml
candle-core = { version = "0.9.2-alpha.1", features = ["vulkan"] }

// Create Vulkan device
let device = Device::new_vulkan(0)?;  // GPU 0
let tensor = Tensor::zeros((1024, 1024), DType::F32, &device)?;
```

### 1.5 Error Handling

**Location**: `/candle-core/src/vulkan_backend/error.rs`

```rust
pub enum VulkanError {
    VkError(ash::vk::Result),
    AllocationError(gpu_allocator::AllocationError),
    KernelError(candle_vulkan_kernels::VulkanError),
    UnsupportedDtype { dtype: DType, op: &'static str },
    Message(String),
    InternalError(String),
}

pub trait WrapErr<T> {
    fn w(self) -> Result<T, Error>;
}
```

---

## 🚧 Phase 2: Core Operations (TODO)

### 2.1 Implement BackendStorage Trait

**Location**: `/candle-core/src/vulkan_backend/mod.rs`

**Required Methods**:

```rust
impl BackendStorage for VulkanStorage {
    // ✅ Already implemented
    fn try_clone(&self, layout: &Layout) -> Result<Self>
    fn dtype(&self) -> DType
    fn device(&self) -> crate::Device
    fn to_cpu_storage(&self) -> Result<CpuStorage>

    // 🚧 TODO: Implement these
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self>
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self>
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self>

    // Binary operations
    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self>

    // Unary operations
    fn unary_impl<U: UnaryOpT>(
        &self,
        layout: &Layout,
    ) -> Result<Self>

    // Reductions
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self>

    // Comparisons
    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self>

    // Matrix operations
    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self>

    // Convolutions
    fn conv1d(/* ... */) -> Result<Self>
    fn conv2d(/* ... */) -> Result<Self>
    fn conv_transpose1d(/* ... */) -> Result<Self>
    fn conv_transpose2d(/* ... */) -> Result<Self>

    // Pooling
    fn avg_pool2d(/* ... */) -> Result<Self>
    fn max_pool2d(/* ... */) -> Result<Self>
    fn upsample_nearest1d(/* ... */) -> Result<Self>
    fn upsample_nearest2d(/* ... */) -> Result<Self>

    // Indexing
    fn gather(&self, layout: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self>
    fn scatter_add(/* ... */) -> Result<Self>
    fn index_select(&self, ids: &Self, layout: &Layout, ids_l: &Layout, dim: usize) -> Result<Self>
    fn index_add(/* ... */) -> Result<Self>

    // Memory operations
    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()>
    fn copy2d(/* ... */) -> Result<()>
}
```

**Implementation Strategy**:

1. **Start with Binary Operations** (`binary_impl`):
   - Create compute pipeline for `binary.comp` shader
   - Allocate descriptor sets
   - Build command buffer with push constants
   - Execute and synchronize

2. **Then Unary Operations** (`unary_impl`)
3. **Then Reductions** (`reduce_op`)
4. **Matrix Multiplication** (`matmul`) - Critical path
5. **Convolutions** - Can leverage im2col approach
6. **Indexing operations**
7. **Memory copy operations**

### 2.2 Command Buffer Execution

**Create Helper Methods**:

```rust
impl VulkanDevice {
    fn execute_kernel(
        &self,
        shader_name: &str,
        workgroup_count: (u32, u32, u32),
        descriptor_sets: &[vk::DescriptorSet],
        push_constants: &[u8],
    ) -> Result<()> {
        // 1. Get or create pipeline
        // 2. Allocate command buffer
        // 3. Bind pipeline and descriptor sets
        // 4. Push constants
        // 5. Dispatch compute
        // 6. Submit to queue
        // 7. Wait for completion
    }

    fn create_descriptor_set(
        &self,
        buffers: &[vk::Buffer],
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        // Allocate from descriptor pool
        // Update with buffer bindings
    }
}
```

### 2.3 Testing Infrastructure

**Create Tests**:

```rust
#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add() {
        let device = Device::new_vulkan(0).unwrap();
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).unwrap();
        let c = (&a + &b).unwrap();
        assert_eq!(c.to_vec1::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let device = Device::new_vulkan(0).unwrap();
        let a = Tensor::arange(0f32, 6f32, &device).unwrap().reshape((2, 3)).unwrap();
        let b = Tensor::arange(0f32, 12f32, &device).unwrap().reshape((3, 4)).unwrap();
        let c = a.matmul(&b).unwrap();
        // Verify result
    }
}
```

---

## 🎯 Phase 3: Quantization Support (TODO)

This is the key differentiator - bringing llama.cpp's quantization success to candle!

### 3.1 Quantization Formats

**Implement these quantization formats** (inspired by llama.cpp):

| Format | Bits | Description | Use Case |
|--------|------|-------------|----------|
| Q4_0   | 4.5  | 4-bit quantization, 32 values share single FP32 scale | General purpose |
| Q4_1   | 5.0  | 4-bit + 32 values share FP32 scale + FP32 min | Better accuracy |
| Q5_0   | 5.5  | 5-bit quantization | Higher quality |
| Q5_1   | 6.0  | 5-bit + scale + min | Best 5-bit |
| Q8_0   | 8.5  | 8-bit quantization | High accuracy |
| Q8_1   | 9.0  | 8-bit + scale + min | Highest accuracy quant |
| IQ2_XXS| ~2.0 | Importance matrix quantization | Extreme compression |
| IQ2_XS | ~2.3 | Importance quantization | Very small |
| IQ3_XXS| ~3.0 | 3-bit importance quantization | Small models |
| IQ3_S  | ~3.4 | 3-bit with better quality | Balanced |

### 3.2 Create Quantization Shaders

**New shader files**:

```
candle-vulkan-kernels/src/shaders/
├── quantize_q4_0.comp          # Quantize F32 → Q4_0
├── quantize_q8_0.comp          # Quantize F32 → Q8_0
├── dequantize_q4_0.comp        # Dequantize Q4_0 → F32
├── dequantize_q8_0.comp        # Dequantize Q8_0 → F32
├── matmul_q4_0_f32.comp        # Q4_0 × F32 matmul
├── matmul_q4_0_q8_0.comp       # Q4_0 × Q8_0 matmul (fast path)
├── matmul_q8_0_f32.comp        # Q8_0 × F32 matmul
├── matmul_iq2_xxs.comp         # IQ2_XXS matmul
└── matmul_iq3_xxs.comp         # IQ3_XXS matmul
```

**Example Q4_0 format**:

```glsl
// Q4_0: 32 weights packed into 16 bytes + 1 FP32 scale
struct block_q4_0 {
    float scale;        // 4 bytes
    uint8_t qs[16];     // 16 bytes, each holds 2 4-bit values
};  // Total: 20 bytes for 32 weights = 4.5 bits per weight

// Matmul kernel
#version 450
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer BlocksA { block_q4_0 blocks_a[]; };
layout(set = 0, binding = 1) readonly buffer BlocksB { block_q4_0 blocks_b[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float output_data[]; };

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    float sum = 0.0;
    for (uint k = 0; k < K / 32; k++) {
        block_q4_0 a_block = blocks_a[row * (K/32) + k];
        block_q4_0 b_block = blocks_b[k * N + col];

        // Compute dot product of quantized blocks
        float block_sum = 0.0;
        for (uint i = 0; i < 16; i++) {
            // Extract two 4-bit values from each byte
            int a0 = int(a_block.qs[i] & 0x0F) - 8;
            int a1 = int(a_block.qs[i] >> 4) - 8;
            int b0 = int(b_block.qs[i] & 0x0F) - 8;
            int b1 = int(b_block.qs[i] >> 4) - 8;

            block_sum += float(a0 * b0 + a1 * b1);
        }

        sum += a_block.scale * b_block.scale * block_sum;
    }

    output_data[row * N + col] = sum;
}
```

### 3.3 Add Quantized DTypes to Candle

**Extend DType enum**:

```rust
// In candle-core/src/dtype.rs
pub enum DType {
    // Existing types...
    U8, U32, I64, BF16, F16, F32, F64, F8E4M3,

    // New quantized types
    Q4_0,     // 4-bit quantization
    Q4_1,     // 4-bit with min
    Q5_0,     // 5-bit quantization
    Q5_1,     // 5-bit with min
    Q8_0,     // 8-bit quantization
    Q8_1,     // 8-bit with min
    IQ2_XXS,  // 2-bit importance quantization
    IQ2_XS,   // 2.3-bit importance quantization
    IQ3_XXS,  // 3-bit importance quantization
    IQ3_S,    // 3.4-bit importance quantization
}
```

**Add quantization methods**:

```rust
impl Tensor {
    pub fn quantize(&self, dtype: DType) -> Result<Tensor> {
        // Quantize F32 tensor to quantized format
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        // Dequantize to F32
    }
}
```

### 3.4 Optimize Matrix Multiplication

**Key optimizations from llama.cpp**:

1. **Cooperative Matrix Operations** (Vulkan 1.3):
```glsl
#version 450
#extension GL_KHR_cooperative_matrix : enable

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;

void main() {
    // Use hardware-accelerated matrix ops
    coopMatLoad(matA, a_ptr, stride, 0);
    coopMatLoad(matB, b_ptr, stride, 0);
    matC = coopMatMulAdd(matA, matB, matC);
    coopMatStore(matC, c_ptr, stride, 0);
}
```

2. **Tiled Matrix Multiplication**:
```glsl
// Use shared memory for tiling
shared float tileA[TILE_SIZE][TILE_SIZE];
shared float tileB[TILE_SIZE][TILE_SIZE];

// Load tiles, compute, accumulate
```

3. **Integer Dot Product** (VK_KHR_shader_integer_dot_product):
```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable

// Fast 4x8-bit dot product
int dot = int(dot4AddSaturateU8Packed(a_packed, b_packed, acc));
```

---

## 🔧 Phase 4: Optimization & Features (TODO)

### 4.1 Performance Optimizations

**Workgroup Size Tuning**:
- Profile different workgroup sizes for each operation
- AMD: 64 threads/workgroup optimal
- NVIDIA: 256-1024 threads/workgroup
- Intel: 128-256 threads/workgroup

**Memory Access Patterns**:
- Coalesce memory accesses
- Use shared memory for frequently accessed data
- Minimize bank conflicts

**Pipeline Barriers**:
- Minimize synchronization points
- Batch operations when possible
- Use async compute queues

### 4.2 Multi-GPU Support

```rust
impl VulkanDevice {
    pub fn device_count() -> Result<usize>
    pub fn new_multi_gpu(ordinals: &[usize]) -> Result<Vec<Self>>
}

// Tensor sharding across GPUs
impl Tensor {
    pub fn shard(&self, devices: &[Device]) -> Result<Vec<Tensor>>
    pub fn gather(shards: &[Tensor], device: &Device) -> Result<Tensor>
}
```

### 4.3 Async Operations

```rust
impl VulkanDevice {
    pub fn execute_async(
        &self,
        ops: Vec<Operation>,
    ) -> Result<AsyncHandle> {
        // Submit work without blocking
    }
}
```

### 4.4 Flash Attention Support

**Create flash attention kernel**:
```glsl
// Flash Attention with quantized KV cache
// From "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
```

### 4.5 Custom Operations

**Allow users to inject custom GLSL shaders**:
```rust
impl VulkanDevice {
    pub fn register_custom_shader(
        &self,
        name: &str,
        spirv: &[u8],
    ) -> Result<()>

    pub fn execute_custom_op(
        &self,
        shader_name: &str,
        inputs: &[&Tensor],
        output_shape: &Shape,
    ) -> Result<Tensor>
}
```

---

## 📊 Phase 5: Validation & Benchmarking (TODO)

### 5.1 Create Examples

**Example 1**: Basic Vulkan Usage
```rust
// candle-examples/examples/vulkan_basics.rs
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;

    // Create tensors
    let a = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;
    let b = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;

    // Operations
    let c = a.matmul(&b)?;
    let d = c.relu()?;

    println!("Result: {:?}", d.to_vec2::<f32>()?);
    Ok(())
}
```

**Example 2**: Quantized Inference
```rust
// candle-examples/examples/vulkan_quantized.rs
use candle_core::{Device, Tensor, DType};

fn main() -> Result<()> {
    let device = Device::new_vulkan(0)?;

    // Load model weights in F32
    let weights = Tensor::randn(0f32, 1f32, (4096, 4096), &device)?;

    // Quantize to Q4_0 (4.5 bits per weight)
    let weights_q4 = weights.quantize(DType::Q4_0)?;

    // Inference with quantized weights
    let input = Tensor::randn(0f32, 1f32, (1, 4096), &device)?;
    let output = input.matmul(&weights_q4)?;

    println!("Memory saved: {:.2}x", 32.0 / 4.5);
    Ok(())
}
```

**Example 3**: Model Inference (LLaMA)
```rust
// candle-examples/examples/vulkan_llama.rs
// Run LLaMA model on Vulkan backend with Q4 quantization
```

### 5.2 Benchmarks

**Create benchmark suite**:

```rust
// candle-core/benches/vulkan_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use candle_core::{Device, Tensor};

fn bench_matmul_f32(c: &mut Criterion) {
    let device = Device::new_vulkan(0).unwrap();
    let sizes = vec![256, 512, 1024, 2048, 4096];

    for size in sizes {
        c.bench_function(&format!("vulkan_matmul_f32_{}", size), |b| {
            let a = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();
            let b = Tensor::randn(0f32, 1f32, (size, size), &device).unwrap();

            b.iter(|| {
                black_box(a.matmul(&b).unwrap());
            });
        });
    }
}

fn bench_matmul_q4_0(c: &mut Criterion) {
    // Benchmark quantized matmul
}

criterion_group!(benches, bench_matmul_f32, bench_matmul_q4_0);
criterion_main!(benches);
```

**Compare against**:
- CPU backend (gemm)
- CUDA backend (cuBLAS)
- Metal backend (MLX)
- llama.cpp Vulkan

### 5.3 Correctness Tests

**Numerical accuracy tests**:
```rust
#[test]
fn test_vulkan_cpu_parity() {
    let cpu = Device::Cpu;
    let vulkan = Device::new_vulkan(0).unwrap();

    let a_cpu = Tensor::randn(0f32, 1f32, (128, 128), &cpu).unwrap();
    let b_cpu = Tensor::randn(0f32, 1f32, (128, 128), &cpu).unwrap();

    let a_vk = a_cpu.to_device(&vulkan).unwrap();
    let b_vk = b_cpu.to_device(&vulkan).unwrap();

    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();
    let c_vk = a_vk.matmul(&b_vk).unwrap().to_device(&cpu).unwrap();

    assert_close(&c_cpu.to_vec1::<f32>().unwrap(),
                 &c_vk.to_vec1::<f32>().unwrap(),
                 1e-5);
}
```

---

## 📝 Phase 6: Documentation & Polish (TODO)

### 6.1 Documentation

**Add rustdoc comments**:
```rust
/// Vulkan backend for Candle ML framework
///
/// Provides cross-platform GPU acceleration using Vulkan compute shaders.
/// Supports quantized operations for efficient inference.
///
/// # Examples
///
/// ```no_run
/// use candle_core::{Device, Tensor};
///
/// let device = Device::new_vulkan(0)?;
/// let tensor = Tensor::zeros((1024, 1024), DType::F32, &device)?;
/// ```
pub struct VulkanDevice { /* ... */ }
```

**Create user guide**:
```markdown
# Vulkan Backend Guide

## Installation

### Requirements
- Vulkan SDK 1.2+
- Compatible GPU (NVIDIA, AMD, Intel, etc.)

### Build
```bash
cargo build --features vulkan
```

## Usage

### Basic Operations
...

### Quantization
...

### Advanced Features
...
```

### 6.2 Error Messages

**Improve error messages**:
```rust
if vulkan_devices.is_empty() {
    return Err(VulkanError::Message(
        "No Vulkan-compatible devices found. \
         Please ensure:\n\
         1. Vulkan drivers are installed\n\
         2. GPU supports Vulkan 1.2+\n\
         3. Vulkan SDK is properly configured"
            .to_string()
    ));
}
```

### 6.3 CI/CD Integration

**Add to GitHub Actions**:
```yaml
# .github/workflows/vulkan.yml
name: Vulkan Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Vulkan SDK
        run: |
          wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
          sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
            https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
          sudo apt update
          sudo apt install vulkan-sdk

      - name: Run tests
        run: cargo test --features vulkan

      - name: Run benchmarks
        run: cargo bench --features vulkan
```

---

## 🎯 Success Metrics

### Performance Targets

| Operation | Target vs CPU | Target vs CUDA |
|-----------|---------------|----------------|
| MatMul F32 (4096x4096) | 50-100x faster | 0.8-1.2x |
| MatMul Q4_0 (4096x4096) | 100-200x faster | 1.0-1.5x (vs FP16) |
| Element-wise ops | 20-50x faster | 0.9-1.1x |
| Reductions | 30-60x faster | 0.9-1.1x |

### Memory Targets

| Quantization | Bits/Weight | Memory Reduction | Target Accuracy Loss |
|--------------|-------------|------------------|---------------------|
| Q4_0 | 4.5 | 7.1x | <2% |
| Q8_0 | 8.5 | 3.8x | <0.5% |
| IQ2_XXS | ~2.0 | 16x | <5% |

### Compatibility Targets

- ✅ NVIDIA GPUs (Pascal+)
- ✅ AMD GPUs (GCN 4+, RDNA)
- ✅ Intel GPUs (Gen 9+)
- ✅ Mobile GPUs (Adreno, Mali)
- ✅ Linux, Windows, macOS

---

## 🚀 Quick Start (For Developers)

### Building with Vulkan Backend

```bash
# Clone repository
git clone https://github.com/huggingface/candle
cd candle

# Checkout Vulkan branch
git checkout claude/new-candle-backend-011CUT32sfK2x7xNUwTQqEmb

# Install Vulkan SDK (Ubuntu/Debian)
sudo apt install vulkan-sdk

# Build with Vulkan support
cargo build --features vulkan

# Run tests
cargo test --features vulkan

# Run example
cargo run --example vulkan_basics --features vulkan
```

### Project Structure

```
candle/
├── candle-vulkan-kernels/       # GLSL shaders + SPIR-V compiler
│   ├── src/
│   │   ├── lib.rs               # Shader loading API
│   │   └── shaders/             # GLSL compute shaders
│   └── build.rs                 # Shader compilation
│
├── candle-core/
│   └── src/
│       ├── vulkan_backend/      # Vulkan backend implementation
│       │   ├── mod.rs
│       │   ├── device.rs        # VulkanDevice
│       │   ├── error.rs
│       │   └── utils.rs
│       ├── dummy_vulkan_backend.rs  # Feature-gated stub
│       ├── device.rs            # Device enum (+ Vulkan variant)
│       ├── storage.rs           # Storage enum (+ Vulkan variant)
│       └── error.rs             # Error types (+ Vulkan errors)
│
└── candle-examples/
    └── examples/
        ├── vulkan_basics.rs     # TODO
        └── vulkan_quantized.rs  # TODO
```

---

## 📚 References

### llama.cpp Vulkan Implementation
- [ggml-vulkan.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- [Vulkan shaders](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan/vulkan-shaders)
- [Quantization formats](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h)

### Vulkan Resources
- [Vulkan Compute Primer](https://www.khronos.org/blog/vulkan-compute-just-the-basics)
- [Cooperative Matrix Extension](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_KHR_cooperative_matrix.adoc)
- [Vulkan Best Practices](https://arm-software.github.io/vulkan_best_practice_for_mobile_developers/)

### Candle Architecture
- [Candle Backend Trait](https://github.com/huggingface/candle/blob/main/candle-core/src/backend.rs)
- [CUDA Backend](https://github.com/huggingface/candle/tree/main/candle-core/src/cuda_backend)
- [Metal Backend](https://github.com/huggingface/candle/tree/main/candle-core/src/metal_backend)

---

## 🤝 Contributing

### Next Steps for Contributors

**Easy Tasks** (Good First Issues):
1. Add more unary operations (sigmoid, silu, swish)
2. Implement comparison operations (eq, ne, lt, le, gt, ge)
3. Add casting between dtypes
4. Write documentation and examples

**Medium Tasks**:
1. Implement convolution operations
2. Add pooling operations
3. Implement indexing operations (gather, scatter, index_select)
4. Optimize workgroup sizes per GPU architecture

**Hard Tasks**:
1. Implement quantized matrix multiplication (Q4_0, Q8_0)
2. Add Flash Attention kernel
3. Implement multi-GPU support
4. Add cooperative matrix operations (Vulkan 1.3)

### Testing Checklist

Before submitting PR:
- [ ] Code compiles with `--features vulkan`
- [ ] All tests pass: `cargo test --features vulkan`
- [ ] Benchmarks run: `cargo bench --features vulkan`
- [ ] No clippy warnings: `cargo clippy --features vulkan`
- [ ] Formatted: `cargo fmt`
- [ ] Documentation builds: `cargo doc --features vulkan`
- [ ] Example runs: `cargo run --example vulkan_basics --features vulkan`

---

## 📄 License

This implementation follows Candle's dual license:
- MIT OR Apache-2.0

---

**Status**: Phase 1 Complete ✅ | Phase 2-6 In Progress 🚧

**Last Updated**: 2025-10-25

**Branch**: `claude/new-candle-backend-011CUT32sfK2x7xNUwTQqEmb`
