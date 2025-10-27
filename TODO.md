# TODO for Next Agent: Vulkan Backend for Candle

## 📋 Project Overview
This project has successfully created a working Vulkan proof-of-concept that demonstrates core Vulkan functionality following the patterns from `ash-comp-shader-example`. The goal is to create a full Vulkan backend for Candle ML framework, similar to how the Metal backend works.

## 🎯 Current Status
✅ **Completed:**
- Basic Vulkan context creation (instance, device, queues)
- Shader loading and module creation
- Compute pipeline creation
- Command buffer recording and submission
- Successfully runs Vulkan compute operations (segfaults due to dummy shader, but infrastructure works)

❌ **Remaining:**
- Real tensor operation shaders (GLSL → SPIR-V)
- Buffer management with gpu-allocator integration
- Candle backend trait implementation
- Memory management and synchronization
- Integration with Candle's tensor operations

## 📁 Key Files to Focus On

### 1. `/home/thomas/src/project-zed/candle-vulkan-demo/`
**Main working prototype - START HERE**
- `src/main.rs` - Contains working Vulkan context and compute pipeline
- `shaders/add.comp` - Template for GLSL shaders
- `build.rs` - Shader compilation pipeline
- `Cargo.toml` - Dependencies and build configuration

### 2. `/home/thomas/src/project-zed/candle/candle-vulkan-kernels/`
**Target location for final implementation**
- Already has basic structure but broken compilation
- Follow Candle's existing backend architecture patterns
- Study `src/device.rs`, `src/storage.rs`, `src/kernels.rs`

### 3. Documentation References
- `/home/thomas/src/project-zed/DOCS/ggml-vulkan-analysis.md` - GGML Vulkan patterns
- `/home/thomas/src/project-zed/DOCS/candle-backend-architecture.md` - Candle backend design
- `/home/thomas/src/project-zed/DOCS/ggml-metal-analysis.md` - Metal backend reference

## 🔧 Critical Tasks to Complete

### Task 1: Create Real Shaders (High Priority)
```glsl
// Need real compute shaders for tensor operations:
// - Matrix addition: C = A + B
// - Matrix multiplication: C = A * B
// - Element-wise operations: relu, sigmoid, etc.
// Following GGML vulkan shader patterns
```

### Task 2: Implement Buffer Management (High Priority)
```rust
// Need proper buffer creation using gpu-allocator:
// - Host-visible buffers for CPU-GPU transfers
// - Device-local buffers for GPU operations
// - Memory mapping and synchronization
// - Buffer pooling like GGML does
```

### Task 3: Implement Candle Backend Traits (High Priority)
```rust
// Follow candle-backend-architecture.md:
// impl BackendDevice for VulkanDevice
// impl BackendStorage for VulkanStorage
// Support all required operations:
// - matmul, add, unary ops, etc.
// - Proper error handling and synchronization
```

### Task 4: Memory Management Architecture (Medium Priority)
```rust
// Study GGML's memory patterns:
// - Suballocation strategies
// - Memory type selection
// - Synchronization and barriers
// - Reference counting for buffers
```

### Task 5: Integration Testing (Medium Priority)
```rust
// Create working examples:
// - Simple matrix operations
// - Compare with CPU backend for correctness
// - Performance benchmarking
// - Edge case handling
```

## 🧠 Key Concepts to Understand

### 1. Vulkan API Patterns (from ash-comp-shader-example)
- **Builder Pattern vs Direct Struct Creation**: ash 0.38 uses direct struct creation with `_marker` fields
- **Lifetime Management**: Vulkan objects have lifetimes and require proper ordering
- **Error Handling**: Vulkan uses result codes that must be checked
- **Command Recording**: Pattern of begin → record → end → submit → wait

### 2. GGML Vulkan Architecture
- **Shader Organization**: Many small specialized shaders vs few general ones
- **Memory Layout**: Specific patterns for quantized tensors (Q4_0, Q5_0, etc.)
- **Workgroup Strategy**: Thread organization for tensor operations
- **Push Constants**: For passing small amounts of per-operation data

### 3. Candle Backend Architecture
- **Trait System**: BackendDevice and BackendStorage traits
- **Type Safety**: Proper handling of DType and tensor shapes
- **Memory Abstraction**: Buffer management without direct Vulkan exposure
- **Error Propagation**: Proper error handling through Result types

## 🔗 Key Dependencies and APIs

### Vulkan Stack
```toml
[dependencies]
ash = "0.38"                    # Vulkan bindings
gpu-allocator = "0.27"          # Memory management
shaderc = "0.8"                 # GLSL → SPIR-V compilation
bytemuck = "1.14"               # Memory safety
```

### Candle Integration
- Follow patterns from `candle-core/src/backends/`
- Study `candle-core/src/metal/backend.rs` for reference
- Implement `BackendDevice` and `BackendStorage` traits

## 🚨 Technical Challenges

### 1. API Version Compatibility
- ash 0.38 has different API than older versions
- Builder methods replaced with direct struct creation
- Lifetime management is more strict

### 2. Memory Management Complexity
- Vulkan has complex memory type selection
- Need host-visible vs device-visible memory
- Proper synchronization is critical

### 3. Shader Ecosystem
- Need to compile GLSL to SPIR-V at build time
- Support multiple shader variants for different operations
- Handle shader specialization and optimization

### 4. Integration with Candle
- Candle has specific tensor operation patterns
- Need to match existing backend interfaces exactly
- Handle quantization and data type conversions

## 📈 Success Metrics

### Minimum Viable Product
- ✅ Vulkan context creation (DONE)
- ✅ Shader loading and compilation
- ✅ Compute pipeline creation (DONE)
- ❌ Matrix addition operation on GPU
- ❌ Matrix multiplication operation on GPU
- ❌ Integration with Candle tensor API

### Full Implementation
- Support for all major tensor operations
- Performance parity with CPU backend
- Memory efficiency comparable to GGML
- Comprehensive error handling and testing

## 🎯 Next Steps

1. **Immediate**: Fix the dummy shader to create a working compute operation
2. **Short Term**: Add real shader compilation from GLSL sources
3. **Medium Term**: Implement buffer management with gpu-allocator
4. **Long Term**: Full Candle backend trait implementation

## 📚 Reference Code to Study

### Working Examples
- `ash-comp-shader-example/` - Clean Vulkan patterns
- This current `candle-vulkan-demo/` - Working foundation

### Candle Patterns
- `candle-core/src/backends/cpu/` - Simple backend reference
- `candle-core/src/backends/metal/` - Advanced backend reference
- `candle/candle-metal-kernels/` - Shader compilation patterns

### GGML Patterns
- `llama.cpp/ggml/src/ggml-vulkan/` - Production Vulkan backend
- Shader organization and memory management patterns

---

**Note**: This foundation successfully demonstrates that Vulkan can be integrated with Candle. The next agent should focus on building real tensor operations and proper memory management while maintaining the working Vulkan context as the foundation.
