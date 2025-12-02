# Candle Vulkan Backend Development Roadmap

## Current Status: Foundation Complete

**What we've built:**
- Full Vulkan context, device, and memory management
- Shader compilation pipeline with `#include` preprocessing
- Strided tensor support matching ggml-vulkan's approach
- Broadcasting for binary ops
- Fastdiv optimization for GPU index computation

**Implemented Operations:**
- Unary: `exp`, `silu`, `gelu`, `relu` (with strided variants)
- Binary: `add`, `mul`, `div` (with broadcast support)
- Infrastructure: `zeros`, `to_cpu`, `from_cpu`

**Tests:** 20/20 passing

---

## Phase 1: Quantization (Priority: CRITICAL)

This is the highest priority. On shared-memory systems like AMD Ryzen AI Max+ 395:
- Memory bandwidth is the bottleneck (not PCIe)
- Q4 is 8x smaller than F32 = 8x more cache efficiency
- Most models are distributed as GGUF quantized weights
- ggml-vulkan has 35+ quantization shaders ready to port

### 1.1 Quantized Storage Infrastructure
- [ ] Create `QVulkanStorage` struct (like `QMetalStorage`)
- [ ] Add `QStorage::Vulkan` variant to the enum
- [ ] Implement `Device::Vulkan` branch in `qzeros()`
- [ ] Wire up `load_quantized()` for Vulkan

### 1.2 Dequantization Shaders (for fallback path)
- [ ] Port `dequant_q4_0.comp` - most common format
- [ ] Port `dequant_q8_0.comp` - higher quality common format
- [ ] Port `dequant_q4_k.comp` through `dequant_q6_k.comp` - k-quants
- [ ] Implement `QVulkanStorage::dequantize()` 

### 1.3 Fused Quantized MatMul (the real win)
- [ ] Port `mul_mat_vec_q4_0.comp` - no dequant step needed!
- [ ] Port `mul_mat_vec_q8_0.comp`
- [ ] Port k-quant variants: `mul_mat_vec_q4_k.comp`, etc.
- [ ] Implement `QVulkanStorage::fwd()` for QMatMul
- [ ] Wire up `QTensor::vulkan_fwd()` in CustomOp1

### 1.4 Quantized Embedding Lookup
- [ ] Port `get_rows_quant.comp` - for embedding tables

**Available ggml-vulkan shaders:**
```
dequant_q4_0.comp, dequant_q4_1.comp, dequant_q5_0.comp, dequant_q5_1.comp
dequant_q8_0.comp, dequant_q2_k.comp, dequant_q3_k.comp, dequant_q4_k.comp
dequant_q5_k.comp, dequant_q6_k.comp, dequant_q8_k.comp
mul_mat_vec_q2_k.comp, mul_mat_vec_q3_k.comp, mul_mat_vec_q4_k.comp
mul_mat_vec_q5_k.comp, mul_mat_vec_q6_k.comp
get_rows_quant.comp, copy_to_quant.comp, copy_from_quant.comp
+ imatrix quants (iq1, iq2, iq3, iq4) for even more compression
```

---

## Phase 2: Core Transformer Ops (Priority: Critical)

These are blocking for running any real model.

### 1.1 MatMul / GEMM
- [ ] Wire up existing `mul_mat_vec.comp` for matvec
- [ ] Port/adapt matrix-matrix multiply from ggml-vulkan
- [ ] Support batched matmul for attention
- **Why:** Every layer needs this

### 1.2 Softmax
- [ ] Wire up existing `soft_max.comp`
- [ ] Ensure numerical stability (max subtraction)
- **Why:** Attention mechanism core

### 1.3 RMS Norm / Layer Norm
- [ ] Wire up existing `rms_norm.comp`
- [ ] Add layer_norm variant
- **Why:** Used in every transformer block

### 1.4 Reduce Ops
- [ ] Sum, mean, max, min along dimensions
- [ ] Argmax/argmin for sampling
- **Why:** Pooling, loss computation, token selection

---

## Phase 3: Memory & Data Movement (Priority: High)

### 2.1 Copy/Cast Operations
- [ ] Wire up `copy.comp`
- [ ] `to_dtype` for F32 <-> F16 <-> BF16
- [ ] `copy_strided_src` for view materialization
- **Why:** Mixed precision, efficient memory layout

### 2.2 Indexing Operations
- [ ] `index_select` - embedding lookups
- [ ] `gather` / `scatter` - advanced indexing
- [ ] `where_cond` - conditional selection
- **Why:** Embeddings, sparse ops, masking

### 2.3 Affine Transform
- [ ] Wire up `scale.comp` 
- [ ] Implement general `affine` (mul + add)
- **Why:** Scaling, bias addition

---

## Phase 4: Expanded Unary/Binary Ops (Priority: Medium)

### 3.1 More Unary Ops
- [ ] `sqrt`, `sin`, `cos` (shaders exist)
- [ ] `tanh`, `sigmoid`
- [ ] `neg`, `abs`, `sign`
- [ ] `log`, `log2`, `exp2`
- [ ] `clamp` (shader exists)
- [ ] `powf`, `elu`

### 3.2 More Binary Ops
- [ ] `sub` (trivial - just negate and add, or new shader)
- [ ] `min`, `max` (element-wise)
- [ ] `cmp` ops (eq, ne, lt, le, gt, ge)

---

## Phase 5: Convolution & Pooling (Priority: Medium-Low)

### 4.1 Convolutions
- [ ] `conv1d`, `conv2d`
- [ ] `conv_transpose1d`, `conv_transpose2d`
- [ ] Check if ggml-vulkan has these or need custom

### 4.2 Pooling
- [ ] `avg_pool2d`, `max_pool2d`
- [ ] `upsample_nearest1d`, `upsample_nearest2d`

---

## Phase 6: Advanced/Specialized (Priority: Low initially)

### 5.1 Attention
- [ ] Scaled dot-product attention kernel
- [ ] Flash attention variant if feasible
- [ ] KV cache support

### 5.2 Quantization
- [ ] Q4_0, Q8_0 support (ggml has these)
- [ ] Mixed precision matmul

### 5.3 Custom Optimizers
- [ ] Muon optimizer kernels (if not in ggml-vulkan)
- [ ] Fused Adam/AdamW

### 5.4 Random Number Generation
- [ ] `rand_uniform`, `rand_normal`
- [ ] Proper GPU RNG state

---

## Resource Mapping

### Shaders we have (ggml-derived):
```
add.comp, mul.comp, div.comp          -> binary ops (done)
exp.comp, silu.comp, gelu.comp, relu.comp -> unary ops (done)
*_strided.comp variants               -> strided support (done)
sqrt.comp, sin.comp, cos.comp, clamp.comp -> easy unary adds
scale.comp                            -> affine
soft_max.comp                         -> softmax
rms_norm.comp                         -> normalization
mul_mat_vec.comp                      -> matvec
copy.comp                             -> data movement
```

### Need to port from ggml-vulkan or write:
- Full GEMM (matrix-matrix)
- Reduce operations
- Indexing operations
- Convolutions
- Attention kernels

### May need custom (not in ggml-vulkan):
- Muon optimizer
- Some candle-specific ops
- Flash attention optimizations

---

## Estimated Effort

| Phase | Ops | Effort | Impact |
|-------|-----|--------|--------|
| **Phase 1** | **Quantization** | 2-3 sessions | **Unlocks GGUF models, 8x memory efficiency** |
| Phase 2 | MatMul, Softmax, Norm, Reduce | 2-3 sessions | Unlocks transformer inference |
| Phase 3 | Copy, Index, Affine | 1-2 sessions | Unlocks embeddings, mixed precision |
| Phase 4 | More unary/binary | 1 session | Completeness |
| Phase 5 | Conv, Pool | 1-2 sessions | CNN support |
| Phase 6 | Attention optimizations | 2-4 sessions | Performance |

---

## Testing Strategy

Each phase should include:
1. Unit tests for each op (like our current 20 tests)
2. Comparison with CPU reference
3. Strided/broadcast variants where applicable

**Milestones:**
- End of Phase 1: Load a GGUF Q4_0 model and run dequantized inference
- End of Phase 2: Run full transformer inference (GPT-2 / Llama small)
- End of Phase 3: Mixed precision workflows

---

## Notes

- **Shared memory advantage:** On AMD Ryzen AI Max+ 395, we skip PCIe transfers entirely
- **Portability:** Everything works on any Vulkan GPU (laptop tested, desktop ready)
- **ggml-vulkan alignment:** Following their patterns means we can port more shaders easily

---

*Last updated: December 2025*
*Status: Foundation complete, ready for Phase 1*
