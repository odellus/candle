//! Flash Attention for AMD GPUs (HIP/ROCm)
//!
//! This crate provides flash attention operations for AMD GPUs using HIP/ROCm.
//! It provides a similar API to candle-flash-attn for CUDA.

use candle::backend::{BackendDevice, BackendStorage};
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

mod ffi {
    use std::ffi::c_void;

    extern "C" {
        pub fn flash_attn_fwd(
            stream: *mut c_void,
            q_ptr: *const c_void,
            k_ptr: *const c_void,
            v_ptr: *const c_void,
            out_ptr: *mut c_void,
            softmax_lse_ptr: *mut c_void,
            alibi_slopes_ptr: *const c_void,
            batch_size: i32,
            seqlen_q: i32,
            seqlen_k: i32,
            num_heads: i32,
            num_heads_k: i32,
            head_dim: i32,
            q_batch_stride: i32,
            q_row_stride: i32,
            q_head_stride: i32,
            k_batch_stride: i32,
            k_row_stride: i32,
            k_head_stride: i32,
            v_batch_stride: i32,
            v_row_stride: i32,
            v_head_stride: i32,
            out_batch_stride: i32,
            out_row_stride: i32,
            out_head_stride: i32,
            softmax_scale: f32,
            softcap: f32,
            is_causal: i32,
            window_size_left: i32,
            window_size_right: i32,
            is_bf16: i32,
        ) -> i32;
    }
}

/// Flash Attention configuration
pub struct FlashAttn {
    /// Softmax scaling factor (typically 1/sqrt(head_dim))
    pub softmax_scale: f32,

    /// Optional ALiBi slopes for position encoding
    pub alibi_slopes: Option<Tensor>,

    /// Left window size for local attention (-1 for unlimited)
    pub window_size_left: Option<usize>,

    /// Right window size for local attention (-1 for unlimited)
    pub window_size_right: Option<usize>,

    /// Softcap value for capping attention scores (None for no capping)
    pub softcap: Option<f32>,
}

impl FlashAttn {
    /// Create a new FlashAttn with default settings
    pub fn new(softmax_scale: f32) -> Self {
        Self {
            softmax_scale,
            alibi_slopes: None,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        }
    }

    /// Set ALiBi slopes
    pub fn with_alibi_slopes(mut self, alibi_slopes: Tensor) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
        self
    }

    /// Set window sizes for local attention
    pub fn with_window_size(mut self, left: Option<usize>, right: Option<usize>) -> Self {
        self.window_size_left = left;
        self.window_size_right = right;
        self
    }

    /// Set softcap value
    pub fn with_softcap(mut self, softcap: f32) -> Self {
        self.softcap = Some(softcap);
        self
    }

    fn hip_fwd_t<T: candle::WithDType>(
        &self,
        q: &candle::HipStorage,
        q_l: &Layout,
        k: &candle::HipStorage,
        k_l: &Layout,
        v: &candle::HipStorage,
        v_l: &Layout,
        is_bf16: bool,
        is_causal: bool,
    ) -> Result<(candle::HipStorage, Shape)> {
        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank})"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size) = k_l.shape().dims4()?;

        if head_size > 128 {
            candle::bail!("only supports head dimension at most 128 (got {head_size})")
        }
        if head_size % 8 != 0 {
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let out_l = Layout::contiguous(&out_shape);
        let out_stride = out_l.stride();

        // Get alibi slopes pointer if specified
        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }
            let (storage, layout) = alibi_slopes.storage_and_layout();
            match &*storage {
                candle::Storage::Hip(hip_storage) => {
                    hip_storage.as_hip_ptr()? as *const std::ffi::c_void
                }
                _ => candle::bail!("alibi_slopes must be on HIP device"),
            }
        } else {
            std::ptr::null()
        };

        // Allocate output tensor
        let dst = unsafe { dev.alloc_uninit(&out_shape, q.dtype())? };
        let dst_ptr = dst.as_hip_ptr()? as *mut std::ffi::c_void;

        // Get source pointers
        let q_ptr = q.as_hip_ptr()? as *const std::ffi::c_void;
        let k_ptr = k.as_hip_ptr()? as *const std::ffi::c_void;
        let v_ptr = v.as_hip_ptr()? as *const std::ffi::c_void;

        let window_size_left = self.window_size_left.map(|x| x as i32).unwrap_or(-1);
        let window_size_right = self.window_size_right.map(|x| x as i32).unwrap_or(-1);
        let softcap = self.softcap.unwrap_or(0.0);

        let stream = dev.stream().as_ptr();

        let result = unsafe {
            ffi::flash_attn_fwd(
                stream,
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                std::ptr::null_mut(), // softmax_lse (optional)
                alibi_slopes_ptr,
                b_sz as i32,
                seqlen_q as i32,
                seqlen_k as i32,
                num_heads as i32,
                num_heads_k as i32,
                head_size as i32,
                q_stride[0] as i32,
                q_stride[1] as i32,
                q_stride[2] as i32,
                k_stride[0] as i32,
                k_stride[1] as i32,
                k_stride[2] as i32,
                v_stride[0] as i32,
                v_stride[1] as i32,
                v_stride[2] as i32,
                out_stride[0] as i32,
                out_stride[1] as i32,
                out_stride[2] as i32,
                self.softmax_scale,
                softcap,
                if is_causal { 1 } else { 0 },
                window_size_left,
                window_size_right,
                if is_bf16 { 1 } else { 0 },
            )
        };

        if result != 0 {
            candle::bail!("flash_attn_fwd failed with error code {result}")
        }

        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn-hip"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("flash-attn is not supported on CPU")
    }

    fn hip_fwd(
        &self,
        q: &candle::HipStorage,
        q_l: &Layout,
        k: &candle::HipStorage,
        k_l: &Layout,
        v: &candle::HipStorage,
        v_l: &Layout,
    ) -> Result<(candle::HipStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.hip_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false, false),
            DType::BF16 => self.hip_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true, false),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Causal flash attention variant
pub struct FlashAttnCausal(pub FlashAttn);

impl candle::CustomOp3 for FlashAttnCausal {
    fn name(&self) -> &'static str {
        "flash-attn-causal-hip"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("flash-attn is not supported on CPU")
    }

    fn hip_fwd(
        &self,
        q: &candle::HipStorage,
        q_l: &Layout,
        k: &candle::HipStorage,
        k_l: &Layout,
        v: &candle::HipStorage,
        v_l: &Layout,
    ) -> Result<(candle::HipStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.0.hip_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false, true),
            DType::BF16 => self.0.hip_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Convenience function for flash attention forward pass
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let op = FlashAttn::new(softmax_scale);
    if causal {
        q.apply_op3(k, v, FlashAttnCausal(op))
    } else {
        q.apply_op3(k, v, op)
    }
}

/// Flash attention with variable-length sequences (varlen)
/// Note: This is a placeholder - full varlen support requires additional kernel work
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    // For now, fall back to regular flash attention
    // TODO: Implement proper varlen support
    flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    /// Reference attention implementation on CPU for comparison
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        // Ensure we're on CPU
        let q = q.to_device(&Device::Cpu)?;
        let k = k.to_device(&Device::Cpu)?;
        let v = v.to_device(&Device::Cpu)?;

        // q, k, v: [batch, seq_len, num_heads, head_dim]
        // Transpose to [batch, num_heads, seq_len, head_dim] for matmul
        // Need contiguous() for matmul to work correctly
        let q = q.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len_q, head_dim]
        let k = k.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len_k, head_dim]
        let v = v.transpose(1, 2)?.contiguous()?; // [batch, num_heads, seq_len_k, head_dim]

        // Compute attention scores: Q @ K^T
        let k_t = k.transpose(3, 2)?.contiguous()?; // [batch, num_heads, head_dim, seq_len_k]
        let scores = q.matmul(&k_t)?; // [batch, num_heads, seq_len_q, seq_len_k]
        let scores = (scores * softmax_scale as f64)?;

        // Apply causal mask if needed - compute manually
        let (batch_size, num_heads, seq_len_q, seq_len_k) = scores.dims4()?;
        let scores_data = scores.flatten_all()?.to_vec1::<f32>()?;

        let mut masked_scores = scores_data.clone();
        if causal {
            for b in 0..batch_size {
                for h in 0..num_heads {
                    for i in 0..seq_len_q {
                        for j in 0..seq_len_k {
                            let idx = b * num_heads * seq_len_q * seq_len_k
                                + h * seq_len_q * seq_len_k
                                + i * seq_len_k
                                + j;
                            if j > i {
                                masked_scores[idx] = -1e9;
                            }
                        }
                    }
                }
            }
        }

        let scores = Tensor::from_vec(
            masked_scores,
            (batch_size, num_heads, seq_len_q, seq_len_k),
            &Device::Cpu,
        )?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, candle::D::Minus1)?;

        // Attention output: attn_weights @ V
        let attn_weights = attn_weights.contiguous()?;
        let output = attn_weights.matmul(&v)?; // [batch, num_heads, seq_len_q, head_dim]

        // Transpose back to [batch, seq_len, num_heads, head_dim]
        output.transpose(1, 2)?.contiguous()
    }

    #[test]
    fn test_flash_attn_basic() -> Result<()> {
        let device = Device::new_hip(0)?;

        let batch_size = 2;
        let seq_len = 64;
        let num_heads = 8;
        let head_dim = 64;

        let q = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)?
            .to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)?
            .to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)?
            .to_dtype(DType::F16)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let output = flash_attn(&q, &k, &v, softmax_scale, true)?;

        assert_eq!(output.dims(), &[batch_size, seq_len, num_heads, head_dim]);

        Ok(())
    }

    #[test]
    fn test_flash_attn_vs_reference_causal() -> Result<()> {
        let hip_device = Device::new_hip(0)?;
        let cpu_device = Device::Cpu;

        let batch_size = 1;
        let seq_len = 32;
        let num_heads = 4;
        let head_dim = 64;

        // Create random tensors on CPU first (use smaller values to avoid overflow)
        let q_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let k_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let v_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // Compute reference attention on CPU
        let ref_output = reference_attention(&q_cpu, &k_cpu, &v_cpu, softmax_scale, true)?;

        // Copy to HIP and compute flash attention
        let q_hip = q_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let k_hip = k_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let v_hip = v_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;

        let flash_output = flash_attn(&q_hip, &k_hip, &v_hip, softmax_scale, true)?;

        // Convert flash output back to CPU F32 for comparison
        let flash_output_cpu = flash_output.to_dtype(DType::F32)?.to_device(&cpu_device)?;

        // Compare outputs - allow for FP16 precision loss
        let diff = (&ref_output - &flash_output_cpu)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

        println!("Causal attention - Max diff: {}, Mean diff: {}", max_diff, mean_diff);

        // FP16 has ~3 decimal digits of precision, so we allow some tolerance
        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
        assert!(mean_diff < 0.01, "Mean diff {} too large", mean_diff);

        Ok(())
    }

    #[test]
    fn test_flash_attn_vs_reference_non_causal() -> Result<()> {
        let hip_device = Device::new_hip(0)?;
        let cpu_device = Device::Cpu;

        let batch_size = 1;
        let seq_len = 32;
        let num_heads = 4;
        let head_dim = 64;

        let q_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let k_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let v_cpu = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // Compute reference attention on CPU (non-causal)
        let ref_output = reference_attention(&q_cpu, &k_cpu, &v_cpu, softmax_scale, false)?;

        // Copy to HIP and compute flash attention
        let q_hip = q_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let k_hip = k_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let v_hip = v_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;

        let flash_output = flash_attn(&q_hip, &k_hip, &v_hip, softmax_scale, false)?;

        let flash_output_cpu = flash_output.to_dtype(DType::F32)?.to_device(&cpu_device)?;

        let diff = (&ref_output - &flash_output_cpu)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

        println!("Non-causal attention - Max diff: {}, Mean diff: {}", max_diff, mean_diff);

        assert!(max_diff < 0.1, "Max diff {} too large", max_diff);
        assert!(mean_diff < 0.01, "Mean diff {} too large", mean_diff);

        Ok(())
    }

    #[test]
    fn test_flash_attn_gqa() -> Result<()> {
        // Test grouped query attention (num_heads > num_heads_k)
        let hip_device = Device::new_hip(0)?;

        let batch_size = 1;
        let seq_len = 32;
        let num_heads = 8;
        let num_heads_k = 2; // GQA: 4 query heads per kv head
        let head_dim = 64;

        let q = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
            .to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads_k, head_dim), &hip_device)?
            .to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads_k, head_dim), &hip_device)?
            .to_dtype(DType::F16)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let output = flash_attn(&q, &k, &v, softmax_scale, true)?;

        assert_eq!(output.dims(), &[batch_size, seq_len, num_heads, head_dim]);

        // Verify output is not NaN or Inf
        let output_cpu = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let max_val = output_cpu.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_val.is_finite(), "Output contains NaN or Inf");

        Ok(())
    }

    #[test]
    fn test_flash_attn_different_head_dims() -> Result<()> {
        let hip_device = Device::new_hip(0)?;

        for head_dim in [32, 64, 96, 128] {
            let batch_size = 1;
            let seq_len = 16;
            let num_heads = 4;

            let q = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;
            let k = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;
            let v = Tensor::randn(0f32, 0.5, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();
            let output = flash_attn(&q, &k, &v, softmax_scale, true)?;

            assert_eq!(output.dims(), &[batch_size, seq_len, num_heads, head_dim]);

            let output_cpu = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
            let max_val = output_cpu.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(max_val.is_finite(), "Output contains NaN or Inf for head_dim={}", head_dim);

            println!("head_dim={} OK", head_dim);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_long_sequence() -> Result<()> {
        // Test with longer sequences to exercise the tiled kernel path
        let hip_device = Device::new_hip(0)?;
        let cpu_device = Device::Cpu;

        let batch_size = 2;
        let seq_len = 512; // Long enough to use tiled kernel
        let num_heads = 8;
        let head_dim = 64;

        let q_cpu = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let k_cpu = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;
        let v_cpu = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &cpu_device)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // Compute reference attention on CPU
        let ref_output = reference_attention(&q_cpu, &k_cpu, &v_cpu, softmax_scale, true)?;

        // Copy to HIP and compute flash attention
        let q_hip = q_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let k_hip = k_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;
        let v_hip = v_cpu.to_device(&hip_device)?.to_dtype(DType::F16)?;

        let flash_output = flash_attn(&q_hip, &k_hip, &v_hip, softmax_scale, true)?;

        let flash_output_cpu = flash_output.to_dtype(DType::F32)?.to_device(&cpu_device)?;

        let diff = (&ref_output - &flash_output_cpu)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

        println!("Long sequence (512) - Max diff: {}, Mean diff: {}", max_diff, mean_diff);

        // Slightly looser tolerance for longer sequences due to FP16 accumulation
        assert!(max_diff < 0.15, "Max diff {} too large for seq_len=512", max_diff);
        assert!(mean_diff < 0.02, "Mean diff {} too large for seq_len=512", mean_diff);

        Ok(())
    }

    #[test]
    fn test_flash_attn_benchmark() -> Result<()> {
        use std::time::Instant;

        let hip_device = Device::new_hip(0)?;

        println!("\n=== Flash Attention Benchmark ===\n");

        // Test multiple configurations
        for (batch_size, seq_len, num_heads, head_dim) in [
            (1, 512, 8, 64),
            (1, 1024, 16, 64),
            (1, 2048, 32, 128),
            (2, 2048, 32, 128),
        ] {
            let q = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;
            let k = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;
            let v = Tensor::randn(0f32, 0.3, (batch_size, seq_len, num_heads, head_dim), &hip_device)?
                .to_dtype(DType::F16)?;

            let softmax_scale = 1.0 / (head_dim as f32).sqrt();
            let iterations = 10;

            // FLOPs: 4 * batch * heads * seq_len^2 * head_dim (2 matmuls)
            let flops = 4.0 * batch_size as f64 * num_heads as f64 * (seq_len as f64).powi(2) * head_dim as f64;

            // Warmup
            let _ = flash_attn(&q, &k, &v, softmax_scale, true)?;
            unsafe { candle::hip_backend::ffi::hipDeviceSynchronize(); }

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = flash_attn(&q, &k, &v, softmax_scale, true)?;
            }
            unsafe { candle::hip_backend::ffi::hipDeviceSynchronize(); }
            let flash_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
            let flash_tflops = flops / (flash_ms / 1000.0) / 1e12;

            // Memory: read Q,K,V + write O = 4 * batch * seq * heads * dim * 2 bytes
            let mem_bytes = 4.0 * batch_size as f64 * seq_len as f64 * num_heads as f64 * head_dim as f64 * 2.0;
            let mem_gbps = mem_bytes / (flash_ms / 1000.0) / 1e9;

            println!("batch={}, seq={}, heads={}, dim={}", batch_size, seq_len, num_heads, head_dim);
            println!("  Time: {:.3} ms | {:.2} TFLOP/s | {:.1} GB/s", flash_ms, flash_tflops, mem_gbps);
        }

        println!("\n=================================\n");

        // Verify output is valid
        let q = Tensor::randn(0f32, 0.3, (1, 512, 8, 64), &hip_device)?.to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 0.3, (1, 512, 8, 64), &hip_device)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 0.3, (1, 512, 8, 64), &hip_device)?.to_dtype(DType::F16)?;
        let output = flash_attn(&q, &k, &v, 0.125, true)?;
        let output_cpu = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let max_val = output_cpu.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_val.is_finite(), "Output contains NaN or Inf");

        Ok(())
    }
}
