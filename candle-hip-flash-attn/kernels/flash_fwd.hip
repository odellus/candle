// Flash Attention Forward Pass for HIP/ROCm
// Optimized tiled implementation with online softmax
// No external dependencies (no CK, no CUTLASS)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cfloat>

// ============================================================================
// Configuration
// ============================================================================

constexpr int WARP_SIZE = 64;  // AMD wavefront size

// Tile sizes - tuned for AMD GPUs
// Each thread block processes BLOCK_M query rows
// We iterate over K/V in chunks of BLOCK_N
constexpr int BLOCK_M = 64;   // Query rows per block
constexpr int BLOCK_N = 64;   // K/V rows per iteration
constexpr int NUM_THREADS = 256;  // Threads per block

// ============================================================================
// Utility functions
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// ============================================================================
// Optimized Flash Attention Kernel
// ============================================================================
//
// Memory layout: Q, K, V are [batch, seqlen, num_heads, head_dim]
//
// Each thread block:
// - Handles BLOCK_M query rows for one (batch, head) pair
// - Iterates over all K/V in blocks of BLOCK_N
// - Uses shared memory for K/V tiles
// - Each thread accumulates output for (BLOCK_M / warps_per_block) rows
//
// Thread organization:
// - NUM_THREADS threads per block
// - Each thread is responsible for some (query_row, head_dim) elements

template<int HEADDIM, bool IS_CAUSAL>
__global__ __launch_bounds__(NUM_THREADS)
void flash_attn_fwd_tiled_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    const float* __restrict__ alibi_slopes,
    const int batch_size,
    const int seqlen_q,
    const int seqlen_k,
    const int num_heads,
    const int num_heads_k,
    const int q_batch_stride,
    const int q_row_stride,
    const int q_head_stride,
    const int k_batch_stride,
    const int k_row_stride,
    const int k_head_stride,
    const int v_batch_stride,
    const int v_row_stride,
    const int v_head_stride,
    const int o_batch_stride,
    const int o_row_stride,
    const int o_head_stride,
    const float softmax_scale,
    const float softcap,
    const int window_size_left,
    const int window_size_right
) {
    // Block/thread indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int m_block = blockIdx.x;  // Which BLOCK_M chunk of queries
    const int tid = threadIdx.x;

    // GQA head mapping
    const int kv_head_idx = head_idx / (num_heads / num_heads_k);

    // Early exit if this block is beyond sequence
    const int q_start = m_block * BLOCK_M;
    if (q_start >= seqlen_q) return;
    const int q_end = min(q_start + BLOCK_M, seqlen_q);

    // ALiBi slope
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;

    // Base pointers for this batch/head
    const __half* Q_base = Q + batch_idx * q_batch_stride + head_idx * q_head_stride;
    const __half* K_base = K + batch_idx * k_batch_stride + kv_head_idx * k_head_stride;
    const __half* V_base = V + batch_idx * v_batch_stride + kv_head_idx * v_head_stride;
    __half* O_base = O + batch_idx * o_batch_stride + head_idx * o_head_stride;

    // ========================================================================
    // Shared memory allocation
    // ========================================================================
    // Layout:
    // - sK: BLOCK_N x HEADDIM (for K tile)
    // - sV: BLOCK_N x HEADDIM (for V tile)
    // - sS: BLOCK_M x BLOCK_N (for attention scores)
    extern __shared__ char smem_[];

    __half* sK = reinterpret_cast<__half*>(smem_);
    __half* sV = sK + BLOCK_N * HEADDIM;
    float* sS = reinterpret_cast<float*>(sV + BLOCK_N * HEADDIM);
    float* sRowMax = sS + BLOCK_M * BLOCK_N;  // For block reduction
    float* sRowSum = sRowMax + BLOCK_M;

    // ========================================================================
    // Per-thread state
    // ========================================================================
    // Each thread handles multiple query rows and accumulates their outputs
    // We divide BLOCK_M rows among threads

    constexpr int ROWS_PER_THREAD = (BLOCK_M + NUM_THREADS - 1) / NUM_THREADS;

    // Per-row accumulators (in registers)
    float row_m[ROWS_PER_THREAD];      // Running max
    float row_l[ROWS_PER_THREAD];      // Running sum of exp
    float row_o[ROWS_PER_THREAD][HEADDIM];  // Output accumulator

    // Initialize
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        row_m[r] = -FLT_MAX;
        row_l[r] = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEADDIM; d++) {
            row_o[r][d] = 0.0f;
        }
    }

    // Load Q into registers (each thread loads its assigned rows)
    float q_reg[ROWS_PER_THREAD][HEADDIM];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int q_row = q_start + tid + r * NUM_THREADS;
        if (q_row < q_end) {
            #pragma unroll
            for (int d = 0; d < HEADDIM; d++) {
                q_reg[r][d] = __half2float(Q_base[q_row * q_row_stride + d]);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEADDIM; d++) {
                q_reg[r][d] = 0.0f;
            }
        }
    }

    // ========================================================================
    // Determine K/V iteration bounds
    // ========================================================================
    int k_start = 0;
    int k_end = seqlen_k;

    if (window_size_left >= 0) {
        k_start = max(0, q_start - window_size_left);
    }
    if (IS_CAUSAL) {
        // For causal, we can skip K positions beyond the last Q in this block
        k_end = min(seqlen_k, q_end);
    } else if (window_size_right >= 0) {
        k_end = min(seqlen_k, q_end + window_size_right);
    }

    // ========================================================================
    // Main loop over K/V blocks
    // ========================================================================
    for (int k_block = k_start; k_block < k_end; k_block += BLOCK_N) {
        const int k_block_end = min(k_block + BLOCK_N, k_end);
        const int num_k = k_block_end - k_block;

        // --------------------------------------------------------------------
        // Cooperatively load K and V tiles into shared memory
        // --------------------------------------------------------------------
        for (int i = tid; i < BLOCK_N * HEADDIM; i += NUM_THREADS) {
            int row = i / HEADDIM;
            int col = i % HEADDIM;
            if (row < num_k) {
                sK[row * HEADDIM + col] = K_base[(k_block + row) * k_row_stride + col];
                sV[row * HEADDIM + col] = V_base[(k_block + row) * v_row_stride + col];
            } else {
                sK[row * HEADDIM + col] = __float2half(0.0f);
                sV[row * HEADDIM + col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // --------------------------------------------------------------------
        // Compute attention scores: S = Q @ K^T
        // Each thread computes scores for its assigned Q rows
        // --------------------------------------------------------------------
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int q_row = q_start + tid + r * NUM_THREADS;
            if (q_row >= q_end) continue;

            // Find max score for this Q row against all K in this block
            float block_max = -FLT_MAX;

            for (int k = 0; k < num_k; k++) {
                // Dot product Q[q_row] @ K[k]
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEADDIM; d++) {
                    score += q_reg[r][d] * __half2float(sK[k * HEADDIM + d]);
                }
                score *= softmax_scale;

                // Apply softcap
                if (softcap > 0.0f) {
                    score = softcap * tanhf(score / softcap);
                }

                // Apply ALiBi
                if (alibi_slopes != nullptr) {
                    score += alibi_slope * ((k_block + k) - q_row);
                }

                // Apply causal mask
                if (IS_CAUSAL && (k_block + k) > q_row) {
                    score = -FLT_MAX;
                }

                // Store score
                sS[(tid + r * NUM_THREADS) * BLOCK_N + k] = score;
                block_max = fmaxf(block_max, score);
            }

            // ------------------------------------------------------------
            // Online softmax update
            // ------------------------------------------------------------
            float prev_max = row_m[r];
            float new_max = fmaxf(prev_max, block_max);

            // Scale factor for previous accumulator
            float scale_prev = expf(prev_max - new_max);

            // Rescale previous l and o
            row_l[r] *= scale_prev;
            #pragma unroll
            for (int d = 0; d < HEADDIM; d++) {
                row_o[r][d] *= scale_prev;
            }

            // Compute exp(score - new_max) and accumulate
            float block_sum = 0.0f;
            for (int k = 0; k < num_k; k++) {
                float score = sS[(tid + r * NUM_THREADS) * BLOCK_N + k];
                float p = expf(score - new_max);
                block_sum += p;

                // Accumulate O += p * V[k]
                #pragma unroll
                for (int d = 0; d < HEADDIM; d++) {
                    row_o[r][d] += p * __half2float(sV[k * HEADDIM + d]);
                }
            }

            row_l[r] += block_sum;
            row_m[r] = new_max;
        }

        __syncthreads();
    }

    // ========================================================================
    // Normalize and write output
    // ========================================================================
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int q_row = q_start + tid + r * NUM_THREADS;
        if (q_row >= q_end) continue;

        float inv_l = (row_l[r] > 0.0f) ? (1.0f / row_l[r]) : 0.0f;

        #pragma unroll
        for (int d = 0; d < HEADDIM; d++) {
            O_base[q_row * o_row_stride + d] = __float2half(row_o[r][d] * inv_l);
        }

        // Write LSE if requested
        if (L != nullptr) {
            int lse_idx = batch_idx * num_heads * seqlen_q + head_idx * seqlen_q + q_row;
            L[lse_idx] = row_m[r] + logf(row_l[r] + 1e-6f);
        }
    }
}

// ============================================================================
// Fallback simple kernel (for correctness verification)
// ============================================================================

template<int HEADDIM, bool IS_CAUSAL>
__global__ void flash_attn_fwd_simple_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    const float* __restrict__ alibi_slopes,
    const int batch_size,
    const int seqlen_q,
    const int seqlen_k,
    const int num_heads,
    const int num_heads_k,
    const int q_batch_stride,
    const int q_row_stride,
    const int q_head_stride,
    const int k_batch_stride,
    const int k_row_stride,
    const int k_head_stride,
    const int v_batch_stride,
    const int v_row_stride,
    const int v_head_stride,
    const int o_batch_stride,
    const int o_row_stride,
    const int o_head_stride,
    const float softmax_scale,
    const float softcap,
    const int window_size_left,
    const int window_size_right
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_row >= seqlen_q) return;

    const int kv_head_idx = head_idx / (num_heads / num_heads_k);

    const __half* Q_ptr = Q + batch_idx * q_batch_stride + head_idx * q_head_stride + q_row * q_row_stride;
    const __half* K_ptr = K + batch_idx * k_batch_stride + kv_head_idx * k_head_stride;
    const __half* V_ptr = V + batch_idx * v_batch_stride + kv_head_idx * v_head_stride;
    __half* O_ptr = O + batch_idx * o_batch_stride + head_idx * o_head_stride + q_row * o_row_stride;

    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;

    // Load query into registers
    float q_reg[HEADDIM];
    #pragma unroll
    for (int d = 0; d < HEADDIM; d++) {
        q_reg[d] = __half2float(Q_ptr[d]);
    }

    // K/V range
    int k_start = 0;
    int k_end = seqlen_k;

    if (window_size_left >= 0) {
        k_start = max(0, q_row - window_size_left);
    }
    if (IS_CAUSAL) {
        k_end = min(seqlen_k, q_row + 1);
    } else if (window_size_right >= 0) {
        k_end = min(seqlen_k, q_row + window_size_right + 1);
    }

    // Online softmax
    float m = -FLT_MAX;
    float l = 0.0f;
    float o_acc[HEADDIM] = {0.0f};

    for (int k_row = k_start; k_row < k_end; k_row++) {
        const __half* K_row = K_ptr + k_row * k_row_stride;
        const __half* V_row = V_ptr + k_row * v_row_stride;

        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEADDIM; d++) {
            score += q_reg[d] * __half2float(K_row[d]);
        }
        score *= softmax_scale;

        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }

        if (alibi_slopes != nullptr) {
            score += alibi_slope * (k_row - q_row);
        }

        float m_new = fmaxf(m, score);
        float p = expf(score - m_new);
        float scale = expf(m - m_new);

        l = l * scale + p;
        #pragma unroll
        for (int d = 0; d < HEADDIM; d++) {
            o_acc[d] = o_acc[d] * scale + p * __half2float(V_row[d]);
        }

        m = m_new;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int d = 0; d < HEADDIM; d++) {
        O_ptr[d] = __float2half(o_acc[d] * inv_l);
    }

    if (L != nullptr) {
        int lse_idx = batch_idx * num_heads * seqlen_q + head_idx * seqlen_q + q_row;
        L[lse_idx] = m + logf(l + 1e-6f);
    }
}

// ============================================================================
// Dispatch function
// ============================================================================

void run_flash_attn_fwd_fp16(
    hipStream_t stream,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* lse_ptr,
    const void* alibi_slopes_ptr,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int q_batch_stride,
    int q_row_stride,
    int q_head_stride,
    int k_batch_stride,
    int k_row_stride,
    int k_head_stride,
    int v_batch_stride,
    int v_row_stride,
    int v_head_stride,
    int o_batch_stride,
    int o_row_stride,
    int o_head_stride,
    float softmax_scale,
    float softcap,
    bool is_causal,
    int window_size_left,
    int window_size_right
) {
    // Choose kernel based on sequence length
    // Use tiled kernel for longer sequences, simple for short ones
    const bool use_tiled = (seqlen_k >= 64);

    if (use_tiled) {
        // Tiled kernel: one block per BLOCK_M query rows
        const int m_blocks = (seqlen_q + BLOCK_M - 1) / BLOCK_M;
        dim3 grid(m_blocks, num_heads, batch_size);
        dim3 block(NUM_THREADS);

        // Shared memory: sK + sV + sS + reduction scratch
        // sK: BLOCK_N * head_dim * sizeof(half)
        // sV: BLOCK_N * head_dim * sizeof(half)
        // sS: BLOCK_M * BLOCK_N * sizeof(float)
        // sRowMax/Sum: BLOCK_M * sizeof(float) each
        size_t smem_size = BLOCK_N * head_dim * sizeof(__half) * 2 +
                           BLOCK_M * BLOCK_N * sizeof(float) +
                           BLOCK_M * sizeof(float) * 2;

        #define LAUNCH_TILED_KERNEL(HDIM, CAUSAL) \
            flash_attn_fwd_tiled_kernel<HDIM, CAUSAL><<<grid, block, smem_size, stream>>>( \
                (const __half*)q_ptr, (const __half*)k_ptr, (const __half*)v_ptr, \
                (__half*)o_ptr, (float*)lse_ptr, (const float*)alibi_slopes_ptr, \
                batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, \
                q_batch_stride, q_row_stride, q_head_stride, \
                k_batch_stride, k_row_stride, k_head_stride, \
                v_batch_stride, v_row_stride, v_head_stride, \
                o_batch_stride, o_row_stride, o_head_stride, \
                softmax_scale, softcap, window_size_left, window_size_right \
            )

        if (is_causal) {
            switch (head_dim) {
                case 32:  LAUNCH_TILED_KERNEL(32, true); break;
                case 64:  LAUNCH_TILED_KERNEL(64, true); break;
                case 96:  LAUNCH_TILED_KERNEL(96, true); break;
                case 128: LAUNCH_TILED_KERNEL(128, true); break;
                default:  LAUNCH_TILED_KERNEL(64, true); break;
            }
        } else {
            switch (head_dim) {
                case 32:  LAUNCH_TILED_KERNEL(32, false); break;
                case 64:  LAUNCH_TILED_KERNEL(64, false); break;
                case 96:  LAUNCH_TILED_KERNEL(96, false); break;
                case 128: LAUNCH_TILED_KERNEL(128, false); break;
                default:  LAUNCH_TILED_KERNEL(64, false); break;
            }
        }
        #undef LAUNCH_TILED_KERNEL
    } else {
        // Simple kernel for short sequences
        const int threads_per_block = 256;
        const int q_blocks = (seqlen_q + threads_per_block - 1) / threads_per_block;
        dim3 grid(q_blocks, num_heads, batch_size);
        dim3 block(threads_per_block);

        #define LAUNCH_SIMPLE_KERNEL(HDIM, CAUSAL) \
            flash_attn_fwd_simple_kernel<HDIM, CAUSAL><<<grid, block, 0, stream>>>( \
                (const __half*)q_ptr, (const __half*)k_ptr, (const __half*)v_ptr, \
                (__half*)o_ptr, (float*)lse_ptr, (const float*)alibi_slopes_ptr, \
                batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, \
                q_batch_stride, q_row_stride, q_head_stride, \
                k_batch_stride, k_row_stride, k_head_stride, \
                v_batch_stride, v_row_stride, v_head_stride, \
                o_batch_stride, o_row_stride, o_head_stride, \
                softmax_scale, softcap, window_size_left, window_size_right \
            )

        if (is_causal) {
            switch (head_dim) {
                case 32:  LAUNCH_SIMPLE_KERNEL(32, true); break;
                case 64:  LAUNCH_SIMPLE_KERNEL(64, true); break;
                case 96:  LAUNCH_SIMPLE_KERNEL(96, true); break;
                case 128: LAUNCH_SIMPLE_KERNEL(128, true); break;
                default:  LAUNCH_SIMPLE_KERNEL(64, true); break;
            }
        } else {
            switch (head_dim) {
                case 32:  LAUNCH_SIMPLE_KERNEL(32, false); break;
                case 64:  LAUNCH_SIMPLE_KERNEL(64, false); break;
                case 96:  LAUNCH_SIMPLE_KERNEL(96, false); break;
                case 128: LAUNCH_SIMPLE_KERNEL(128, false); break;
                default:  LAUNCH_SIMPLE_KERNEL(64, false); break;
            }
        }
        #undef LAUNCH_SIMPLE_KERNEL
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int flash_attn_fwd(
    hipStream_t stream,
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* out_ptr,
    void* softmax_lse_ptr,
    const void* alibi_slopes_ptr,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    int head_dim,
    int q_batch_stride,
    int q_row_stride,
    int q_head_stride,
    int k_batch_stride,
    int k_row_stride,
    int k_head_stride,
    int v_batch_stride,
    int v_row_stride,
    int v_head_stride,
    int out_batch_stride,
    int out_row_stride,
    int out_head_stride,
    float softmax_scale,
    float softcap,
    int is_causal,
    int window_size_left,
    int window_size_right,
    int is_bf16
) {
    if (is_bf16) {
        // TODO: Add BF16 support
        return -1;
    }

    run_flash_attn_fwd_fp16(
        stream,
        q_ptr, k_ptr, v_ptr, out_ptr, softmax_lse_ptr, alibi_slopes_ptr,
        batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_dim,
        q_batch_stride, q_row_stride, q_head_stride,
        k_batch_stride, k_row_stride, k_head_stride,
        v_batch_stride, v_row_stride, v_head_stride,
        out_batch_stride, out_row_stride, out_head_stride,
        softmax_scale, softcap,
        is_causal != 0,
        window_size_left, window_size_right
    );

    return 0;
}

} // extern "C"
