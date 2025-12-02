//! Kernel management and pipeline caching
//!
//! Similar to candle-metal-kernels' Kernels struct - handles loading
//! and caching of compute pipelines.
//!
//! Descriptor pool management is inspired by llama.cpp/ggml-vulkan:
//! - Pre-allocates descriptor sets in pools of POOL_SIZE (256)
//! - Uses an index counter to cycle through sets
//! - Creates new pools on demand when exhausted
//! - Resets index at synchronization points

use crate::context::VulkanContext;
use crate::error::{Result, VulkanError};
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Shader sources embedded at compile time
pub mod source {
    /// Matrix-vector multiplication shader (our custom one)
    pub const MATVEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/matvec.spv"));

    // GGML-derived shaders (MIT licensed from llama.cpp)

    // Unary ops - contiguous (fast path, simple push constants)
    pub const EXP_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/exp_f32.spv"));
    pub const SILU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/silu_f32.spv"));
    pub const GELU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gelu_f32.spv"));
    pub const RELU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/relu_f32.spv"));

    // Unary ops - strided (full stride support via generic_unary_head.glsl)
    pub const EXP_F32_STRIDED: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/exp_f32_strided.spv"));
    pub const SILU_F32_STRIDED: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/silu_f32_strided.spv"));
    pub const GELU_F32_STRIDED: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gelu_f32_strided.spv"));
    pub const RELU_F32_STRIDED: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/relu_f32_strided.spv"));

    // Other strided unary ops
    pub const SQRT_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sqrt_f32.spv"));
    pub const SIN_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sin_f32.spv"));
    pub const COS_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cos_f32.spv"));
    pub const CLAMP_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/clamp_f32.spv"));
    pub const SCALE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/scale_f32.spv"));

    // Binary ops (strided with broadcast support via generic_binary_head.glsl)
    pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/add_f32.spv"));
    pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/mul_f32.spv"));
    pub const DIV_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/div_f32.spv"));

    // Copy (strided)
    pub const COPY_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/copy_f32.spv"));

    // Dequantization shaders
    pub const DEQUANT_Q4_0_F32: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/dequant_q4_0_f32.spv"));
    pub const DEQUANT_Q8_0_F32: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/dequant_q8_0_f32.spv"));

    // Quantized matrix-vector multiplication shaders
    pub const MUL_MAT_VEC_Q4_K_F32: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/mul_mat_vec_q4_k_f32.spv"));
}

/// Number of descriptor sets per pool (matches ggml-vulkan's VK_DEVICE_DESCRIPTOR_POOL_SIZE)
const POOL_SIZE: u32 = 256;

/// Maximum number of storage buffer bindings per descriptor set
const MAX_BUFFERS_PER_SET: u32 = 8;

/// A cached compute pipeline with its layout
pub struct CachedPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

/// Descriptor pool management inspired by llama.cpp/ggml-vulkan
struct DescriptorPoolManager {
    /// All created pools
    pools: Vec<vk::DescriptorPool>,
    /// Pre-allocated descriptor sets from all pools
    descriptor_sets: Vec<vk::DescriptorSet>,
    /// Current index into descriptor_sets (cycles on reset)
    current_idx: AtomicUsize,
    /// The common descriptor set layout used for all sets
    common_layout: vk::DescriptorSetLayout,
}

impl DescriptorPoolManager {
    fn new(device: &ash::Device, common_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let mut manager = Self {
            pools: Vec::new(),
            descriptor_sets: Vec::new(),
            current_idx: AtomicUsize::new(0),
            common_layout,
        };

        // Pre-allocate one pool worth of descriptor sets
        manager.grow(device)?;

        Ok(manager)
    }

    /// Allocate a new pool and its descriptor sets
    fn grow(&mut self, device: &ash::Device) -> Result<()> {
        // Create a new pool
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_BUFFERS_PER_SET * POOL_SIZE)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(POOL_SIZE);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        self.pools.push(pool);

        // Allocate all descriptor sets from this pool
        let layouts = vec![self.common_layout; POOL_SIZE as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        self.descriptor_sets.extend(sets);

        Ok(())
    }

    /// Get the next available descriptor set, growing if needed
    fn acquire(&mut self, device: &ash::Device) -> Result<vk::DescriptorSet> {
        let idx = self.current_idx.fetch_add(1, Ordering::Relaxed);

        // Grow if we've exhausted all sets
        if idx >= self.descriptor_sets.len() {
            self.grow(device)?;
        }

        // After grow, idx should be valid
        if idx < self.descriptor_sets.len() {
            Ok(self.descriptor_sets[idx])
        } else {
            // This shouldn't happen, but handle it gracefully
            Err(VulkanError::DescriptorPoolExhausted)
        }
    }

    /// Reset the index counter (call at synchronization points)
    fn reset(&self) {
        self.current_idx.store(0, Ordering::Relaxed);
    }

    /// Cleanup all pools
    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for pool in self.pools.drain(..) {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        self.descriptor_sets.clear();
    }
}

/// Manages compute pipelines and shader loading
pub struct Kernels {
    context: Arc<VulkanContext>,
    pipelines: RwLock<HashMap<&'static str, CachedPipeline>>,
    /// Common descriptor set layout for all kernels (max bindings)
    common_descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool manager (llama.cpp style)
    pool_manager: Mutex<DescriptorPoolManager>,
}

impl Kernels {
    /// Create a new Kernels instance
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        // Create a common descriptor set layout with max bindings
        // All our kernels use at most MAX_BUFFERS_PER_SET storage buffers
        let bindings: Vec<_> = (0..MAX_BUFFERS_PER_SET)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let common_descriptor_set_layout =
            unsafe { context.device.create_descriptor_set_layout(&layout_info, None)? };

        // Create the pool manager
        let pool_manager =
            DescriptorPoolManager::new(&context.device, common_descriptor_set_layout)?;

        Ok(Self {
            context,
            pipelines: RwLock::new(HashMap::new()),
            common_descriptor_set_layout,
            pool_manager: Mutex::new(pool_manager),
        })
    }

    /// Get the Vulkan context
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    /// Reset descriptor set allocation index
    /// Call this at synchronization points (e.g., after queue submit + wait)
    pub fn reset_descriptor_sets(&self) {
        self.pool_manager.lock().reset();
    }

    /// Load or retrieve a cached pipeline
    pub fn load_pipeline(
        &self,
        name: &'static str,
        spirv: &[u8],
        _num_buffers: u32,
        push_constant_size: u32,
    ) -> Result<&CachedPipeline> {
        // Check cache first
        {
            let cache = self.pipelines.read();
            if let Some(pipeline) = cache.get(name) {
                // Safety: we never remove from the cache, so this pointer remains valid
                return Ok(unsafe { &*(pipeline as *const CachedPipeline) });
            }
        }

        // Create new pipeline (outside of any lock)
        let pipeline = self.create_pipeline(spirv, push_constant_size)?;

        // Insert into cache
        {
            let mut cache = self.pipelines.write();
            cache.insert(name, pipeline);
        }

        // Re-acquire read lock to return reference
        let cache = self.pipelines.read();
        // Safety: we just inserted it and never remove from cache
        Ok(unsafe { &*(cache.get(name).unwrap() as *const CachedPipeline) })
    }

    fn create_pipeline(
        &self,
        spirv: &[u8],
        push_constant_size: u32,
    ) -> Result<CachedPipeline> {
        let device = &self.context.device;

        // Create shader module
        // SPIR-V must be aligned to u32
        let spirv_words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe {
            device
                .create_shader_module(&shader_info, None)
                .map_err(|e| VulkanError::ShaderLoad(e.to_string()))?
        };

        // Use the common descriptor set layout
        let descriptor_set_layout = self.common_descriptor_set_layout;

        // Create pipeline layout with push constants
        let push_constant_range = if push_constant_size > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_size)]
        } else {
            vec![]
        };

        let layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_range);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // Create compute pipeline
        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| VulkanError::PipelineCreation(format!("{:?}", e.1)))?[0]
        };

        // Clean up shader module (no longer needed)
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(CachedPipeline {
            pipeline,
            layout,
            descriptor_set_layout,
        })
    }

    /// Allocate a descriptor set from the pool
    /// Uses llama.cpp-style cycling through pre-allocated sets
    pub fn allocate_descriptor_set(
        &self,
        _layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        // Ignore the passed layout, use our pre-allocated sets
        let mut manager = self.pool_manager.lock();
        manager.acquire(&self.context.device)
    }

    /// Free a descriptor set - no-op with the new pooling strategy
    /// Sets are reused via reset_descriptor_sets() at sync points
    pub fn free_descriptor_set(&self, _set: vk::DescriptorSet) -> Result<()> {
        // No-op: sets are recycled at sync points, not individually freed
        Ok(())
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        unsafe {
            // Destroy pool manager first (frees all descriptor pools)
            self.pool_manager.lock().destroy(&self.context.device);

            // Destroy pipelines
            let cache = self.pipelines.read();
            for (_, p) in cache.iter() {
                self.context.device.destroy_pipeline(p.pipeline, None);
                self.context.device.destroy_pipeline_layout(p.layout, None);
                // Don't destroy descriptor_set_layout here - it's the common one
            }

            // Destroy the common descriptor set layout
            self.context
                .device
                .destroy_descriptor_set_layout(self.common_descriptor_set_layout, None);
        }
    }
}
