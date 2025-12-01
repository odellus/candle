//! Kernel management and pipeline caching
//!
//! Similar to candle-metal-kernels' Kernels struct - handles loading
//! and caching of compute pipelines.

use crate::context::VulkanContext;
use crate::error::{Result, VulkanError};
use ash::vk;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Shader sources embedded at compile time
pub mod source {
    /// Matrix-vector multiplication shader (our custom one)
    pub const MATVEC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/matvec.spv"));

    // GGML-derived shaders (MIT licensed from llama.cpp)
    // Unary ops
    pub const EXP_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/exp_f32.spv"));
    pub const SILU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/silu_f32.spv"));
    pub const GELU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gelu_f32.spv"));
    pub const RELU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/relu_f32.spv"));
    pub const SQRT_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sqrt_f32.spv"));
    pub const SIN_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sin_f32.spv"));
    pub const COS_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cos_f32.spv"));
    pub const CLAMP_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/clamp_f32.spv"));
    pub const SCALE_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/scale_f32.spv"));

    // Binary ops
    pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/add_f32.spv"));
    pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/mul_f32.spv"));
    pub const DIV_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/div_f32.spv"));

    // Copy
    pub const COPY_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/copy_f32.spv"));
}

/// A cached compute pipeline with its layout
pub struct CachedPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

/// Manages compute pipelines and shader loading
pub struct Kernels {
    context: Arc<VulkanContext>,
    pipelines: RwLock<HashMap<&'static str, CachedPipeline>>,
    descriptor_pool: vk::DescriptorPool,
}

impl Kernels {
    /// Create a new Kernels instance
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        // Create a descriptor pool for kernel descriptor sets
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(256)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(64)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = unsafe { context.device.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            context,
            pipelines: RwLock::new(HashMap::new()),
            descriptor_pool,
        })
    }

    /// Get the Vulkan context
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    /// Load or retrieve a cached pipeline
    pub fn load_pipeline(
        &self,
        name: &'static str,
        spirv: &[u8],
        num_buffers: u32,
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
        let pipeline = self.create_pipeline(spirv, num_buffers, push_constant_size)?;

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
        num_buffers: u32,
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

        // Create descriptor set layout with N storage buffer bindings
        let bindings: Vec<_> = (0..num_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

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

    /// Allocate a descriptor set for a pipeline
    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { self.context.device.allocate_descriptor_sets(&alloc_info)? };
        Ok(sets[0])
    }

    /// Free a descriptor set
    pub fn free_descriptor_set(&self, set: vk::DescriptorSet) -> Result<()> {
        unsafe {
            self.context
                .device
                .free_descriptor_sets(self.descriptor_pool, &[set])?;
        }
        Ok(())
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        unsafe {
            let cache = self.pipelines.read();
            for (_, p) in cache.iter() {
                self.context.device.destroy_pipeline(p.pipeline, None);
                self.context.device.destroy_pipeline_layout(p.layout, None);
                self.context
                    .device
                    .destroy_descriptor_set_layout(p.descriptor_set_layout, None);
            }
            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
