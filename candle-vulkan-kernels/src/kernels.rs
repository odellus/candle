//! Vulkan kernel management and pipeline creation
//!
//! This module provides kernel compilation and execution following GGML patterns
//! for efficient GPU tensor operations.

use crate::{
    device::VulkanContext,
    error::{Result, VulkanError},
    quant_types::{BlockQ4_0, GgmlDType},
    storage::VulkanStorage,
};
use ash::{vk, Device};
use parking_lot::Mutex;
use shaderc::{CompileOptions, Compiler, OptimizationLevel, ShaderKind, SourceLanguage};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Kernels {
    context: Arc<VulkanContext>,
    compiler: Arc<Compiler>,
    shader_cache: Arc<Mutex<HashMap<String, vk::ShaderModule>>>,
    pipeline_cache: Arc<Mutex<HashMap<String, vk::Pipeline>>>,
}

impl Kernels {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        Ok(Self {
            context,
            compiler: Arc::new(
                Compiler::new().ok_or_else(|| {
                    VulkanError::Message("Failed to create shader compiler".into())
                })?,
            ),
            shader_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Compile SPIR-V shader from GLSL source
    pub fn compile_shader(
        &self,
        glsl_source: &str,
        shader_type: ShaderKind,
        entry_point: &str,
    ) -> Result<Vec<u32>> {
        let options = CompileOptions::new()
            .ok_or_else(|| VulkanError::Message("Failed to create compile options".into()))?;

        let options = options
            .set_source_language(SourceLanguage::GLSL)
            .set_optimization_level(OptimizationLevel::Performance);

        let result = self.compiler.compile_into_spirv(
            glsl_source,
            shader_type,
            "shader.comp",
            "main",
            Some(&options),
        );

        match result {
            Ok(spirv) => Ok(spirv.as_binary().to_vec()),
            Err(e) => Err(VulkanError::ShaderCompilation(e.to_string())),
        }
    }

    /// Create shader module from SPIR-V bytecode
    pub fn create_shader_module(&self, spirv: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);

        unsafe {
            self.context
                .device
                .create_shader_module(&create_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    /// Get or create shader module from GLSL source
    pub fn get_shader_module(&self, glsl_source: &str) -> Result<vk::ShaderModule> {
        let shader_key = format!(
            "sha256:{}",
            sha2::Sha256::new().chain_update(glsl_source).finalize()
        );

        if let Some(shader) = self.shader_cache.lock().get(&shader_key) {
            return Ok(*shader);
        }

        let spirv = self.compile_shader(glsl_source, ShaderKind::Compute, "main")?;
        let shader_module = self.create_shader_module(&spirv)?;

        self.shader_cache.lock().insert(shader_key, shader_module);

        Ok(shader_module)
    }

    /// Create compute pipeline from shader
    pub fn create_pipeline(
        &self,
        shader_module: vk::ShaderModule,
        push_constant_range: vk::PushConstantRange,
        descriptor_set_layout_bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::Pipeline> {
        // Create descriptor set layout
        let layout_bindings = descriptor_set_layout_bindings
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let descriptor_set_layout = self.create_descriptor_set_layout(&layout_bindings)?;

        // Create pipeline layout
        let pipeline_layout =
            self.create_pipeline_layout(&[descriptor_set_layout], &[push_constant_range])?;

        // Create pipeline
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name("main")
                    .build(),
            )
            .layout(pipeline_layout);

        let pipelines = unsafe {
            self.context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        }
        .map_err(|(_, e)| VulkanError::AshError(e))?;

        Ok(pipelines[0])
    }

    /// Get or create pipeline for kernel
    pub fn get_pipeline(
        &self,
        shader_source: &str,
        push_constant_range: vk::PushConstantRange,
        descriptor_bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::Pipeline> {
        let pipeline_key = format!(
            "sha256:{}",
            sha2::Sha256::new()
                .chain_update(shader_source)
                .chain_update(format!("{:?}", push_constant_range).as_bytes())
                .chain_update(format!("{:?}", descriptor_bindings).as_bytes())
                .finalize()
        );

        if let Some(pipeline) = self.pipeline_cache.lock().get(&pipeline_key) {
            return Ok(*pipeline);
        }

        let shader_module = self.get_shader_module(shader_source)?;
        let pipeline =
            self.create_pipeline(shader_module, push_constant_range, descriptor_bindings)?;

        self.pipeline_cache.lock().insert(pipeline_key, pipeline);

        Ok(pipeline)
    }

    /// Create descriptor set layout from bindings
    pub fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        unsafe {
            self.context
                .device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    /// Create pipeline layout with descriptor set layouts and push constants
    pub fn create_pipeline_layout(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<vk::PipelineLayout> {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        unsafe {
            self.context
                .device
                .create_pipeline_layout(&create_info, None)
                .map_err(|e| VulkanError::AshError(e))
        }
    }

    /// Create descriptor set from layout
    pub fn allocate_descriptor_set(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.context.descriptor_pool)
            .set_layouts(&[descriptor_set_layout]);

        unsafe {
            let descriptor_sets = self
                .context
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| VulkanError::AshError(e))?;

            Ok(descriptor_sets[0])
        }
    }

    /// Update descriptor set with buffer bindings
    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        buffer_infos: &[vk::DescriptorBufferInfo],
    ) -> Result<()> {
        let write_descriptor_sets = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, buffer_info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[*buffer_info])
                    .build()
            })
            .collect::<Vec<_>>();

        unsafe {
            self.context
                .device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Ok(())
    }

    /// Execute compute kernel
    pub fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        descriptor_set: vk::DescriptorSet,
        push_constants: &[u8],
        dispatch_groups: [u32; 3],
    ) -> Result<()> {
        unsafe {
            self.context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            if !push_constants.is_empty() {
                // TODO: Store pipeline layout and use it here
                // self.context.device.cmd_push_constants(
                //     command_buffer,
                //     pipeline_layout,
                //     vk::ShaderStageFlags::COMPUTE,
                //     0,
                //     push_constants,
                // );
            }

            // TODO: Store pipeline layout and use it here
            // self.context.device.cmd_bind_descriptor_sets(
            //     command_buffer,
            //     vk::PipelineBindPoint::COMPUTE,
            //     pipeline_layout,
            //     0,
            //     &[descriptor_set],
            //     &[],
            // );

            self.context.device.cmd_dispatch(
                command_buffer,
                dispatch_groups[0],
                dispatch_groups[1],
                dispatch_groups[2],
            );
        }

        Ok(())
    }
}

/// Handle for specific kernel operations
pub struct KernelHandle {
    pub pipeline: vk::Pipeline,
    pub descriptor_set: vk::DescriptorSet,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl KernelHandle {
    pub fn new(
        kernels: &Kernels,
        shader_source: &str,
        push_constant_range: vk::PushConstantRange,
        descriptor_bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<Self> {
        let pipeline =
            kernels.get_pipeline(shader_source, push_constant_range, descriptor_bindings)?;
        let descriptor_set_layout = kernels.create_descriptor_set_layout(descriptor_bindings)?;
        let descriptor_set = kernels.allocate_descriptor_set(descriptor_set_layout)?;

        Ok(Self {
            pipeline,
            descriptor_set,
            descriptor_set_layout,
        })
    }

    pub fn update_buffers(
        &self,
        kernels: &Kernels,
        buffer_infos: &[vk::DescriptorBufferInfo],
    ) -> Result<()> {
        kernels.update_descriptor_set(self.descriptor_set, buffer_infos)
    }
}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        // Note: Cleanup should be handled by the context destruction
        // In a real implementation, we would need to track these and free them
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_compilation() {
        let glsl_source = r#"
        #version 450
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 256) in;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            // Simple shader for testing
        }
        "#;

        let compiler = Compiler::new().expect("Failed to create compiler");
        let result = compiler.compile_into_spirv(
            glsl_source,
            ShaderKind::Compute,
            "test.comp",
            "main",
            Some(
                &CompileOptions::new()
                    .unwrap()
                    .set_source_language(SourceLanguage::GLSL),
            ),
        );

        assert!(result.is_ok(), "Shader compilation failed: {:?}", result);
    }

    #[test]
    fn test_kernel_key_generation() {
        let shader_source = "simple shader";
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(16)
            .build();

        let descriptor_bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build()];

        // This test ensures the key generation works without panicking
        let key = format!(
            "sha256:{}",
            sha2::Sha256::new()
                .chain_update(shader_source)
                .chain_update(format!("{:?}", push_constant_range).as_bytes())
                .chain_update(format!("{:?}", descriptor_bindings).as_bytes())
                .finalize()
        );

        assert!(!key.is_empty());
    }
}
