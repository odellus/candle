use anyhow::Result;
use ash::util::read_spv;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Entry};
use gpu_allocator::vulkan::Allocator;
use log::{error, info};
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::ptr;

// Simple buffer struct based on ash-comp-shader-example
#[derive(Debug)]
struct Buffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    size: usize,
    device: ash::Device,
}

impl Buffer {
    unsafe fn write_data<T>(&mut self, allocator: &mut Allocator, data: &[T]) -> Result<()> {
        let allocation = self
            .allocation
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Buffer has no allocation"))?;

        let mapped_ptr = allocator.map_memory(allocation)?;
        let data_ptr = mapped_ptr as *const T as *mut T;
        ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        allocator.unmap_memory(allocation)?;

        Ok(())
    }

    unsafe fn read_data<T>(&mut self, allocator: &mut Allocator, data: &mut [T]) -> Result<()> {
        let allocation = self
            .allocation
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Buffer has no allocation"))?;

        let mapped_ptr = allocator.map_memory(allocation)?;
        let data_ptr = mapped_ptr as *const T;
        ptr::copy_nonoverlapping(data_ptr, data.as_mut_ptr(), data.len());
        allocator.unmap_memory(allocation)?;

        Ok(())
    }
}

// Vulkan context structure based on ash-comp-shader-example
struct VulkanContext {
    entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
}

impl VulkanContext {
    fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        // Create instance
        let app_name = CString::new("candle-vulkan-demo")?;
        let engine_name = CString::new("Candle")?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::make_version(1, 2, 0));

        let layer_names: &[&CString] = &[&CString::new("VK_LAYER_KHRONOS_validation")?];
        let extension_names = ash_window::enumerate_required_extensions()?;

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(layer_names)
            .enabled_extension_names(&extension_names);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // Select physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices[0]; // Use first GPU

        // Find compute queue family
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let compute_queue_family = queue_family_properties
            .iter()
            .position(|qp| qp.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow::anyhow!("No compute queue family found"))?
            as u32;

        // Create device
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&[1.0]);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        // Get compute queue
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        // Create command pool
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(compute_queue_family);

        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };

        // Create descriptor pool
        let descriptor_pool_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3);

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(std::slice::from_ref(&descriptor_pool_size))
            .max_sets(1);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? };

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            compute_queue,
            compute_queue_family,
            command_pool,
            descriptor_pool,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: usize,
    usage: vk::BufferUsageFlags,
) -> Result<Buffer> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let allocation_create_info = gpu_allocator::vulkan::AllocationCreateDesc {
        name: "Buffer",
        requirements: gpu_allocator::vulkan::MemoryRequirements {
            size: size as u64,
            alignment: 1,
            memory_type_bits: 0,
        },
        location: gpu_allocator::vulkan::MemoryLocation::CpuToGpu,
        linear: true,
    };

    let (buffer, allocation) =
        unsafe { allocator.create_buffer(&buffer_info, &allocation_create_info)? };

    Ok(Buffer {
        buffer,
        allocation: Some(allocation),
        size,
        device: device.clone(),
    })
}

fn load_shader_module(device: &ash::Device, shader_path: &Path) -> Result<vk::ShaderModule> {
    let spv_data = fs::read(shader_path)?;
    let spv = read_spv(&mut spv_data.as_slice())?;

    let create_info = vk::ShaderModuleCreateInfo::builder().code(spv.as_slice());

    unsafe { device.create_shader_module(&create_info, None) }
        .map_err(|e| anyhow::anyhow!("Failed to create shader module: {:?}", e))
}

fn main() -> Result<()> {
    env_logger::init();
    info!("Starting Candle Vulkan Demo");

    // Initialize Vulkan
    info!("Creating Vulkan context...");
    let ctx = VulkanContext::new()?;
    info!("✅ Vulkan context created");

    // Create allocator
    info!("Creating allocator...");
    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: ctx.instance.clone(),
            device: ctx.device.clone(),
            physical_device: ctx.physical_device,
            device_name: "Unknown".to_string(),
            preferred_device_type: gpu_allocator::vulkan::DeviceType::DiscreteGpu,
            allocation_sizes: gpu_allocator::vulkan::AllocationSizeStrategy::Optimal,
        })?;
    info!("✅ Allocator created");

    // Matrix dimensions
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;

    // Create input buffers
    info!("Creating input buffers...");
    let size = M * N * std::mem::size_of::<f32>();
    let mut buffer_a = create_buffer(
        &ctx.device,
        &mut allocator,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let mut buffer_b = create_buffer(
        &ctx.device,
        &mut allocator,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let mut buffer_c = create_buffer(
        &ctx.device,
        &mut allocator,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    // Initialize input data
    info!("Initializing input data...");
    let matrix_a: Vec<f32> = (0..M * N).map(|i| i as f32).collect();
    let matrix_b: Vec<f32> = (0..M * N).map(|i| (i * 2) as f32).collect();

    unsafe {
        buffer_a.write_data(&mut allocator, &matrix_a)?;
        buffer_b.write_data(&mut allocator, &matrix_b)?;
    }
    info!("✅ Input data written to GPU buffers");

    // Load shaders
    info!("Loading shaders...");
    let shader_dir = std::env::var("SHADER_DIR")?;
    let add_shader = load_shader_module(&ctx.device, &Path::new(&shader_dir).join("add.spv"))?;
    let matmul_shader =
        load_shader_module(&ctx.device, &Path::new(&shader_dir).join("matmul.spv"))?;
    info!("✅ Shaders loaded");

    // Create compute pipelines
    info!("Creating compute pipelines...");

    // Add pipeline
    let add_pipeline_layout = {
        let set_layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];

        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_layout_bindings);
        let set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&set_layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(12) // 3 * u32
            .offset(0);

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        (set_layout, layout)
    };

    // Matmul pipeline
    let matmul_pipeline_layout = {
        let set_layout_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];

        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_layout_bindings);
        let set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&set_layout_info, None)?
        };

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(18) // 6 * u32
            .offset(0);

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe { ctx.device.create_pipeline_layout(&layout_info, None)? };

        (set_layout, layout)
    };

    // Create add pipeline
    let add_pipeline = {
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(add_shader)
            .name(&CString::new("main")?);

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(add_pipeline_layout.1);

        unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )?
                .0[0]
        }
    };

    // Create matmul pipeline
    let matmul_pipeline = {
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(matmul_shader)
            .name(&CString::new("main")?);

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(matmul_pipeline_layout.1);

        unsafe {
            ctx.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )?
                .0[0]
        }
    };

    info!("✅ Compute pipelines created");

    // Test matrix addition
    info!("\n--- Testing Matrix Addition ---");
    {
        // Allocate descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(ctx.descriptor_pool)
            .set_layouts(std::slice::from_ref(&add_pipeline_layout.0));

        let descriptor_set = unsafe { ctx.device.allocate_descriptor_sets(&alloc_info)? }[0];

        // Update descriptor set
        let buffer_infos = [
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer_a.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build(),
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer_b.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build(),
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer_c.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build(),
        ];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0]))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1]))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2]))
                .build(),
        ];

        unsafe {
            ctx.device
                .update_descriptor_sets(&write_descriptor_sets, &[])
        };

        // Create command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };
        let command_buffer = command_buffers[0];

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            ctx.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        // Bind pipeline and descriptor set
        unsafe {
            ctx.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                add_pipeline,
            )
        };
        unsafe {
            ctx.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                add_pipeline_layout.1,
                0,
                &[descriptor_set],
                &[],
            )
        };

        // Set push constants
        let push_constants = [M as u32, N as u32, N as u32];
        unsafe {
            ctx.device.cmd_push_constants(
                command_buffer,
                add_pipeline_layout.1,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            )
        };

        // Dispatch
        unsafe {
            ctx.device
                .cmd_dispatch(command_buffer, ((M * N + 255) / 256) as u32, 1, 1)
        };

        // End command buffer
        unsafe { ctx.device.end_command_buffer(command_buffer)? };

        // Submit command buffer
        let submit_info =
            vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            ctx.device.queue_submit(
                ctx.compute_queue,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?
        };
        unsafe { ctx.device.queue_wait_idle(ctx.compute_queue)? };

        // Read results
        let mut result = vec![0.0f32; M * N];
        unsafe { buffer_c.read_data(&mut allocator, &mut result)? };

        // Verify results
        let mut errors = 0;
        for i in 0..M * N {
            let expected = matrix_a[i] + matrix_b[i];
            if (result[i] - expected).abs() > 1e-6 {
                errors += 1;
            }
        }

        info!(
            "Matrix addition test: {} errors out of {} elements",
            errors,
            M * N
        );
        if errors == 0 {
            info!("✅ Matrix addition test passed!");
        } else {
            error!("❌ Matrix addition test failed!");
        }
    }

    // Cleanup
    unsafe {
        ctx.device.destroy_pipeline(add_pipeline, None);
        ctx.device.destroy_pipeline(matmul_pipeline, None);
        ctx.device
            .destroy_pipeline_layout(add_pipeline_layout.1, None);
        ctx.device
            .destroy_pipeline_layout(matmul_pipeline_layout.1, None);
        ctx.device
            .destroy_descriptor_set_layout(add_pipeline_layout.0, None);
        ctx.device
            .destroy_descriptor_set_layout(matmul_pipeline_layout.0, None);
        ctx.device.destroy_shader_module(add_shader, None);
        ctx.device.destroy_shader_module(matmul_shader, None);
    }

    info!("Demo completed successfully!");
    Ok(())
}
