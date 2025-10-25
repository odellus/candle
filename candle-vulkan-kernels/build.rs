fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only build shaders in release mode to avoid recompilation during development
    if cfg!(debug_assertions) {
        println!("cargo:rerun-if-changed=src/shaders/");
        return Ok(());
    }

    let shaders = [
        "dequant_q4_0.comp",
        "mul_mat_vec.comp",
        // Add more shaders here as we implement them
    ];

    let compiler = shaderc::Compiler::new().ok_or("Failed to create shader compiler")?;

    for shader in shaders {
        let shader_path = format!("src/shaders/{}", shader);
        let output_path = format!("src/shaders/{}.spv", shader);

        let glsl_source = std::fs::read_to_string(&shader_path)?;
        let spirv = compiler.compile_into_spirv(
            &glsl_source,
            shaderc::ShaderKind::Compute,
            &shader_path,
            "main",
            None,
        )?;

        std::fs::write(&output_path, unsafe {
            std::slice::from_raw_parts(
                spirv.as_binary().as_ptr() as *const u8,
                spirv.as_binary().len() * std::mem::size_of::<u32>(),
            )
        })?;
        println!("Compiled {} to {}", shader_path, output_path);
    }

    Ok(())
}
