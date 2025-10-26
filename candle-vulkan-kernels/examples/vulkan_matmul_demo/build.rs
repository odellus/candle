use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    // Get the OUT_DIR where we'll put compiled shaders
    let out_dir = env::var("OUT_DIR")?;
    let shaders_dir = Path::new("shaders");
    let dest_dir = Path::new(&out_dir).join("shaders");

    // Create destination directory
    fs::create_dir_all(&dest_dir)?;

    // Find all .comp shader files and compile them
    if shaders_dir.exists() {
        for entry in fs::read_dir(shaders_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("comp") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
                let output_path = dest_dir.format!("{}.spv", stem);

                println!("Compiling shader: {}", path.display());

                // Use shaderc to compile GLSL to SPIR-V
                let mut compiler = shaderc::Compiler::new()
                    .ok_or_else(|| "Failed to create shader compiler".to_string())?;

                let options = shaderc::CompileOptions::new()
                    .ok_or_else(|| "Failed to create compile options".to_string())?
                    .set_optimization_level(shaderc::OptimizationLevel::Performance);

                let result = compiler.compile_into_spirv(
                    &fs::read_to_string(&path)?,
                    shaderc::ShaderKind::Compute,
                    &path.to_string_lossy(),
                    "main",
                    Some(&options),
                )?;

                fs::write(&output_path, result.as_binary())?;
                println!("Compiled {} -> {}", stem, output_path.display());
            }
        }
    } else {
        println!("Warning: shaders directory not found, skipping shader compilation");
    }

    // Set shader directory for the binary
    println!("cargo:rustc-env=SHADER_DIR={}", out_dir);

    Ok(())
}
