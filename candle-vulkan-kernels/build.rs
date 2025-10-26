use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the OUT_DIR where we'll put compiled shaders
    let out_dir = env::var("OUT_DIR")?;
    let out_path = PathBuf::from(&out_dir);

    // Find all shader files in examples directory and copy them to output
    let src_shaders_dir = Path::new("examples/q4_0_demo/shaders");
    if src_shaders_dir.exists() {
        // Create output directory for compiled shaders
        let dest_shaders_dir = out_path.join("shaders");
        fs::create_dir_all(&dest_shaders_dir)?;

        // Copy each shader file from examples (since those already have working compiled shaders)
        for entry in fs::read_dir(src_shaders_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();

                // Copy the shader to output
                let dest_path = dest_shaders_dir.join(&file_name);
                fs::copy(&path, &dest_path)?;
                println!("Copied shader: {}", file_name);
            }
        }
    }

    println!("cargo:rerun-if-changed=examples/q4_0_demo/shaders/");

    // Make sure we can find the compiled shaders
    println!("cargo:rustc-env=SHADER_DIR={}", out_dir);

    Ok(())
}
