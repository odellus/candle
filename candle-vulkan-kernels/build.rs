//! Build script - compiles GLSL shaders to SPIR-V
//! Handles #include preprocessing like ggml-vulkan

use shaderc::{Compiler, CompileOptions, ShaderKind, IncludeType, ResolvedInclude, IncludeCallbackResult};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Shader variant configuration
struct ShaderVariant {
    name: &'static str,
    source: &'static str,
    defines: &'static [(&'static str, &'static str)],
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shader_out_dir = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&shader_out_dir).unwrap();

    let compiler = Compiler::new().expect("Failed to create shader compiler");

    // Our simple shaders (no includes)
    let simple_shaders = [
        ("matvec.comp", ShaderKind::Compute),
    ];

    for (shader_name, shader_kind) in simple_shaders {
        let shader_path = Path::new("src/shaders").join(shader_name);
        println!("cargo:rerun-if-changed={}", shader_path.display());

        let source = fs::read_to_string(&shader_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", shader_path.display(), e));

        let spirv = compiler
            .compile_into_spirv(&source, shader_kind, shader_name, "main", None)
            .unwrap_or_else(|e| panic!("Failed to compile {}: {}", shader_name, e));

        let output_name = shader_name.replace(".comp", ".spv");
        let output_path = shader_out_dir.join(&output_name);

        fs::write(&output_path, spirv.as_binary_u8())
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path.display(), e));

        println!("cargo:warning=Compiled {} -> {}", shader_name, output_name);
    }

    // GGML shaders with includes and variants
    let ggml_shader_dir = Path::new("src/shaders/ggml");

    // Track all ggml shader files for rebuild
    if let Ok(entries) = fs::read_dir(ggml_shader_dir) {
        for entry in entries.flatten() {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    // Define shader variants to compile
    // Each variant has a name, source file, and preprocessor defines
    let variants: Vec<ShaderVariant> = vec![
        // Unary ops - f32 contiguous (use generic_head.glsl - simple KX/KY params)
        ShaderVariant { name: "exp_f32", source: "exp.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float")] },
        ShaderVariant { name: "silu_f32", source: "silu.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float")] },
        ShaderVariant { name: "gelu_f32", source: "gelu.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float")] },
        ShaderVariant { name: "relu_f32", source: "relu.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float")] },

        // Unary ops - f32 strided (use generic_unary_head.glsl - full stride support)
        ShaderVariant { name: "exp_f32_strided", source: "exp_strided.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "silu_f32_strided", source: "silu_strided.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "gelu_f32_strided", source: "gelu_strided.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "relu_f32_strided", source: "relu_strided.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },

        // Other strided unary ops (already use generic_unary_head.glsl)
        ShaderVariant { name: "sqrt_f32", source: "sqrt.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "sin_f32", source: "sin.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "cos_f32", source: "cos.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "clamp_f32", source: "clamp.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },
        ShaderVariant { name: "scale_f32", source: "scale.comp", defines: &[("A_TYPE", "float"), ("D_TYPE", "float"), ("FLOAT_TYPE", "float")] },

        // Binary ops - f32 (already use generic_binary_head.glsl with full stride/broadcast support)
        ShaderVariant { name: "add_f32", source: "add.comp", defines: &[
            ("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"), ("QUANT_K", "1"), ("QUANT_R", "1"),
        ]},
        ShaderVariant { name: "mul_f32", source: "mul.comp", defines: &[
            ("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"), ("QUANT_K", "1"), ("QUANT_R", "1"),
        ]},
        ShaderVariant { name: "div_f32", source: "div.comp", defines: &[
            ("A_TYPE", "float"), ("B_TYPE", "float"), ("D_TYPE", "float"),
            ("FLOAT_TYPE", "float"), ("QUANT_K", "1"), ("QUANT_R", "1"),
        ]},

        // Copy (strided)
        ShaderVariant { name: "copy_f32", source: "copy.comp", defines: &[
            ("A_TYPE", "float"), ("D_TYPE", "float"),
        ]},

        // Dequantization shaders
        ShaderVariant { name: "dequant_q4_0_f32", source: "dequant_q4_0.comp", defines: &[
            ("D_TYPE", "float"),
        ]},
        ShaderVariant { name: "dequant_q8_0_f32", source: "dequant_q8_0.comp", defines: &[
            ("D_TYPE", "float"),
        ]},
    ];

    for variant in &variants {
        compile_ggml_shader(&compiler, ggml_shader_dir, &shader_out_dir, variant);
    }
}

fn compile_ggml_shader(
    compiler: &Compiler,
    shader_dir: &Path,
    out_dir: &Path,
    variant: &ShaderVariant,
) {
    let source_path = shader_dir.join(variant.source);

    let source = match fs::read_to_string(&source_path) {
        Ok(s) => s,
        Err(e) => {
            println!("cargo:warning=Skipping {}: {}", variant.name, e);
            return;
        }
    };

    let mut options = CompileOptions::new().expect("Failed to create compile options");

    // Set target environment to Vulkan 1.2
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);

    // Add preprocessor defines
    for (name, value) in variant.defines {
        options.add_macro_definition(name, Some(value));
    }

    // Set up include callback
    let shader_dir_owned = shader_dir.to_path_buf();
    options.set_include_callback(move |name, include_type, source_file, _depth| {
        handle_include(&shader_dir_owned, name, include_type, source_file)
    });

    let result = compiler.compile_into_spirv(
        &source,
        ShaderKind::Compute,
        variant.source,
        "main",
        Some(&options),
    );

    match result {
        Ok(spirv) => {
            let output_path = out_dir.join(format!("{}.spv", variant.name));
            fs::write(&output_path, spirv.as_binary_u8())
                .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path.display(), e));
            println!("cargo:warning=Compiled {} -> {}.spv", variant.source, variant.name);
        }
        Err(e) => {
            println!("cargo:warning=Failed to compile {}: {}", variant.name, e);
        }
    }
}

fn handle_include(
    shader_dir: &Path,
    name: &str,
    _include_type: IncludeType,
    _source_file: &str,
) -> IncludeCallbackResult {
    let include_path = shader_dir.join(name);

    match fs::read_to_string(&include_path) {
        Ok(content) => Ok(ResolvedInclude {
            resolved_name: include_path.to_string_lossy().into_owned(),
            content,
        }),
        Err(e) => Err(format!("Failed to include {}: {}", name, e)),
    }
}
