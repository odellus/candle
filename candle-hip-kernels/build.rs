use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Post-process hipified files to fix known hipify-perl gaps
fn fix_hipify_gaps(src_dir: &Path) {
    // hipify-perl doesn't convert cuda_bf16.h -> hip/hip_bfloat16.h
    // This is a known limitation, see: https://github.com/ROCm/HIPIFY/issues/643
    for entry in fs::read_dir(src_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "hip" || ext == "h" {
                if let Ok(content) = fs::read_to_string(&path) {
                    let fixed = content
                        .replace(
                            "#include \"cuda_bf16.h\"",
                            "#include \"hip/hip_bfloat16.h\"",
                        )
                        .replace("#include <cuda_bf16.h>", "#include <hip/hip_bfloat16.h>")
                        .replace("__nv_bfloat16", "hip_bfloat16");
                    if fixed != content {
                        fs::write(&path, fixed).unwrap();
                    }
                }
            }
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/hip_compat.h");

    // Rerun if any kernel source changes
    let src_dir = Path::new("src");
    if src_dir.exists() {
        for entry in fs::read_dir(src_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "cpp" || ext == "hip" || ext == "h" {
                    println!("cargo::rerun-if-changed={}", path.display());
                }
            }
        }
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find HIP installation
    let rocm_path = env::var("ROCM_PATH")
        .or_else(|_| env::var("HIP_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    let hipcc = PathBuf::from(&rocm_path).join("bin/hipcc");

    if !hipcc.exists() {
        println!(
            "cargo::warning=hipcc not found at {}. Skipping HIP kernel compilation.",
            hipcc.display()
        );
        println!("cargo::warning=Set ROCM_PATH or HIP_PATH environment variable to your ROCm installation.");

        // Generate empty module file so the crate still compiles
        let kernels_path = out_dir.join("kernels.rs");
        fs::write(&kernels_path, generate_empty_kernels()).unwrap();
        return;
    }

    // Detect GPU architecture
    let gpu_arch = detect_gpu_arch(&rocm_path).unwrap_or_else(|| "gfx906".to_string());
    println!(
        "cargo::warning=Compiling HIP kernels for architecture: {}",
        gpu_arch
    );

    // Fix known hipify-perl gaps before compiling
    fix_hipify_gaps(src_dir);

    // Kernel source files to compile
    let kernel_sources = [
        "affine",
        "binary",
        "cast",
        "conv",
        "fill",
        "indexing",
        "quantized",
        "reduce",
        "sort",
        "ternary",
        "unary",
    ];

    let mut compiled_objects = Vec::new();
    let include_dir = src_dir.to_path_buf();

    for kernel in &kernel_sources {
        let src_file = src_dir.join(format!("{}.hip", kernel));
        if !src_file.exists() {
            println!(
                "cargo::warning=Kernel source {} not found, skipping",
                src_file.display()
            );
            continue;
        }

        let obj_file = out_dir.join(format!("{}.o", kernel));

        let mut args = vec![
            "-c".to_string(),
            "-fPIC".to_string(),
            "-O3".to_string(),
            "-std=c++17".to_string(),
            format!("--offload-arch={}", gpu_arch),
            "-I".to_string(),
            include_dir.to_str().unwrap().to_string(),
        ];

        // Add C++ standard library include paths (needed for some ROCm versions)
        for gcc_version in ["13", "12", "11", "14"] {
            let cpp_include = format!("/usr/include/c++/{}", gcc_version);
            let cpp_arch_include = format!("/usr/include/x86_64-linux-gnu/c++/{}", gcc_version);
            if Path::new(&cpp_include).exists() {
                args.push("-I".to_string());
                args.push(cpp_include);
                if Path::new(&cpp_arch_include).exists() {
                    args.push("-I".to_string());
                    args.push(cpp_arch_include);
                }
                break;
            }
        }

        args.extend([
            "-o".to_string(),
            obj_file.to_str().unwrap().to_string(),
            "-x".to_string(),
            "hip".to_string(),
            src_file.to_str().unwrap().to_string(),
        ]);

        let status = Command::new(&hipcc)
            .args(&args)
            .status()
            .expect("Failed to execute hipcc");

        if !status.success() {
            panic!("Failed to compile {}", src_file.display());
        }

        compiled_objects.push(obj_file);
    }

    if compiled_objects.is_empty() {
        println!("cargo::warning=No kernel sources found. Generating empty module.");
        let kernels_path = out_dir.join("kernels.rs");
        fs::write(&kernels_path, generate_empty_kernels()).unwrap();
        return;
    }

    // Create static library
    let lib_path = out_dir.join("libcandle_hip_kernels.a");
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

    let status = Command::new(&ar)
        .args(["rcs", lib_path.to_str().unwrap()])
        .args(compiled_objects.iter().map(|p| p.to_str().unwrap()))
        .status()
        .expect("Failed to execute ar");

    if !status.success() {
        panic!("Failed to create static library");
    }

    // Also generate HSACO (HIP equivalent of PTX) for runtime loading
    let mut hsaco_modules = Vec::new();
    for kernel in &kernel_sources {
        let src_file = src_dir.join(format!("{}.hip", kernel));
        if !src_file.exists() {
            continue;
        }

        let hsaco_file = out_dir.join(format!("{}.hsaco", kernel));

        let mut hsaco_args = vec![
            "-fPIC".to_string(),
            "-O3".to_string(),
            "-std=c++17".to_string(),
            "--genco".to_string(),
            format!("--offload-arch={}", gpu_arch),
            "-I".to_string(),
            include_dir.to_str().unwrap().to_string(),
        ];

        // Add C++ standard library include paths
        for gcc_version in ["13", "12", "11", "14"] {
            let cpp_include = format!("/usr/include/c++/{}", gcc_version);
            let cpp_arch_include = format!("/usr/include/x86_64-linux-gnu/c++/{}", gcc_version);
            if Path::new(&cpp_include).exists() {
                hsaco_args.push("-I".to_string());
                hsaco_args.push(cpp_include);
                if Path::new(&cpp_arch_include).exists() {
                    hsaco_args.push("-I".to_string());
                    hsaco_args.push(cpp_arch_include);
                }
                break;
            }
        }

        hsaco_args.extend([
            "-o".to_string(),
            hsaco_file.to_str().unwrap().to_string(),
            "-x".to_string(),
            "hip".to_string(),
            src_file.to_str().unwrap().to_string(),
        ]);

        let status = Command::new(&hipcc).args(&hsaco_args).status();

        if let Ok(status) = status {
            if status.success() {
                hsaco_modules.push((kernel.to_string(), hsaco_file));
            }
        }
    }

    // Generate Rust bindings with embedded HSACO
    let kernels_path = out_dir.join("kernels.rs");
    let kernels_code = generate_kernels_rs(&hsaco_modules);
    fs::write(&kernels_path, kernels_code).unwrap();

    // Link instructions
    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=candle_hip_kernels");

    // Link against HIP runtime
    let hip_lib_dir = PathBuf::from(&rocm_path).join("lib");
    println!("cargo::rustc-link-search=native={}", hip_lib_dir.display());
    println!("cargo::rustc-link-lib=dylib=amdhip64");
}

fn detect_gpu_arch(rocm_path: &str) -> Option<String> {
    // Try using rocminfo to detect GPU
    let rocminfo = PathBuf::from(rocm_path).join("bin/rocminfo");
    if rocminfo.exists() {
        if let Ok(output) = Command::new(&rocminfo).output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("gfx") {
                    // Extract gfxXXX from the line
                    for word in line.split_whitespace() {
                        if word.starts_with("gfx") {
                            return Some(word.to_string());
                        }
                    }
                }
            }
        }
    }

    // Check environment variable
    if let Ok(arch) = env::var("HIP_ARCH") {
        return Some(arch);
    }

    // Default based on common architectures
    None
}

fn generate_empty_kernels() -> String {
    r#"
// Auto-generated - no HIP kernels available
pub const ALL_IDS: [&str; 0] = [];

pub struct Module {
    name: &'static str,
}

impl Module {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn index(&self) -> usize {
        0
    }

    pub fn hsaco(&self) -> &'static [u8] {
        &[]
    }
}
"#
    .to_string()
}

fn generate_kernels_rs(hsaco_modules: &[(String, PathBuf)]) -> String {
    let mut code = String::new();

    code.push_str("// Auto-generated HIP kernel bindings\n\n");

    // Generate module constants
    let mut module_names = Vec::new();
    for (name, path) in hsaco_modules {
        let const_name = name.to_uppercase();
        module_names.push(const_name.clone());

        code.push_str(&format!(
            "pub const {}_HSACO: &[u8] = include_bytes!(\"{}\");\n",
            const_name,
            path.display()
        ));
    }

    code.push_str("\n");

    // Generate ALL_IDS array
    code.push_str(&format!(
        "pub const ALL_IDS: [&str; {}] = [\n",
        module_names.len()
    ));
    for name in &module_names {
        code.push_str(&format!("    \"{}\",\n", name.to_lowercase()));
    }
    code.push_str("];\n\n");

    // Generate Module struct and instances
    code.push_str(
        r#"
#[derive(Clone, Copy, Debug)]
pub struct Module {
    index: usize,
    name: &'static str,
    hsaco: &'static [u8],
}

impl Module {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn hsaco(&self) -> &'static [u8] {
        self.hsaco
    }
}

"#,
    );

    // Generate module constants
    for (i, (name, _)) in hsaco_modules.iter().enumerate() {
        let const_name = name.to_uppercase();
        code.push_str(&format!(
            "pub const {}: Module = Module {{ index: {}, name: \"{}\", hsaco: {}_HSACO }};\n",
            const_name, i, name, const_name
        ));
    }

    code
}
