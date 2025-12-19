// Build script for candle-hip-flash-attn
// Compiles HIP kernels using ROCm's Composable Kernel (CK) FMHA implementation

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

const KERNEL_FILES: [&str; 1] = [
    "kernels/flash_fwd.hip",
];

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo::rerun-if-changed={kernel_file}");
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);

    // Get ROCm path
    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let rocm_include = format!("{}/include", rocm_path);

    // Get GPU architecture
    let gpu_arch = std::env::var("HIP_GPU_ARCH").unwrap_or_else(|_| {
        // Try to detect GPU architecture
        if let Ok(output) = Command::new("rocminfo").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("gfx") {
                    if let Some(arch) = line.split_whitespace().find(|s| s.starts_with("gfx")) {
                        return arch.to_string();
                    }
                }
            }
        }
        "gfx90a".to_string() // Default to MI200 series
    });

    println!("cargo:warning=Compiling HIP flash attention kernels for architecture: {}", gpu_arch);

    // Compile each kernel file
    for kernel_file in KERNEL_FILES.iter() {
        let kernel_path = PathBuf::from(kernel_file);
        let kernel_name = kernel_path.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}.o", kernel_name));

        let status = Command::new("hipcc")
            .args([
                "-c",
                "-fPIC",
                "-O3",
                "-std=c++17",
                "-I/usr/include/c++/13",
                "-I/usr/include/x86_64-linux-gnu/c++/13",
                &format!("-I{}", rocm_include),
                &format!("-I{}/ck", rocm_include),
                &format!("-I{}/ck_tile", rocm_include),
                &format!("--offload-arch={}", gpu_arch),
                "-D__HIP_PLATFORM_AMD__",
                "-DUSE_ROCM",
                "-o", obj_file.to_str().unwrap(),
                kernel_file,
            ])
            .status()
            .context("Failed to run hipcc")?;

        if !status.success() {
            anyhow::bail!("hipcc failed to compile {}", kernel_file);
        }
    }

    // Create static library
    let lib_file = out_dir.join("libhipflashattention.a");
    let obj_files: Vec<_> = KERNEL_FILES.iter()
        .map(|f| {
            let kernel_path = PathBuf::from(f);
            let kernel_name = kernel_path.file_stem().unwrap().to_str().unwrap();
            out_dir.join(format!("{}.o", kernel_name))
        })
        .collect();

    let mut ar_cmd = Command::new("ar");
    ar_cmd.args(["rcs", lib_file.to_str().unwrap()]);
    for obj in &obj_files {
        ar_cmd.arg(obj.to_str().unwrap());
    }

    let status = ar_cmd.status().context("Failed to run ar")?;
    if !status.success() {
        anyhow::bail!("ar failed to create static library");
    }

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=hipflashattention");
    println!("cargo::rustc-link-lib=dylib=amdhip64");
    println!("cargo::rustc-link-lib=dylib=stdc++");

    Ok(())
}
